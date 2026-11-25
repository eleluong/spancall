"""Simple τ-bench Retail environment, task suite, and official τ-bench adapter.

Provides:
1. ``make_tau_retail_tools()``: faithful τ-bench retail tool definitions
   (scalar lookups + policy-sensitive write tools tagged with ``risk_tier="high"``
   and ``array`` parameters where appropriate).
2. ``SimpleTauRetailEnvironment``: stateful in-memory retail database with
   realistic users, orders, and products, plus final-DB-state verification.
3. ``build_simple_tau_retail_suite()``: 10 curated multi-turn retail tasks
   covering lookup -> inspect -> confirm -> mutate workflows.
4. ``TauBenchAgentAdapter``: drop-in adapter implementing the official
   ``tau_bench.agents.base.Agent`` ``solve(env, task_index)`` protocol when
   ``tau_bench`` is installed.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from spancall.agent.tools.base import BaseToolEnvironment, ToolExecutionResult
from spancall.data.schema import ParameterSchema, ToolDefinition


TAU_RETAIL_SYSTEM_PROMPT = (
    "You are a helpful retail customer service agent. "
    "First authenticate the user using their name and zip code or email, "
    "look up their user details and order details before taking any action, "
    "and ALWAYS ask the user for explicit confirmation before canceling, "
    "modifying, returning, or exchanging an order. "
    "Once the requested action is complete, reply clearly to the user."
)


def make_tau_retail_tools() -> List[ToolDefinition]:
    """Return τ-bench retail tool definitions (matching official retail schemas)."""
    return [
        ToolDefinition(
            name="find_user_id_by_name_zip",
            description="Find a user id by first name, last name, and zip code.",
            risk_tier="low",
            parameters=[
                ParameterSchema(name="first_name", description="User first name", param_type="string", required=True),
                ParameterSchema(name="last_name", description="User last name", param_type="string", required=True),
                ParameterSchema(name="zip", description="5-digit zip code", param_type="string", required=True),
            ],
        ),
        ToolDefinition(
            name="find_user_id_by_email",
            description="Find a user id by email address.",
            risk_tier="low",
            parameters=[
                ParameterSchema(name="email", description="User email address", param_type="string", required=True),
            ],
        ),
        ToolDefinition(
            name="get_user_details",
            description="Get full details of a user by user_id.",
            risk_tier="low",
            parameters=[
                ParameterSchema(name="user_id", description="Unique user identifier", param_type="string", required=True),
            ],
        ),
        ToolDefinition(
            name="get_order_details",
            description="Get status and items of an order by order_id.",
            risk_tier="low",
            parameters=[
                ParameterSchema(name="order_id", description="Order identifier such as #W1001", param_type="string", required=True),
            ],
        ),
        ToolDefinition(
            name="get_product_details",
            description="Get product variants and availability by product_id.",
            risk_tier="low",
            parameters=[
                ParameterSchema(name="product_id", description="Product identifier", param_type="string", required=True),
            ],
        ),
        ToolDefinition(
            name="cancel_pending_order",
            description="Cancel a pending order after explicit user confirmation.",
            risk_tier="high",
            parameters=[
                ParameterSchema(name="order_id", description="Order identifier", param_type="string", required=True),
                ParameterSchema(
                    name="reason",
                    description="Cancellation reason",
                    param_type="string",
                    required=True,
                    enum=["no longer needed", "ordered by mistake"],
                ),
            ],
        ),
        ToolDefinition(
            name="modify_pending_order_address",
            description="Update the shipping address of a pending order after confirmation.",
            risk_tier="high",
            parameters=[
                ParameterSchema(name="order_id", description="Order identifier", param_type="string", required=True),
                ParameterSchema(name="address1", description="Primary street address", param_type="string", required=True),
                ParameterSchema(name="address2", description="Secondary address line", param_type="string", required=False),
                ParameterSchema(name="city", description="City name", param_type="string", required=True),
                ParameterSchema(name="state", description="Two-letter state code", param_type="string", required=True),
                ParameterSchema(name="country", description="Country name", param_type="string", required=True),
                ParameterSchema(name="zip", description="5-digit zip code", param_type="string", required=True),
            ],
        ),
        ToolDefinition(
            name="modify_pending_order_items",
            description="Modify items in a pending order after explicit user confirmation.",
            risk_tier="high",
            parameters=[
                ParameterSchema(name="order_id", description="Order identifier", param_type="string", required=True),
                ParameterSchema(name="item_ids", description="List of current item ids to replace", param_type="array", items_type="string", required=True),
                ParameterSchema(name="new_item_ids", description="List of replacement item ids", param_type="array", items_type="string", required=True),
                ParameterSchema(name="payment_method_id", description="Payment method id for any price difference", param_type="string", required=True),
            ],
        ),
        ToolDefinition(
            name="return_delivered_order_items",
            description="Return items from a delivered order after explicit user confirmation.",
            risk_tier="high",
            parameters=[
                ParameterSchema(name="order_id", description="Order identifier", param_type="string", required=True),
                ParameterSchema(name="item_ids", description="List of item ids to return", param_type="array", items_type="string", required=True),
                ParameterSchema(name="payment_method_id", description="Refund payment method id", param_type="string", required=True),
            ],
        ),
    ]


def _default_retail_db() -> Dict[str, Any]:
    """Deterministic in-memory retail DB modeled on τ-bench retail."""
    return {
        "users": {
            "yusuf_rossi_9620": {
                "user_id": "yusuf_rossi_9620",
                "name": {"first_name": "Yusuf", "last_name": "Rossi"},
                "email": "yusuf.rossi@example.com",
                "address": {
                    "address1": "763 Broadway",
                    "address2": "Suite 135",
                    "city": "New York",
                    "state": "NY",
                    "country": "USA",
                    "zip": "10003",
                },
                "payment_methods": ["credit_card_9513926"],
                "orders": ["#W1001", "#W1002"],
            },
            "mei_chen_4412": {
                "user_id": "mei_chen_4412",
                "name": {"first_name": "Mei", "last_name": "Chen"},
                "email": "mei.chen@example.com",
                "address": {
                    "address1": "412 Market St",
                    "address2": "",
                    "city": "San Francisco",
                    "state": "CA",
                    "country": "USA",
                    "zip": "94105",
                },
                "payment_methods": ["gift_card_4412001", "credit_card_4412009"],
                "orders": ["#W2001", "#W2002"],
            },
            "alex_rivera_7801": {
                "user_id": "alex_rivera_7801",
                "name": {"first_name": "Alex", "last_name": "Rivera"},
                "email": "alex.rivera@example.com",
                "address": {
                    "address1": "901 Congress Ave",
                    "address2": "Apt 4B",
                    "city": "Austin",
                    "state": "TX",
                    "country": "USA",
                    "zip": "78701",
                },
                "payment_methods": ["paypal_7801555"],
                "orders": ["#W3001"],
            },
        },
        "orders": {
            "#W1001": {
                "order_id": "#W1001",
                "user_id": "yusuf_rossi_9620",
                "status": "pending",
                "address": {
                    "address1": "763 Broadway",
                    "address2": "Suite 135",
                    "city": "New York",
                    "state": "NY",
                    "country": "USA",
                    "zip": "10003",
                },
                "items": [
                    {"item_id": "item_101_blue_m", "product_id": "prod_shirt_101", "name": "Cotton Shirt", "price": 45.0},
                ],
                "payment_method_id": "credit_card_9513926",
            },
            "#W1002": {
                "order_id": "#W1002",
                "user_id": "yusuf_rossi_9620",
                "status": "delivered",
                "address": {
                    "address1": "763 Broadway",
                    "address2": "Suite 135",
                    "city": "New York",
                    "state": "NY",
                    "country": "USA",
                    "zip": "10003",
                },
                "items": [
                    {"item_id": "item_202_black", "product_id": "prod_mug_202", "name": "Insulated Mug", "price": 28.0},
                ],
                "payment_method_id": "credit_card_9513926",
            },
            "#W2001": {
                "order_id": "#W2001",
                "user_id": "mei_chen_4412",
                "status": "pending",
                "address": {
                    "address1": "412 Market St",
                    "address2": "",
                    "city": "San Francisco",
                    "state": "CA",
                    "country": "USA",
                    "zip": "94105",
                },
                "items": [
                    {"item_id": "item_301_s", "product_id": "prod_jacket_301", "name": "Rain Jacket", "price": 120.0},
                ],
                "payment_method_id": "credit_card_4412009",
            },
            "#W2002": {
                "order_id": "#W2002",
                "user_id": "mei_chen_4412",
                "status": "delivered",
                "address": {
                    "address1": "412 Market St",
                    "address2": "",
                    "city": "San Francisco",
                    "state": "CA",
                    "country": "USA",
                    "zip": "94105",
                },
                "items": [
                    {"item_id": "item_401_white", "product_id": "prod_earbuds_401", "name": "Wireless Earbuds", "price": 89.0},
                ],
                "payment_method_id": "gift_card_4412001",
            },
            "#W3001": {
                "order_id": "#W3001",
                "user_id": "alex_rivera_7801",
                "status": "pending",
                "address": {
                    "address1": "901 Congress Ave",
                    "address2": "Apt 4B",
                    "city": "Austin",
                    "state": "TX",
                    "country": "USA",
                    "zip": "78701",
                },
                "items": [
                    {"item_id": "item_501_10", "product_id": "prod_shoes_501", "name": "Running Shoes", "price": 110.0},
                ],
                "payment_method_id": "paypal_7801555",
            },
        },
        "products": {
            "prod_shirt_101": {
                "product_id": "prod_shirt_101",
                "name": "Cotton Shirt",
                "variants": [
                    {"item_id": "item_101_blue_m", "color": "blue", "size": "M", "available": True, "price": 45.0},
                    {"item_id": "item_101_blue_l", "color": "blue", "size": "L", "available": True, "price": 45.0},
                    {"item_id": "item_101_red_m", "color": "red", "size": "M", "available": False, "price": 45.0},
                ],
            },
            "prod_jacket_301": {
                "product_id": "prod_jacket_301",
                "name": "Rain Jacket",
                "variants": [
                    {"item_id": "item_301_s", "size": "S", "available": True, "price": 120.0},
                    {"item_id": "item_301_m", "size": "M", "available": True, "price": 120.0},
                ],
            },
        },
    }


class SimpleTauRetailEnvironment(BaseToolEnvironment):
    """Stateful retail environment matching τ-bench retail semantics."""

    def __init__(
        self,
        goal_checker: Optional[Callable[[Dict[str, Any], List[Dict[str, Any]]], bool]] = None,
        initial_db: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._initial_db = copy.deepcopy(initial_db) if initial_db is not None else _default_retail_db()
        self._goal_checker = goal_checker
        self.db: Dict[str, Any] = {}
        self.call_history: List[Dict[str, Any]] = []
        self._tools = make_tau_retail_tools()
        self.reset()

    def get_available_tools(self) -> List[ToolDefinition]:
        return list(self._tools)

    def reset(self) -> None:
        self.db = copy.deepcopy(self._initial_db)
        self.call_history = []

    def evaluate_task_success(self) -> bool:
        if self._goal_checker is None:
            return len(self.call_history) > 0
        return bool(self._goal_checker(self.db, self.call_history))

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        args = dict(arguments or {})
        self.call_history.append({"tool_name": tool_name, "arguments": args})

        if tool_name == "find_user_id_by_name_zip":
            fn = str(args.get("first_name", "")).strip().lower()
            ln = str(args.get("last_name", "")).strip().lower()
            zp = str(args.get("zip", "")).strip()
            for uid, u in self.db["users"].items():
                if (
                    u["name"]["first_name"].lower() == fn
                    and u["name"]["last_name"].lower() == ln
                    and str(u["address"]["zip"]) == zp
                ):
                    return ToolExecutionResult(success=True, output={"user_id": uid})
            return ToolExecutionResult(success=False, output={"error": "User not found"}, error="user_not_found")

        if tool_name == "find_user_id_by_email":
            em = str(args.get("email", "")).strip().lower()
            for uid, u in self.db["users"].items():
                if u["email"].lower() == em:
                    return ToolExecutionResult(success=True, output={"user_id": uid})
            return ToolExecutionResult(success=False, output={"error": "User not found"}, error="user_not_found")

        if tool_name == "get_user_details":
            uid = str(args.get("user_id", ""))
            user = self.db["users"].get(uid)
            if not user:
                return ToolExecutionResult(success=False, output={"error": f"Unknown user_id {uid}"}, error="unknown_user")
            return ToolExecutionResult(success=True, output=copy.deepcopy(user))

        if tool_name == "get_order_details":
            oid = str(args.get("order_id", ""))
            order = self.db["orders"].get(oid)
            if not order:
                return ToolExecutionResult(success=False, output={"error": f"Unknown order_id {oid}"}, error="unknown_order")
            return ToolExecutionResult(success=True, output=copy.deepcopy(order))

        if tool_name == "get_product_details":
            pid = str(args.get("product_id", ""))
            prod = self.db["products"].get(pid)
            if not prod:
                return ToolExecutionResult(success=False, output={"error": f"Unknown product_id {pid}"}, error="unknown_product")
            return ToolExecutionResult(success=True, output=copy.deepcopy(prod))

        if tool_name == "cancel_pending_order":
            oid = str(args.get("order_id", ""))
            reason = str(args.get("reason", ""))
            order = self.db["orders"].get(oid)
            if not order:
                return ToolExecutionResult(success=False, output={"error": f"Unknown order_id {oid}"}, error="unknown_order")
            if order["status"] != "pending":
                return ToolExecutionResult(success=False, output={"error": "Only pending orders can be cancelled"}, error="not_pending")
            if reason not in ("no longer needed", "ordered by mistake"):
                return ToolExecutionResult(success=False, output={"error": f"Invalid reason: {reason}"}, error="invalid_reason")
            order["status"] = "cancelled"
            order["cancel_reason"] = reason
            return ToolExecutionResult(success=True, output=copy.deepcopy(order), state_changed=True)

        if tool_name == "modify_pending_order_address":
            oid = str(args.get("order_id", ""))
            order = self.db["orders"].get(oid)
            if not order:
                return ToolExecutionResult(success=False, output={"error": f"Unknown order_id {oid}"}, error="unknown_order")
            if order["status"] != "pending":
                return ToolExecutionResult(success=False, output={"error": "Only pending orders can be modified"}, error="not_pending")
            for k in ("address1", "address2", "city", "state", "country", "zip"):
                if k in args and args[k] is not None:
                    order["address"][k] = str(args[k])
            return ToolExecutionResult(success=True, output=copy.deepcopy(order), state_changed=True)

        if tool_name == "modify_pending_order_items":
            oid = str(args.get("order_id", ""))
            order = self.db["orders"].get(oid)
            if not order:
                return ToolExecutionResult(success=False, output={"error": f"Unknown order_id {oid}"}, error="unknown_order")
            if order["status"] != "pending":
                return ToolExecutionResult(success=False, output={"error": "Only pending orders can be modified"}, error="not_pending")
            old_ids = list(args.get("item_ids") or [])
            new_ids = list(args.get("new_item_ids") or [])
            if not old_ids or len(old_ids) != len(new_ids):
                return ToolExecutionResult(success=False, output={"error": "item_ids and new_item_ids must be non-empty and same length"}, error="bad_items")
            mapping = dict(zip(old_ids, new_ids))
            for it in order["items"]:
                if it["item_id"] in mapping:
                    it["item_id"] = mapping[it["item_id"]]
            return ToolExecutionResult(success=True, output=copy.deepcopy(order), state_changed=True)

        if tool_name == "return_delivered_order_items":
            oid = str(args.get("order_id", ""))
            order = self.db["orders"].get(oid)
            if not order:
                return ToolExecutionResult(success=False, output={"error": f"Unknown order_id {oid}"}, error="unknown_order")
            if order["status"] != "delivered":
                return ToolExecutionResult(success=False, output={"error": "Only delivered orders can be returned"}, error="not_delivered")
            item_ids = list(args.get("item_ids") or [])
            if not item_ids:
                return ToolExecutionResult(success=False, output={"error": "item_ids required"}, error="bad_items")
            order["status"] = "return_requested"
            order["returned_item_ids"] = item_ids
            order["refund_payment_method_id"] = str(args.get("payment_method_id", ""))
            return ToolExecutionResult(success=True, output=copy.deepcopy(order), state_changed=True)

        return ToolExecutionResult(success=False, output={"error": f"Unknown tool: {tool_name}"}, error="unknown_tool")


class ScriptedUserSimulator:
    """Deterministic multi-turn user simulator for Simple τ-bench tasks.

    Replies with scripted follow-ups (e.g., confirmation or missing details)
    when the agent asks a question, and emits ``###STOP###`` once all scripted
    replies are exhausted.
    """

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self._idx = 0

    def __call__(self, agent_text: str, messages: List[Dict[str, Any]]) -> Optional[str]:
        if self._idx < len(self._replies):
            reply = self._replies[self._idx]
            self._idx += 1
            return reply
        return "###STOP###"


@dataclass
class TauRetailTask:
    task_id: str
    instruction: str
    user_replies: List[str]
    goal_checker: Callable[[Dict[str, Any], List[Dict[str, Any]]], bool]
    category: str = "multi_turn"


def _called(history: List[Dict[str, Any]], tool_name: str, **expected_args: Any) -> bool:
    for c in history:
        if c["tool_name"] != tool_name:
            continue
        if all(c["arguments"].get(k) == v for k, v in expected_args.items()):
            return True
    return False


def build_simple_tau_retail_suite() -> List[TauRetailTask]:
    """10 curated Simple τ-bench Retail tasks (scalar lookup -> chaining -> mutation)."""
    return [
        TauRetailTask(
            task_id="tau_retail_01_lookup_order_status",
            category="lookup_chain",
            instruction=(
                "Hi, I'm Yusuf Rossi (zip 10003). Can you check the status of my order #W1001?"
            ),
            user_replies=[],
            goal_checker=lambda db, h: (
                _called(h, "find_user_id_by_name_zip", first_name="Yusuf", last_name="Rossi", zip="10003")
                and _called(h, "get_order_details", order_id="#W1001")
            ),
        ),
        TauRetailTask(
            task_id="tau_retail_02_lookup_by_email_user_details",
            category="lookup_chain",
            instruction=(
                "Please look up my account using my email mei.chen@example.com and tell me my recent orders."
            ),
            user_replies=[],
            goal_checker=lambda db, h: (
                _called(h, "find_user_id_by_email", email="mei.chen@example.com")
                and _called(h, "get_user_details", user_id="mei_chen_4412")
            ),
        ),
        TauRetailTask(
            task_id="tau_retail_03_cancel_pending_order",
            category="mutation_confirm",
            instruction=(
                "Hi, my email is yusuf.rossi@example.com. I ordered #W1001 by mistake and want to cancel it."
            ),
            user_replies=["Yes, please go ahead and cancel order #W1001."],
            goal_checker=lambda db, h: (
                db["orders"]["#W1001"]["status"] == "cancelled"
                and db["orders"]["#W1001"].get("cancel_reason") == "ordered by mistake"
            ),
        ),
        TauRetailTask(
            task_id="tau_retail_04_cancel_no_longer_needed",
            category="mutation_confirm",
            instruction=(
                "I'm Alex Rivera in zip 78701. Please cancel my pending order #W3001 because it is no longer needed."
            ),
            user_replies=["Yes, I confirm cancelling #W3001."],
            goal_checker=lambda db, h: (
                db["orders"]["#W3001"]["status"] == "cancelled"
                and db["orders"]["#W3001"].get("cancel_reason") == "no longer needed"
            ),
        ),
        TauRetailTask(
            task_id="tau_retail_05_modify_shipping_address",
            category="mutation_confirm",
            instruction=(
                "Hi, I'm Mei Chen (zip 94105). I need to change the shipping address on pending order #W2001 "
                "to 888 Mission St, San Francisco, CA, USA, 94103."
            ),
            user_replies=["Yes, please update the shipping address."],
            goal_checker=lambda db, h: (
                db["orders"]["#W2001"]["address"]["address1"] == "888 Mission St"
                and db["orders"]["#W2001"]["address"]["zip"] == "94103"
            ),
        ),
        TauRetailTask(
            task_id="tau_retail_06_modify_order_item_size",
            category="complex_array_mutation",
            instruction=(
                "Hi, my email is yusuf.rossi@example.com. On pending order #W1001, can you change my Cotton Shirt "
                "(item_101_blue_m) to size L? Check product prod_shirt_101 first."
            ),
            user_replies=["Yes, please switch item_101_blue_m to item_101_blue_l using credit_card_9513926."],
            goal_checker=lambda db, h: (
                any(it["item_id"] == "item_101_blue_l" for it in db["orders"]["#W1001"]["items"])
            ),
        ),
        TauRetailTask(
            task_id="tau_retail_07_return_delivered_item",
            category="complex_array_mutation",
            instruction=(
                "Hi, I'm Mei Chen (mei.chen@example.com). I'd like to return the Wireless Earbuds (item_401_white) "
                "from delivered order #W2002 back to gift_card_4412001."
            ),
            user_replies=["Yes, please submit the return for item_401_white."],
            goal_checker=lambda db, h: (
                db["orders"]["#W2002"]["status"] == "return_requested"
                and "item_401_white" in (db["orders"]["#W2002"].get("returned_item_ids") or [])
            ),
        ),
        TauRetailTask(
            task_id="tau_retail_08_cannot_cancel_delivered",
            category="policy_refusal",
            instruction=(
                "I'm Yusuf Rossi (zip 10003). Can you cancel order #W1002? I no longer need it."
            ),
            user_replies=[],
            goal_checker=lambda db, h: (
                _called(h, "get_order_details", order_id="#W1002")
                and not _called(h, "cancel_pending_order")
                and db["orders"]["#W1002"]["status"] == "delivered"
            ),
        ),
        TauRetailTask(
            task_id="tau_retail_09_product_availability_check",
            category="lookup_chain",
            instruction=(
                "Can you check if the Rain Jacket (prod_jacket_301) is available in size M?"
            ),
            user_replies=[],
            goal_checker=lambda db, h: _called(h, "get_product_details", product_id="prod_jacket_301"),
        ),
        TauRetailTask(
            task_id="tau_retail_10_multi_step_find_and_return",
            category="complex_array_mutation",
            instruction=(
                "Hi, I'm Yusuf Rossi (zip 10003). I want to return the Insulated Mug from my delivered order #W1002 "
                "to my credit card on file."
            ),
            user_replies=["Yes, please process the return."],
            goal_checker=lambda db, h: (
                db["orders"]["#W1002"]["status"] == "return_requested"
                and "item_202_black" in (db["orders"]["#W1002"].get("returned_item_ids") or [])
            ),
        ),
    ]


# --------------------------------------------------------------------------- #
# Official tau-bench Agent adapter (works when `tau_bench` is installed)
# --------------------------------------------------------------------------- #
class TauBenchAgentAdapter:
    """Adapter implementing ``tau_bench.agents.base.Agent`` using ``ToolCallRouter``.

    When official ``tau_bench`` is installed (``pip install git+https://github.com/sierra-research/tau-bench.git``),
    this adapter wraps any ``ToolCallRouter`` (``LLM_ONLY``, ``ROUTER_SCHEMA``,
    ``HYBRID_FILL_SCHEMA``, ``SHADOW``) and runs inside ``env.reset(task_index)`` /
    ``env.step(Action(...))`` while recording per-turn ``StepTrace`` telemetry.
    """

    def __init__(
        self,
        router: Any,
        tools_info: Sequence[Dict[str, Any]],
        wiki: str,
        max_steps: int = 30,
        max_context_tokens: int = 16384,
        k_recent: int = 32,
        context_budget_tokens: int = 2048,
        prevent_duplicate_calls: bool = True,
    ) -> None:
        from spancall.combine import coerce_tools

        self.router = router
        self.tools = coerce_tools(tools_info)
        self.wiki = wiki
        self.max_steps = int(max_steps)
        self.max_context_tokens = int(max_context_tokens)
        self.k_recent = int(k_recent)
        self.context_budget_tokens = int(context_budget_tokens)
        self.prevent_duplicate_calls = bool(prevent_duplicate_calls)
        self.last_traces: List[Any] = []

    def solve(self, env: Any, task_index: Optional[int] = None, max_num_steps: Optional[int] = None) -> Any:
        """Run one official τ-bench episode and return its SolveResult."""
        from spancall.agent.context import (
            latest_user_utterance,
            rows_from_messages,
            select_recent_context,
            trim_messages,
        )
        from spancall.agent.router import _canonical_call_key

        try:
            from tau_bench.types import Action, RESPOND_ACTION_NAME, SolveResult  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Official tau_bench package is not installed. "
                "Install it or use SimpleTauRetailEnvironment for local evaluation."
            ) from exc

        steps_cap = int(max_num_steps or self.max_steps)
        reset_res = env.reset(task_index=task_index)
        obs = reset_res.observation
        info = reset_res.info.model_dump() if hasattr(reset_res.info, "model_dump") else dict(reset_res.info)
        reward = 0.0

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        executed_calls = set()
        self.last_traces = []

        for step_idx in range(steps_cap):
            messages, trimmed = trim_messages(messages, max_context_tokens=self.max_context_tokens)
            utt = latest_user_utterance(messages, fallback=obs)
            traj_rows = rows_from_messages(messages, exclude_last_user=True)
            recent_rows, recent_feats = select_recent_context(
                traj_rows,
                tools=self.tools,
                k_recent=self.k_recent,
                context_budget_tokens=self.context_budget_tokens,
            )
            result, step_trace = self.router.route_turn(
                utterance=utt,
                messages=messages,
                available_tools=self.tools,
                session_context=recent_rows,
                executed_calls=executed_calls if self.prevent_duplicate_calls else None,
            )
            if step_trace is not None:
                step_trace.turn_index = step_idx
                step_trace.context_trimmed = trimmed
                step_trace.spancall_features.update(recent_feats)
                self.last_traces.append(step_trace)

            if result.get("hit"):
                tool_name = str(result["tool_name"])
                arguments = dict(result.get("arguments") or {})
                executed_calls.add(_canonical_call_key(tool_name, arguments))
                call_id = str(result.get("call_id") or f"call_tau_{step_idx}")
                messages.append(self.router.synthesize_injected_call(tool_name, arguments, call_id))
                action = Action(name=tool_name, kwargs=arguments)
                env_res = env.step(action)
                messages.append(self.router.synthesize_tool_response(tool_name, env_res.observation, call_id))
            else:
                text = str(result.get("text") or "How else can I assist you?")
                messages.append({"role": "assistant", "content": text})
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": text})
                env_res = env.step(action)
                messages.append({"role": "user", "content": env_res.observation})

            reward = float(env_res.reward)
            info = env_res.info.model_dump() if hasattr(env_res.info, "model_dump") else dict(env_res.info)
            if env_res.done:
                break

        return SolveResult(reward=reward, info=info, messages=messages, total_cost=0.0)
