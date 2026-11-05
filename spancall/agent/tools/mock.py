"""Deterministic mock tool environments for agent-loop tests and ablation runs.

These environments model real stateful tool surfaces (e.g. an airline
booking domain) with fixed inventories so that agent/eval runs are fully
reproducible. They implement the full :class:`BaseToolEnvironment` contract
including ``session_context`` (the recent-state feed for the SpanCall fast
path) and ``evaluate_task_success``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from spancall.agent.tools.base import BaseToolEnvironment, ToolExecutionResult
from spancall.data.schema import ParameterSchema, ToolDefinition


def make_booking_tools() -> List[ToolDefinition]:
    """The deterministic airline-booking tool suite (used by the mock env)."""
    return [
        ToolDefinition(
            name="search_flights",
            description="Search for available flights by origin, destination and date.",
            risk_tier="read_only",
            parameters=[
                ParameterSchema(name="origin", description="Departure airport code (IATA).", param_type="string"),
                ParameterSchema(name="destination", description="Arrival airport code (IATA).", param_type="string"),
                ParameterSchema(name="date", description="Travel date (ISO YYYY-MM-DD).", param_type="string"),
                ParameterSchema(name="max_price", description="Maximum price filter.", param_type="number", required=False),
            ],
        ),
        ToolDefinition(
            name="book_flight",
            description="Book a flight by its flight id.",
            risk_tier="side_effecting",
            parameters=[
                ParameterSchema(name="flight_id", description="The flight id returned by search_flights.", param_type="string"),
                ParameterSchema(
                    name="seat_class",
                    description="Cabin class to book.",
                    param_type="string",
                    required=False,
                    enum=["economy", "premium", "business"],
                ),
            ],
        ),
        ToolDefinition(
            name="cancel_booking",
            description="Cancel a previously created booking by reference.",
            risk_tier="side_effecting",
            parameters=[
                ParameterSchema(name="booking_ref", description="Booking reference (e.g. BR-1001).", param_type="string"),
            ],
        ),
    ]


class MockBookingEnvironment(BaseToolEnvironment):
    """Deterministic airline-booking environment.

    State:
        ``FLIGHTS`` — a fixed 3-flight inventory.
        ``bookings`` — list of booked flight records.
        ``_last_search`` — the latest search output (fed to
            ``session_context`` so the resolver can disambiguate later turns).
        ``required_flight_id`` — when set, ``evaluate_task_success`` is true
            once that flight has been booked (the canonical multi-turn task).
    """

    FLIGHTS: Dict[str, Dict[str, Any]] = {
        "F001": {"flight_id": "F001", "origin": "SFO", "destination": "JFK", "date": "2026-11-05", "price": 199.0},
        "F002": {"flight_id": "F002", "origin": "SFO", "destination": "LAX", "date": "2026-11-05", "price": 89.0},
        "F003": {"flight_id": "F003", "origin": "JFK", "destination": "LHR", "date": "2026-11-06", "price": 649.0},
    }

    def __init__(
        self,
        required_flight_id: Optional[str] = None,
        booking_enabled: bool = True,
    ) -> None:
        self.required_flight_id = required_flight_id
        self.booking_enabled = booking_enabled
        self.bookings: List[Dict[str, Any]] = []
        self._last_search: List[Dict[str, Any]] = []
        self.reset()

    # ------------------------------------------------------------------ #
    # BaseToolEnvironment contract
    # ------------------------------------------------------------------ #
    def get_available_tools(self) -> List[ToolDefinition]:
        return make_booking_tools()

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        fn = getattr(self, f"_exec_{tool_name}", None)
        if fn is None:
            return ToolExecutionResult(success=False, output=None, error=f"unknown tool: {tool_name}")
        try:
            return fn(arguments)
        except (TypeError, ValueError) as exc:
            return ToolExecutionResult(success=False, output=None, error=str(exc))

    def reset(self) -> None:
        self.bookings = []
        self._last_search = []

    def evaluate_task_success(self) -> bool:
        if not self.required_flight_id:
            return False
        return any(b.get("flight_id") == self.required_flight_id for b in self.bookings)

    def session_context(self) -> List[Dict[str, Any]]:
        """Recent session rows: last search results + latest booking ref."""
        rows: List[Dict[str, Any]] = list(self._last_search)
        if self.bookings:
            rows.append(
                {
                    "key": "booking_ref",
                    "value": str(self.bookings[-1].get("booking_ref", "")),
                    "source": "tool:book_flight",
                }
            )
        return rows

    # ------------------------------------------------------------------ #
    # Tool implementations
    # ------------------------------------------------------------------ #
    def _exec_search_flights(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        origin = arguments.get("origin")
        destination = arguments.get("destination")
        date = arguments.get("date")
        missing = [k for k, v in {"origin": origin, "destination": destination, "date": date}.items() if not v]
        if missing:
            return ToolExecutionResult(
                success=False, output=None, error=f"missing required argument(s): {', '.join(missing)}"
            )
        max_price = arguments.get("max_price")
        flights = [
            dict(f) for f in self.FLIGHTS.values()
            if f["origin"] == origin and f["destination"] == destination and f["date"] == date
        ]
        if max_price is not None:
            flights = [f for f in flights if f["price"] <= float(max_price)]
        self._last_search = [
            {"key": "flight_count", "value": str(len(flights)), "source": "tool:search_flights"}
        ] + [{"key": "flight_id", "value": f["flight_id"], "source": "tool:search_flights"} for f in flights]
        return ToolExecutionResult(success=True, output={"flights": flights}, state_changed=False)

    def _exec_book_flight(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        flight_id = arguments.get("flight_id")
        if not flight_id:
            return ToolExecutionResult(success=False, output=None, error="missing required argument: flight_id")
        if flight_id not in self.FLIGHTS:
            return ToolExecutionResult(success=False, output=None, error=f"unknown flight id: {flight_id}")
        existing = [b for b in self.bookings if b["flight_id"] == flight_id]
        if existing:
            # Bookings are idempotent: re-booking the same flight returns the
            # existing reference instead of creating a duplicate.
            return ToolExecutionResult(
                success=True, output={"booking_ref": existing[0]["booking_ref"], "already_booked": True}
            )
        if not self.booking_enabled:
            return ToolExecutionResult(success=False, output=None, error="bookings are disabled")
        seat_class = arguments.get("seat_class") or "economy"
        booking = {
            "booking_ref": f"BR-{1001 + len(self.bookings)}",
            "flight_id": flight_id,
            "seat_class": seat_class,
            "price": self.FLIGHTS[flight_id]["price"],
        }
        self.bookings.append(booking)
        return ToolExecutionResult(
            success=True, output={"booking_ref": booking["booking_ref"]}, state_changed=True
        )

    def _exec_cancel_booking(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        ref = arguments.get("booking_ref")
        if not ref:
            return ToolExecutionResult(success=False, output=None, error="missing required argument: booking_ref")
        before = len(self.bookings)
        self.bookings = [b for b in self.bookings if b["booking_ref"] != ref]
        if len(self.bookings) == before:
            return ToolExecutionResult(success=False, output=None, error=f"booking not found: {ref}")
        return ToolExecutionResult(
            success=True, output={"cancelled": ref, "remaining": len(self.bookings)}, state_changed=True
        )