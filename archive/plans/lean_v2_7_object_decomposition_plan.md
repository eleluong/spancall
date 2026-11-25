# Lean v2.7 — Object-Param Decomposition (Dotted Pseudo-Params)

**Status:** Proposed
**Scope:** Decompose `object`-typed params into scalar dotted pseudo-params end-to-end; **arrays remain untouched** (no decode changes, no metric changes in this plan).
**Motivation:** v2.6 precision analysis — object values currently fall through `_align_value` Case E (scalar string) and fail to `OFF_WINDOW` as `unresolved`, hurting coverage; when partially matched they leak junk. Decomposition turns each object leaf into the "easy question" (a single scalar span selection) the lean flow is good at.

---

## 1. Design

### 1.1 Schema-level flattening (data sources + schema)

- Extend JSON-Schema parsing in `spancall/data/sources/hermes.py` (`_tool_from_json_schema`) and `spancall/data/sources/xlam.py` to recurse into nested `properties`:
  - `{"type": "object", "properties": {"color": {"type": "string"}, "size": {"type": "integer"}}}` for param `filters` → two `ParameterSchema` leaves: `filters.color` (string), `filters.size` (integer), each inheriting `required`/`enum`/`default`/`description` from the leaf spec.
  - Object depth cap: **3 levels** (`filters.a.b.c`); deeper or free-form objects (`additionalProperties`, no `properties`) stay a single `object` param → routed to `unresolved` explicitly (Case G below).
  - Arrays of objects (`items.type == "object"`): leave unresolved (out of scope v2.7).
- Store the nesting map in `ParameterSchema`-adjacent metadata: a per-tool `object_paths: {tool: {"filters": ["filters.color", "filters.size"]}}` on `ToolDefinition` (new optional field, default `None` — backward compatible with every existing tool).

### 1.2 Aligner (spancall/data/align.py)

- **New Case G before Case E:** if `p.param_type == "object"` and the value is a dict with no decomposition (free-form), return `self._fail(p, "OBJECT_UNDECOMPOSED")` — an explicit, countable drop reason instead of the current silent scalar-path failure.
- Decomposed leaves never reach the object branch: GT arguments `{"filters": {"color": "red"}}` are **pre-flattened before alignment** by a small `flatten_gt_args` step in the aligner entrypoint: `{"filters.color": "red", "filters.size": 10}`. Leaf alignment then reuses Cases B–E unchanged.
- Targets carry the dotted `param_name`; nothing else in `SpanTarget` changes.

### 1.3 Template / parser (span_template/)

- **No grammar change.** `PARAM_RE = r"<\|param\|>\s+([^\s(]+)"` already accepts dotted names; `<|none|>` and R1–R9 are unaffected. Add a template test asserting dotted names render/parse round-trip.
- **Multi-tool namespace fix (required):** the gate payload key `f"{t.name}.{p.name}"` (`template.py:630`) becomes ambiguous with dotted params (`tool.a.b`). Switch the multi-tool payload namespace separator to `::` (matching the resolver mask-key convention) or quote-escape; keep single-tool keys bare. Mirror this in `parse_template` payload reconstruction.

### 1.4 Tensorize / loss (spancall/lean/tensorize.py, loss.py)

- No changes: leaves are ordinary params with ordinary `(B, K, C)` slots. `param_drop_reasons` gains `OBJECT_UNDECOMPOSED` as a fifth *expected* (not defect) reason.
- Cost note: each leaf adds one H3 slot + one `<|param|>` line — within existing budgets for depth ≤ 3.

### 1.5 Resolver reassembly (spancall/lean/resolver.py)

- After decode, before `StrictTypeCaster`/`SchemaValidator`/gates: reassemble dotted leaves into nested dicts using the tool's `object_paths` map (`unflatten_payload`). Leaf-level `cast_failures` / `missing_required` report the dotted name; whole-call gating stays all-or-nothing on the *root* object param (a required object missing any required leaf → abstain, exactly like today's scalar behavior).
- `parity_spans` keys already use `tool::param`; dotted names flow through `_mask_key` unchanged — extend `_parity_spans_for_call` (eval_p1) to emit dotted keys for flattened GT.

### 1.6 Eval (spancall/lean/eval_p1.py, eval/toolcall/)

- `_deep_equal` already compares nested dicts; reassembled payloads compare naturally against original GT. No metric changes.
- `classify_eval_cohort` / `_gt_params` operate on schema params — with flattened schemas both sides use dotted names; add a synthetic test proving E2E exact-match on a decomposed-object sample.
- Report a new coverage stat: `object_params_decomposed` vs `object_params_unresolved` per split (expected: coverage win).

## 2. Execution steps

1. `flatten`/`unflatten` helpers + `object_paths` on `ToolDefinition` (schema.py) with unit tests.
2. Hermes + xLAM schema parsing for nested `properties` (depth ≤ 3) with fixture tests.
3. Aligner Case G (`OBJECT_UNDECOMPOSED`) + GT arg flattening; aligner tests.
4. Template multi-tool namespace separator fix + round-trip test for dotted names (guard: `tests/test_notebook_cell_guard.py` scope check may need the new helper names whitelisted).
5. Resolver `unflatten_payload` before cast/validate; parity key extension.
6. Eval coverage stat + E2E synthetic test.
7. Re-run `scripts/realign_v2.py` (template stamp changes if namespace separator changes → new `data/processed_v2_7/`), then re-run the v2.6 notebook flow (new `scripts/build_lean_v2_7_colab_notebook.py` derived from the v2.6 builder) to measure: exact_F, fast-path coverage delta on object-bearing samples, and STEP2B_POINTER_RANK_MISS delta.

## 3. Risks

- **Template stamp churn:** changing the multi-tool payload separator invalidates stamps → full realign + retrain (expected, budgeted).
- **Bleed into tensorize counts:** more params per tool → longer renders; verify `render_T` histogram shift and C=512 truncation rate stays within current envelope.
- **Free-form objects remain unresolved** — that is by design (retrieval-first contract); the plan converts only schema-declared leaves.

## 4. Out of scope (explicit)

- Array decode changes (element-set, list-size head) — deferred.
- Order-insensitive list comparison in `payload_equal` — deferred.
- Per-param refusal (T6) — deferred.
