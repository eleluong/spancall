# Research & System Proposal: **SpanCall** (Revised — Speculative Fast Path for Tool-Calling)

**Title:** *SpanCall: A Lightweight High-Confidence Fast Path for Context-Ready Tool Calls, with Full Abstention to Normal LLM Tool-Calling*

---

## 0. Claim Ledger (Read This First)

This proposal makes **narrow, measurable claims** and explicitly disclaims the rest.

**We claim:**
- For a bounded but high-frequency class of tool calls — where *every* argument is already present, extractable, and context-ready (verbatim entities, literals, booleans, schema enums, literal dates) — a small (~150M-param) bidirectional encoder + parameterized virtual context can emit the **complete payload** in one forward pass (1 decoding step) with **whole-call precision ≥ 95% on held-out paraphrases** (a *target* calibrated in §3.4, not an assumption), at an *expected* order-of-magnitude lower FLOPs than a 7B autoregressive decode (a hypothesis; the measured FLOPs/latency table is a Phase 4 deliverable, §6.2).
- When SpanCall is *not* high-confidence, it **abstains entirely** and the agent performs its **normal, unchanged LLM tool call**. There is no partial payload, no per-parameter fill, no hybrid. Abstention is the safe default; a hit is never forced.
- **Agentic compatibility is by construction, not an add-on:** the router sits *before* LLM tool-call generation. Easy, context-ready calls short-circuit the LLM; everything else flows through the agent's existing, untouched pipeline.

**We explicitly do NOT claim:**
- Zero hallucination, or ~95% of *all* calls hitting the fast path. Wrong-value errors (pointing at the *wrong mentioned entity*) can and will occur — we measure them and calibrate abstention against them ([§6.2](#62-key-metrics-honest-set)). Coverage depends on traffic; the router is instrumented from day one (§8).
- Solving the canonicalization problem in general. Paraphrase→canonical mapping is *learned semantic behavior*, bounded to a class we can enumerate in `[CTX_ENUM]` / `[CTX_TIME]`. Unlisted paraphrases abstain; the LLM handles them.
- Strict O(1) total cost. Encoder attention is O(L²); the pointer head is a boundary-factorized O(L) scorer (§3.3). "O(1)" refers only to *decoding steps* (= 1) vs. autoregressive M steps.

---

## 1. Executive Summary

Mainstream tool-calling relies on autoregressive LLMs generating JSON token-by-token: flexible, but slow (25–35 sequential steps per call), compute-heavy, and prone to inventing enum values or altering entities.

Real production traffic (search filters, CRM, booking, IoT, ERP) is heavily skewed: **a large majority of calls reuse a small set of schemas and a smaller set of phrase patterns** — verbatim entities, "direct/non-stop", "next week", "business class", "today/tomorrow", literal numbers. These are the cases where extractive, span-based prediction is *structurally* safer and faster than generation.

**SpanCall** is a small (~150M-param) bidirectional encoder that acts as a **speculative router**:

1. A **deterministic Virtual Context Pool** injects pre-resolved constants — anchored dates, booleans, schema enums — next to the user query.
2. Parameter values become **span-selection operations** against the query + context pool, scored by a parallel boundary-factorized pointer head in a **single forward pass**.
3. A **whole-payload confidence gate** emits the tool call **only if every required parameter resolves at ≥ τ**. Otherwise it **abstains entirely** and the agent proceeds with its **normal LLM tool call** — unchanged.

**Bottom line:** SpanCall does not replace the agent or the LLM. It accelerates the *easy majority* of tool calls at ~10× lower cost per call and p50 latency in tens of ms, while *never* producing a partial or fabricated payload: any uncertainty → full abstention → the normal pipeline (R7 sanitization is enforced in the reference implementation, §3.1.1). Success = **measured Hit Rate × Precision-at-Hit** on real traffic, not a theoretically perfect extractor.

---

## 2. Problem Statement & Motivation

Four concrete inefficiencies of autoregressive tool-calling, as relevant to the *common case*:

1. **Autoregressive Bottleneck:** `{"origin": "Hanoi", "date": "2026-10-24", "seat_class": "BUSINESS"}` requires 25–35 sequential decoding steps, most spent on `{`, `"`, `:`, `,` boilerplate.
2. **Value & Enum Hallucination:** Generative models invent enum values or distort entities unless wrapped in grammar-constrained decoding (Outlines, XGrammar), which further slows throughput.
3. **The Canonicalization Problem — real, but bounded:** extractive QA was historically dismissed because it could only highlight verbatim words ("tomorrow", not `2026-10-24`; nothing for `true` when the user says "non-stop"). We partially solve this *for enumerated classes* by injecting the canonical values themselves (dates, booleans, enums) into context. We do **not** claim to canonicalize unbounded values; those abstain.
4. **All-or-Nothing Execution:** Autoregressive JSON cannot be parsed until the final bracket is emitted; there is no cheap "partial answer with confidence."

**Hypothesis (narrow, testable):** For a bounded class of tool calls — every argument verbatim, boolean, enum, or literal-date in context — a bidirectional encoder with a parameterized virtual context emits the **complete payload** at **≥ 95% precision on held-out paraphrases** in 1 pass, with **whole-call abstention** covering the rest (≥ τ per parameter is required to hit). Precision and coverage are *measured against real traffic*, not assumed.

---

## 3. Architecture: SpanCall (Fast Path)

Three stages: **1) Deterministic Virtual Context Injection**, **2) Joint Bidirectional Encoding**, **3) Boundary-Factorized Multi-Span Scoring** — followed by a **whole-payload confidence gate** (§3.4).

### 3.1 Input Formulation: The SpanCall Template

SpanCall consumes **one flat text sequence** assembled by a **deterministic Jinja template** — a chat template, *inverted*. A chat template serializes a conversation for a generative model to *continue*; the SpanCall template serializes (schema, context pool, utterance) for an encoder to *point at*, and the template's own **marker grammar** makes the extraction legend **derivable from the text itself** — no external annotation, no string-regex heuristics, no drift between doc and code. Same deterministic function in → same text + same legend out.

This is *the* interface contract. The rest of the system (training alignment §5, inference mapping §3.3, evaluation §6) is defined against this template and the reference implementation `span_template/` (Python pkg, mirrors `apply_chat_template`).

#### 3.1.1 The template (spec) — single source of truth

The normative copy is [`span_template/template.jinja`](../span_template/template.jinja); the block below is its rendering contract. Keep them in sync — [`test_template.py`](../span_template/test_template.py) asserts the rendered segments, the entry/offset contract, marker-injection abstention (R7), and time-window/enum/quoting behavior; a byte-exact render-vs-spec diff fixture (mirroring `RENDERING_CONTRACT` in [`template.py`](../span_template/template.py)) is a pending test. Optional segments (`[CTX_TIME]`, `[CTX_ENUM]`, `[CTX_RECENT]`) are omitted when empty; `[SEP]` precedes `[CTX_RECENT]` only when it is present.

```jinja
[CLS]
[TOOL] {{ tool.name }}{% if tool.description %} ({{ tool.description }}){% endif %}
[PARAM] {{ p.name }}{% if p.description %} ({{ p.description }}){% endif %}   {# one line per param #}
[SEP]
[CTX_META] [NULL] | true | false | [UNRESOLVED]
[CTX_TIME] {{ key }}={{ iso }} | ...          {# if time_window #}
[CTX_ENUM] {{ enums | join(' | ') }}          {# if enums #}
[SEP]
{{ utterance }}                                {# sanitized, R7 #}
[SEP]                                          {# if recent_context #}
[CTX_RECENT] {{ key }}={{ quote(value) }} | ...  {# if recent_context #}
[EOS]
```

**Marker grammar (the legend is *in* the text):**

| Marker | What it brackets | Type |
| --- | --- | --- |
| `[TOOL] … (…)` | the candidate tool schema | schema anchor (bridge, not target) |
| `[PARAM] name (…)` | one parameter → its hidden state is the question vector `q_k` | schema anchor |
| `[CTX_META] [NULL] \| true \| false \| [UNRESOLVED]` | meta virtual values | **targets** (value only) |
| `[CTX_TIME] k=iso` | deterministic date pool | key = **bridge**, `iso` = **target** |
| `[CTX_ENUM] v1 \| v2 \| …` | union of schema enum values | **targets** (value only) |
| ` {{ utterance }} ` | current user message | **targets** (all tokens) |
| `[CTX_RECENT] key="value"` | recent context (recent turns, last tool outputs) | key = **bridge**, quoted value = **target** |

**Quoting rule (R1):** recent-context values are always emitted as `"escaped-string"` (`\` → `\\`, `"` → `\"`), so char→entry boundaries are unambiguous — a quoted value may contain ` | ` without being split. Keys are plain identifiers (`[A-Za-z_][A-Za-z0-9_.-]*`), never quoted. (See the `quote`/`_unescape` pair in `span_template/template.py`; unquoted enum entries are the one place the separator still needs rejection, R5.)

**Targetable region (R2):** Ω = CTX value tokens ∪ enum values ∪ utterance tokens ∪ quoted recent-context value interiors. Everything else — segment markers, ` | ` separators, `(`, `)`, `[PARAM]`/`[TOOL]` tokens, keys — is a **bridge** (attention path, never a target). This is what turns "spelled word → canonical value" and "mention → memory row" into learnable attention, not regex.

**Whole-call gating (R3):** SpanCall emits a tool call **only if every required parameter resolves at ≥ τ**. Any parameter that is optional-and-absent → `[NULL]` → omitted (a declared JSON Schema `default` is applied deterministically by the post-processor, `pipeline_io.md` Q6 — the encoder never invents one); any *required* parameter that abstains → **the whole call abstains** → normal LLM tool call. There is no partial payload. (`List[T]` values are multiple pointer targets scoring ≥ τ — never serialized list literals.) *Required vs. optional comes from the JSON schema and is enforced by the post-processor, not by the encoder*; the `(…)` description may mention it as a hint only. A param is `[NULL]` only if the schema marks it optional; a required param pointing at `[NULL]` ⇒ abstain.

**Time window (R4):** bounded, configurable — `today`, weekday names (`next_mon`, `this_fri`, …), `+k days` for k ∈ {1,2,3,7} → `today`, `+1d`, `+2d`, `+3d`, `+7d` plus weekday entries and a closed idiom map (`tomorrow`=`+1d`, `day_after_tomorrow`=`+2d`, `tonight`=today, date-params only), ~10–16 dated tokens, < 0.5ms to build (D1, remediation_plan.md). Off-window phrases ("in 3 months", "Q1 2027", "next week") are **not targetable** ⇒ call abstains ⇒ LLM. **Collisions:** different keys may share a date (e.g. `+3d`, `next_mon`, `tomorrow` can collide). Keys are never merged; instead targets are compared by *canonical value* — span scores of equal-valued entries are aggregated (log-sum-exp) at inference, and **all** equal-valued entries are positives in training (§5). Weekday key names are a closed vocabulary (`this_<dow>`, `next_<dow>`) fixed by the builder. *Reference-implementation status:* [`_weekday`](../span_template/template.py) now implements the exact semantics — `this_<dow>` is 0–6 days ahead (on its own weekday it resolves to *today*), `next_<dow>` is strictly future (1–7 days ahead) — and +k offset keys parse in full (`+1d`, not truncated). The clock is anchored to a passed/recorded date; the caller supplies the user-local date (`local_today`, D3), and per-session override is allowed — the template never guesses a timezone.

**Enum pool (R5):** union of every parameter's `enum` values, deduped, ` | `-joined; empty pool ⇒ the `[CTX_ENUM]` line is omitted. **Multi-word enum values** (internal whitespace) are emitted quoted (D2, remediation_plan.md) so they stay single atomic entries and never fragment the pool: the [`_enum_token`](../span_template/template.py) filter quotes on demand, and the parser tokenizer (`_ENUM_QUOTED_RE` in [`parser.py`](../span_template/parser.py)) maps the quoted token back to the canonical value. ` | ` inside any value is still rejected at ingestion (quoting protects whitespace, not the separator), and leading/trailing whitespace in an enum value ⇒ `ValueError` before rendering ([`validate_enums`](../span_template/template.py)).

**Recent context (R6):** optional segment — last N assistant tool outputs and session constants, injected verbatim. Only *quoted values* are targetable (R1). Budget: rolling window, configurable eviction — unbounded memory and cache invalidation are tracked open items (§8). Max sequence length L is derived from the backbone (default 32768 tokens; ModernBERT was 8k); inputs that would overflow after eviction **abstain** rather than truncate schema or context silently.

**Untrusted-text sanitization (R7):** the utterance and every recent-context value are *untrusted*. Because the legend is parsed back from rendered text, a user typing `[SEP]`, `[CTX_ENUM] …`, `\n[PARAM]` or a newline could forge structure. The renderer therefore (a) **rejects/abstains** on any utterance containing a reserved marker (`[CLS] [SEP] [EOS] [TOOL] [PARAM] [NONE] [CTX_*] [NULL] [UNRESOLVED]`, plus the P6 atomic-delimiter tokens `[BAR]`/`[EQ]`/`[Q]` — the authoritative list lives in one constant, [`MARKER_TOKENS`](../span_template/sanitize.py)), and (b) collapses newlines in the utterance to a single space, so the one-line-per-segment parse is unambiguous. This is a template-level guarantee and is independent of model behavior. **Status:** enforced in the reference implementation — [`sanitize_utterance`](../span_template/sanitize.py) runs P0/P1 (NFKC detection copy, reserved-marker rejection on message *and* recent-context rows, control-char stripping, whitespace collapse with `norm2orig`) before rendering; any violation raises `SanitizeError(marker_injection|empty|too_long)` ⇒ abstain (proposal §0, §8). Both `build_input` and `build_input_multi` run it, and the multi-marker/fullwidth adversarial fixtures live in [`test_template.py`](../span_template/test_template.py).

**Entry-snapped spans (R8):** CTX values (dates, enums, meta) are *atomic*: a span must cover a whole entry, never a sub-token fragment (`2026-10` is not a valid date target). The pointer head scores at *entry* granularity for CTX entries (pooled over the entry's subword tokens) and at token-span granularity only for utterance and recent-context interiors.

**Per-parameter validity masks (R9):** `[CTX_ENUM]` is the union over all params, so the post-processor masks each parameter's candidates to *its own* allowed enum values, to its type (date params ↔ `[CTX_TIME]`; bool params ↔ `true`/`false`), and to schema constraints. A candidate outside the mask is invalid → abstain, regardless of score.

#### 3.1.2 The legend is parsed from the text

`build_input(...)` renders the Jinja template **and** parses the marker grammar to produce both:

```python
SpanTemplateOutput {
  text:    str,      # exactly what the encoder consumes
  entries: [Entry],  # parsed from text: {char_start, char_end, value,
                     #  segment, selectable, kind, key}
  # selectable == True  ⇔  entry lies in Ω
  # kind ∈ {param_anchor, bridge_text, enum, date, bool, null, unresolved, string, key}
}
```

`selectable == True` ⇔ the entry's char span lies in Ω. Training derives positive spans directly from `entries` (§5 reverse alignment); inference masks the pointer head to `selectable` spans only (§3.3). Both sides consume the *same* parser, so the legend can never diverge from what the encoder sees.

#### 3.1.3 Worked instantiation — agentic turn 2 ("book that flight")

**View A — the rendered template (what the encoder sees):**

```text
[CLS]
[TOOL] book_flight (Book a selected flight)
[PARAM] flight_id (Selected flight identifier)
[PARAM] passenger_name (Full name on reservation)
[PARAM] email (Notification email address)
[PARAM] class (Service class; optional)
[SEP]
[CTX_META] [NULL] | true | false | [UNRESOLVED]
[CTX_TIME] today=2026-10-23 | +1d=2026-10-24 | +3d=2026-10-26 | +7d=2026-10-30 | next_mon=2026-10-26   ← +3d and next_mon collide; handled by R4
[SEP]
Book that flight, send the receipt to my work email
[SEP]
[CTX_RECENT] picked="VN123" | user.name="Anh Luong" | user.email="anh.luong@corp.io"
[EOS]
```

**View B — the parsed legend (what the pointer head resolves):**

| Param | Value | Source region | Legend locator | Route |
| --- | --- | --- | --- | --- |
| `flight_id` | `VN123` | Recent context | `segment="recent"`, `key="picked"`, `selectable=True` | fast — "that flight" anaphor-bridges to the memory row; zero synthesis (§3.4) |
| `passenger_name` | `Anh Luong` | Recent context | `segment="recent"`, `key="user.name"`, `selectable=True` | fast |
| `email` | `anh.luong@corp.io` | Recent context | `segment="recent"`, `key="user.email"`, `selectable=True` | fast — "my work email" → session row |
| `class` | *omitted* | CTX meta pool | `[NULL]` | default — schema marks `class` optional (R3), silently defaulted |

Both views are produced by the **same** deterministic `build_input` call; View B is parsed from View A.

> Anaphora beyond verbatim-value chaining ("book the *other* flight") is **not** claimed — the call abstains and the LLM does it (§4, §8).

### 3.2 Joint Bidirectional Encoder

X of length L passes through a pretrained bidirectional backbone (DeBERTa-v3-base or ModernBERT-base, ~150M params):

$$H = \text{Encoder}(X) \in \mathbb{R}^{L \times d}$$

Full bidirectional self-attention lets "tomorrow" attend to `tomorrow=2026-10-24`, "direct" to `true`, "business class" to `BUSINESS`. **This is learned paraphrase matching**, not keyword lookup — we do not pretend otherwise. Generalization to unseen phrasing is measured explicitly (§6).

### 3.3 Boundary-Factorized Pointer Head (Linear Span Cost)

We use a **GlobalPointer-style factorization** (not a full biaffine span MLP) to keep the head cost O(L), not O(L·W_max):

1. **Parameter Representation:** hidden state at the `[PARAM]` token for parameter k, projected once:

$$q_k = \text{MLP}_{\text{param}}(H_{\text{idx}(p_k)}) \in \mathbb{R}^{d}$$

2. **Boundary Scores:** start logits and end logits per position:

$$s_{k,i} = q_k^\top \text{MLP}_{\text{start}}(H_i), \qquad e_{k,j} = q_k^\top \text{MLP}_{\text{end}}(H_j)$$

3. **Span Score:** aggregate across boundaries (Rotary-relative position bias), evaluated only on spans in the targetable region Ω with width ≤ W_max:

$$M(k, i, j) = \text{combine}(s_{k,i}, e_{k,j}, \text{pos}(j-i))$$

4. **Deterministic Mapping:**
- **Abstain (per-parameter):** if score for `[UNRESOLVED]` wins, or the max *valid* (R9-masked) span score < τ. `[UNRESOLVED]` is a *learned* abstention (trained on unalignable rows); τ is a *calibrated* abstention on top of it. Both must pass: `[UNRESOLVED]` is evaluated first, and τ is calibrated only on the examples where it did not win. The margin to the runner-up *distinct* value is also required (≥ δ) to catch origin↔destination swaps.
- **Single-value:** argmax over Ω → decode substring → type-cast (int/float/bool/ISO-date). Ambiguous casts ("five", "around 500") abstain.
- **Multi-value (`List[T]`):** the list-size head (H6, `heads.md` §H6) predicts `n`; select the top-`n` **non-overlapping** spans by score, tie-break left-to-right (deterministic, documented — `pipeline_io.md` Q2 step 5).

### 3.4 Whole-Call Confidence Gate (All-or-Nothing)

The gate is the **whole point**:

- **Hit:** every *required* parameter resolved at ≥ τ (optional+absent → `[NULL]` → default) and the post-processor's schema validation passes (types, enums, cross-param constraints) ⇒ emit the complete payload.
- **Any required parameter abstains, or validation fails** ⇒ **abstain the entire call** ⇒ the agent performs its **normal, unchanged LLM tool call**. No partial payload, no hybrid fill, no per-parameter LLM round-trip.
- **Calibration is at whole-call level.** Per-parameter errors compound (four params at 98% ≈ 92% call precision), so τ (and δ) are fit on a held-out set to hit the *call-level* Precision-at-Hit target, using a call confidence such as `min` over required params. Per-schema-class thresholds fall back to a global τ for unseen schemas (needed for the zero-shot claim, §6.1).
- **Risk tiers:** tools are tagged `read_only` / `idempotent` / `side_effecting` (booking, payment, delete). Tiers only change the calibrated τ (stricter for side-effecting); no tool is LLM-only — the LLM is called only when confidence is low.

```text
               ┌─ hit (every req ≥ τ) ──► emit full payload  (fast path, ~10-20ms)
 utterance ──► │
               └─ abstain (any req < τ) ──► normal LLM tool call (unchanged agent pipeline)
```

> Rationale vs. the earlier full-biaffine draft: MLP-per-span costs ~L·W_max·3d² ≈ 7–10× encoder FLOPs. Boundary factorization keeps head cost ≈ linear in L. FLOPs table is part of the Phase 4 deliverables.

---

## 4. Placement in the Agent Loop: Speculate Before the LLM

SpanCall does not replace the agent or the LLM. It sits **in front of** the LLM's tool-call generation:

- **Loop position:** when the agent (any framework) is about to emit a tool call for a candidate schema, SpanCall runs first (~10–20ms, one batched pass if several candidate schemas). It emits the payload **only if fully confident** (all required args ≥ τ). Otherwise the agent falls through to its **normal LLM tool call** — the pipeline is untouched.
- **Fast path:** extractive, contextual, and recent-context-resident calls → one forward pass. Does not require the LLM to have planned anything differently: it is a drop-in accelerator.
- **Abstention is the safe default:** a miss costs ~10–20ms + the normal LLM call; a false hit costs a wrong tool call (calibrated to ≤ 5% by τ). The router is instrumented from day one (§8).
- **Batching for multi-call turns:** when the planner emits several candidate calls in one step, all schemas are scored in a single batched forward pass.

This division of labor is the actual value proposition: cheap+precise for the easy majority, unashamed LLM for the tail — with **zero risk of a partial or fabricated payload** anywhere in the system: R7 sanitization is enforced in the reference implementation (§3.1.1), and unverifiable or ambiguous extractions abstain to the normal pipeline.

---

## 5. Training & Silver Data Pipeline

Convert existing JSON tool-calling datasets (xLAM-60k, Glaive FC v2, ToolBench) into span supervision:

1. **Build Virtual Context:** populate `[CTX_ENUM]` from schema; `[CTX_META]` with booleans/NULL/UNRESOLVED; `[CTX_TIME]` anchored to a clock that is **independent of the ground-truth value** — the example's recorded date when present, otherwise a *synthetic clock sampled per example* (datasets such as xLAM/Glaive carry none). Anchoring to GT is leakage and is explicitly forbidden; a leakage test (shuffle the anchor, check accuracy drops only for date params) is part of Phase 1.
2. **Reverse Substring Alignment** (parses the same marker grammar as §3.1): null/omitted → `[NULL]` **only if the schema marks the param optional** (a value equal to its default but explicitly stated in U aligns to the span, not `[NULL]`); booleans → `true`/`false`; enum/date-in-context → value token(s), with *all* equal-valued entries as positives (R4); verbatim entity/list in U → span(s); else → `[UNRESOLVED]`. Align the *whole payload*, not per-parameter in isolation. A row is kept for the fast-path set only if *every* required param aligns; rows with any `[UNRESOLVED]` required param are kept as **abstention examples** for the call-level gate.
3. **Hard negatives:** synthesize entity-swap variants (origin↔destination, sender↔recipient) and marker-injection variants (R7) so wrong-value and injection behavior are trained and tested, not hoped for.
4. **Noise handling:** the `[UNRESOLVED]` class is a small, skewed minority — we use multi-positive softmax cross-entropy over `A_k ∪ {∅}` (`mechanism.md` §8.2) with abstention rows up-weighted, because silent mis-extraction is our worst failure mode; we want the model *eager to abstain* when uncertain. A Class-Balanced Focal + Multi-Label Circle variant is kept as an ablation for `array` params.

$$\mathcal{L}_k = -\log\frac{\sum_{c\in P_k}\exp M(k,c)}{\exp M(k,\varnothing)+\sum_{c'\in A_k}\exp M(k,c')}$$

**Data provenance note:** xLAM-60k is a repackaged mix and Glaive v2 provenance is unclear — license compliance review before shipping.

---

## 6. Evaluation Plan & Metrics

### 6.1 Benchmarks
1. **BFCL Simple / Multiple / Relevance** — zero-shot schema generalization (with the caveat: schema *selection* is out of scope; we evaluate resolution given a candidate schema).
2. **SNIPS & MultiWOZ 2.2 (converted)** — slot accuracy at high speed.
3. **Internal OOD-Paraphrase Suite** — the load-bearing eval: 2k held-out utterances written by engineers *not* in the training loop, exercising unseen ways to say the same 30 intents (this is where the "learned paraphrase matching" claim is actually tested).
4. **Multi-Turn Agentic Loop (BFCL multi-turn, tau-bench — converted)** — task-level success and per-call fast-path metrics measured *inside* a ReAct-style harness with the router placed before LLM tool-call generation.

### 6.2 Key Metrics (honest set)
- **Hit Rate × Precision-at-Hit (the primary pair).** Per *call*: fraction of calls the fast path emits (Hit Rate) and the fraction of those hits whose payload is correct (Precision-at-Hit). τ is calibrated to hold **Precision-at-Hit ≥ 95%**, then Hit Rate at that operating point is reported. This replaces the untestable "0.0% hallucination".
- **Wrong-Value Error Rate** — proportion of hits that point to the wrong *mentioned* value (e.g., destination → origin). This is the metric that actually matters and is invisible to "fabrication" definitions; we measure it explicitly.
- **Coverage by Pattern Class** — per pattern: verbatim entities / booleans / enums / literal dates / recent-context values / other. Never an aggregate a dominant class can pad.
- **Abstention Accuracy** — of the calls that abstained, what fraction would have been wrong if emitted (i.e., abstention precision); ensures the gate isn't silently emitting garbage it labels "fast."
- **Agent Task Success Rate** — end-to-end task completion in the loop harness **with vs. without** the fast path (same planner, same LLM), so the win is proven; mixed latency (hit-rate × fast-latency + miss-rate × LLM-latency) reported at loop level.
- **Latency** — fast-path p50/p99 GPU and CPU, plus LLM baseline; FLOPs + params table vs. Qwen-2.5-7B / Llama-3.1-8B / GLiNER (all measured, none asserted).
- **Baselines (fairness):** (1) regex/rules + the same virtual context; (2) a small (~0.5B) generator fine-tuned on the same data with grammar-constrained decoding; (3) GLiNER-style span extraction + deterministic mapper. The "order-of-magnitude" claim is only evaluated against (2), not just a 7B model.
- **Downstream cost of wrong hits** — per risk tier (§3.4): retry/correction rate and side-effects caused, not just call accuracy.
- **Adversarial suite** — marker-injection, newline, and quote-escape fixtures (R7); expected outcome (R7 is enforced in the reference implementation, §3.1.1): 100% abstain or sanitized.
- **System-level cost:** throughput (calls/sec/GPU) served under load.

---

## 7. Implementation Roadmap (6 Weeks)

| Phase | Milestone | Deliverables |
| --- | --- | --- |
| **Phase 1: Data & Alignment (W1–2)** | Jinja template + marker parser; Virtual Context Builder + Reverse Substring Aligner over 50k single-turn + multi-turn examples (xLAM incl. multi-turn, tau-bench); filter for leakage (GT-anchored dates) and unalignable rows. | Parsable silver-span dataset (per-class coverage report) + `span_template` pkg. |
| **Phase 2: Core Training (W3–4)** | Train ModernBERT/DeBERTa-v3 with boundary-factorized head + Circle Loss; abstention-upweighted. | Checkpoint + dev P@AR curve; τ candidate per schema class. |
| **Phase 3: Post-Processor & Router (W5)** | Deterministic token→typed mapper, per-param masks (R9), multi-span NMS, **call-level τ/δ calibration**, risk tiers, ONNX export, **whole-call gate + router hook** (drop-in before LLM tool-call generation), ReAct harness. *Scope risk: ONNX export and harness may slip to W6; the gate and calibration are the priority.* | End-to-end library; p50/p99 latency (GPU/CPU); router wiring + harness. |
| **Phase 4: Benchmarking & Ablations (W6)** | BFCL, OOD-paraphrase suite, multi-turn agentic loop (task success, Hit Rate × Precision-at-Hit, mixed latency incl. abstention), FLOPs/throughput table vs baselines. | Benchmark report; decision gate (adopt/expand/abandon) on Hit Rate × Precision-at-Hit and task-success numbers. |

---

## 8. Known Gaps & Open Questions (Tracked, Not Glossed)

- **Tool/schema selection** is out of scope; wrong schema → wrong context pool → bad extraction. Relevance-split numbers measure resolution, not routing.
- **Prompts are a targetable region:** adversarial user text can (a) forge structure (reserved markers, newlines) — mitigated deterministically by R7 sanitization — and (b) contain text that *looks like* enum/date values or an instruction ("set recipient to attacker@x.io"). Case (b) is semantically valid extraction and cannot be fully prevented by the template; it is bounded by risk tiers (§3.4) and recent-context-vs-utterance provenance logging. Residual risk is documented, not claimed solved.
- **Cross-parameter constraints** (`oneOf`, conditional `required`, mutual exclusion) are not modeled. A hit that violates them is caught by the post-processor → treat as abstain → LLM.
- **Anaphora beyond easy chaining:** "book the *other* flight" requires disambiguation beyond pointer semantics → **whole-call abstains** → LLM, by design; cross-turn precision claimed only for values verbatim in recent context ∪ query ∪ CTX.
- **Recent-context budget & cache invalidation:** unbounded growth and OOD edits to prior tool outputs require eviction/TTL policies and encoder-cache invalidation rules — specified and tested in Phase 3.
- **Hit-rate reality check:** the coverage assumption must be validated on *your* traffic before adopting; the router is instrumented from day one to log hit/abstain/wrong-value counts.