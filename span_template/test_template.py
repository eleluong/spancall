"""Tests for the SpanCall input template (Jinja render -> marker parse).

Run:  .venv/bin/python -m unittest discover -s span_template -p 'test_*.py' -v
"""

import unittest
from datetime import date

from span_template import (
    Entry,
    Param,
    RecentRow,
    SanitizeError,
    SpanTemplateOutput,
    Tool,
    build_input,
    build_input_multi,
    build_time_window,
    gate_tool_call,
    sanitize_utterance,
    tokenize_offset_mapping,
)


def demo_tool(enum=("ECONOMY", "BUSINESS", "FIRST")) -> Tool:
    return Tool(
        name="book_flight",
        description="Book a selected flight",
        params=[
            Param("flight_id", "Selected flight identifier", required=True),
            Param("passenger_name", "Full name on reservation", required=True),
            Param("email", "Notification email address", required=True),
            Param("class", "Service class", enum=enum, required=False),
        ],
    )


def demo_input(now: date = date(2026, 10, 23), sanitize: bool = True) -> SpanTemplateOutput:
    return build_input(
        tool=demo_tool(),
        utterance="Book that flight, send the receipt to my work email",
        recent_context=[
            RecentRow("picked", "VN123", source="agent"),
            RecentRow("user.name", "Anh Luong", source="session"),
            RecentRow("user.email", "anh.luong@corp.io", source="session"),
        ],
        now=now,
        weekday_tokens=("next_mon",),
        sanitize=sanitize,
    )


class SanitizeTest(unittest.TestCase):
    def test_marker_injection_abstains(self):
        # The current templates render the Qwen-native <|...|> markers; BOTH the
        # native forms and the legacy bracketed [X] forms are reserved by the R7
        # regex (sanitize._MARKER_RE), so any of them must abstain.
        for bad in ("<|sep|>", "ignore <|param|> x", "x <|ctx_recent|> y",
                    "fullwidth ＜｜ｓｅｐ｜＞", "u <|none|> v", "j<|unresolved|>k",
                    "<|bar|>", "<|eq|>", "<|q|>",
                    "[SEP]", "ignore [PARAM] x", "x [CTX_RECENT] y",
                    "fullwidth ［ＳＥＰ］", "u [NONE] v", "j[unresolved]k",
                    "[BAR]", "[EQ]", "[Q]"):
            with self.assertRaises(SanitizeError,
                                   msg=f"marker {bad!r} not rejected") as cm:
                sanitize_utterance(bad).check()
            self.assertEqual(cm.exception.reason, "marker_injection")
            with self.assertRaises(SanitizeError):
                build_input(demo_tool(), bad, now=date(2026, 10, 23))

    def test_empty_and_too_long(self):
        with self.assertRaises(SanitizeError) as cm:
            sanitize_utterance(" \n\t ").check()
        self.assertEqual(cm.exception.reason, "empty")

        # By default, U_MAX_DEFAULT is None (no character-length cap; long inputs
        # are permitted and guarded by token budgeting and post-tokenization overflow)
        long = "x" * 5000
        res = sanitize_utterance(long)
        self.assertIsNone(res.reason)
        self.assertEqual(len(res.norm), 5000)

        # When max_chars is explicitly passed, it enforces too_long
        with self.assertRaises(SanitizeError) as cm:
            sanitize_utterance(long, max_chars=2048).check()
        self.assertEqual(cm.exception.reason, "too_long")

        import inspect
        signature = inspect.signature(sanitize_utterance)
        assert "max_chars" in signature.parameters

    def test_clean_text_unchanged(self):
        res = sanitize_utterance("Book a flight to Hanoi, please!")
        self.assertIsNone(res.reason)
        self.assertEqual(res.norm, "Book a flight to Hanoi, please!")
        self.assertEqual(res.orig[res.norm2orig[5]], res.norm[5])

    def test_newline_collapse_preserves_norm2orig(self):
        res = sanitize_utterance("Book  \n  flight   now")
        self.assertEqual(res.norm, "Book flight now")
        for i, ch in enumerate(res.norm):
            self.assertEqual(res.orig[res.norm2orig[i]], ch)

    def test_recent_row_marker_injection_rejected(self):
        with self.assertRaises(ValueError):
            RecentRow("picked", "VN<|sep|>123")   # value contains a marker
        with self.assertRaises(ValueError):
            RecentRow("picked", "up<|tool|>grade")
        with self.assertRaises(ValueError):
            RecentRow("picked", "VN[SEP]123")     # legacy bracketed form too
        with self.assertRaises(ValueError):
            RecentRow("p[i]cked", "ok")         # key fails the plain-identifier rule
        with self.assertRaises(ValueError):
            RecentRow("<|sep|>", "ok")            # key contains a reserved marker

    def test_norm2orig_roundtrip_via_build_input(self):
        out = build_input(demo_tool(), "  Book   that \n flight ",
                          now=date(2026, 10, 23))
        self.assertIsNotNone(out.norm)
        self.assertEqual(out.norm.orig, "  Book   that \n flight ")
        # P0 collapses whitespace runs and drops the leading/trailing runs, so
        # the rendered utterance line is trimmed (no dangling edge spaces).
        self.assertEqual(out.norm.norm, "Book that flight")
        for i, ch in enumerate(out.norm.norm):
            self.assertEqual(out.norm.orig[out.norm.norm2orig[i]], ch)

    def test_marker_tokens_are_single_authoritative(self):
        # R7: one authoritative marker vocabulary — the parser must not drift
        # from the sanitizer's copy. Same object, not just equal values.
        import span_template.parser as _parser
        import span_template.sanitize as _sanitize
        self.assertIs(_parser.MARKER_TOKENS, _sanitize.MARKER_TOKENS)


class BuildInputTest(unittest.TestCase):
    def test_text_contains_all_segments_in_order(self):
        out = demo_input()
        text = out.text
        self.assertIn("<|cls|>", text)
        self.assertIn("<|tool|> book_flight (Book a selected flight)", text)
        self.assertIn("<|ctx_meta|> <|null|> | true | false | <|unresolved|>", text)
        self.assertIn("<|ctx_time|> today=2026-10-23", text)
        self.assertIn("<|sep|>", text)
        self.assertIn("Book that flight, send the receipt to my work email", text)
        self.assertIn('<|ctx_recent|> picked="VN123"', text)
        self.assertIn("<|eos|>", text)
        order = [text.index(m) for m in ("<|cls|>", "<|tool|>", "<|ctx_enum|>",
                                         "Book that flight", "<|ctx_recent|>", "<|eos|>")]
        self.assertEqual(order, sorted(order), "segment order violated")

    def test_char_offsets_line_up_with_text(self):
        out = demo_input()
        for e in out.entries:
            self.assertEqual(out.text[e.char_start : e.char_end], e.text,
                             f"entry {e!r} offset mismatch")

    def test_entries_have_expected_kinds_and_targetable_flags(self):
        out = demo_input()
        kinds = {e.kind for e in out.entries}
        self.assertIn("param_anchor", kinds)
        self.assertIn("date", kinds)       # CTX_TIME values
        self.assertIn("unresolved", kinds)  # <|unresolved|> target
        self.assertIn("string", kinds)      # utterance + recent values
        # only selectable entries can be targets
        for e in out.entries:
            if not e.selectable:
                self.assertIn(e.kind, {"param_anchor", "key", "bridge_text"})

    def test_recent_keys_are_bridges_and_values_targets(self):
        out = demo_input()
        recent = [e for e in out.entries if e.segment == "recent"]
        self.assertTrue(any(e.value == "picked" and not e.selectable for e in recent))
        self.assertTrue(any(e.value == "VN123" and e.selectable for e in recent))
        val = next(e for e in recent if e.value == "VN123")
        self.assertEqual(val.text, "VN123")

    def test_time_window_contains_today_and_next_monday(self):
        out = demo_input()
        dates = {e.key: e.iso for e in build_time_window(date(2026, 10, 23), weekdays=("next_mon",))}
        self.assertEqual(dates["today"], "2026-10-23")
        self.assertEqual(dates["next_mon"], "2026-10-26")  # Fri -> next Mon

    def test_offset_keys_are_full_keys(self):
        out = demo_input()
        keys = {e.key for e in out.entries if e.segment == "ctx_time" and not e.selectable}
        self.assertIn("+1d", keys)
        self.assertIn("+3d", keys)
        self.assertIn("+7d", keys)
        self.assertNotIn("d", keys)   # regression: truncated to "d" before

    def test_enum_pool_omitted_when_no_enums(self):
        tool = Tool("noop", description="", params=[Param("a", "")])
        out = build_input(tool, "hello", now=date(2026, 10, 23))
        self.assertNotIn("<|ctx_enum|>", out.text)

    def test_recent_value_quoting_escapes(self):
        out = build_input(
            tool=demo_tool(),
            utterance="ping",
            recent_context=[RecentRow("note", 'a "quoted" \\ value')],
            now=date(2026, 10, 23),
        )
        self.assertIn('<|ctx_recent|> note="a \\"quoted\\" \\\\ value"', out.text)

    def test_recent_value_roundtrips_through_unescape(self):
        out = build_input(
            tool=demo_tool(),
            utterance="ping",
            recent_context=[RecentRow("note", 'a "quoted" \\ value')],
            now=date(2026, 10, 23),
        )
        recent = [e for e in out.entries if e.segment == "recent" and e.selectable]
        self.assertTrue(any(e.value == 'a "quoted" \\ value' for e in recent))


class TimeWindowTest(unittest.TestCase):
    def test_this_weekday_on_its_own_day_is_today(self):
        # 2026-10-30 is a Friday; this_fri must be 2026-10-30 (today), not +7d
        fri = date(2026, 10, 30)
        dates = {e.key: e.iso for e in
                 build_time_window(fri, weekdays=("this_fri", "next_fri"))}
        self.assertEqual(dates["this_fri"], "2026-10-30")
        self.assertEqual(dates["next_fri"], "2026-11-06")  # strictly future

    def test_this_vs_next_distinct(self):
        # 2026-10-23 is a Friday; the next Monday is the 26th both for
        # this_mon (0..6 days ahead) and next_mon (delta != 0, no +7 bump).
        fri = date(2026, 10, 23)
        dates = {e.key: e.iso for e in
                 build_time_window(fri, weekdays=("this_mon", "next_mon"))}
        self.assertEqual(dates["this_mon"], "2026-10-26")
        self.assertEqual(dates["next_mon"], "2026-10-26")
        # From Mon 10-26, Friday is +4d for BOTH (delta != 0, so no +7 bump):
        mon = date(2026, 10, 26)
        dates2 = {e.key: e.iso for e in
                  build_time_window(mon, weekdays=("this_fri", "next_fri"))}
        self.assertEqual(dates2["this_fri"], "2026-10-30")
        self.assertEqual(dates2["next_fri"], "2026-10-30")
        # The two diverge only when base IS the target weekday (covered by
        # test_this_weekday_on_its_own_day_is_today: this_fri=today, next_fri=+7d).

    def test_colliding_values_are_separate_entries(self):
        out = build_input(demo_tool(), "ping", now=date(2026, 10, 23))
        dates = [e for e in out.entries if e.kind == "date"]
        vals = [d.value for d in dates]
        self.assertEqual(vals.count("2026-10-26"), 2)  # +3d and next_mon collide

    def test_unsupported_weekday_raises(self):
        with self.assertRaises(ValueError):
            build_time_window(date(2026, 10, 23), weekdays=("last_mon",))

    def test_idiom_map_tomorrow(self):
        # 2026-10-23 is a Friday; tomorrow == +1d == 2026-10-24
        dates = {e.key: e.iso for e in build_time_window(date(2026, 10, 23))}
        self.assertEqual(dates["tomorrow"], "2026-10-24")

    def test_idiom_map_day_after_tomorrow(self):
        dates = {e.key: e.iso for e in build_time_window(date(2026, 10, 23))}
        self.assertEqual(dates["day_after_tomorrow"], "2026-10-25")  # +2d

    def test_idiom_map_tonight_is_today(self):
        dates = {e.key: e.iso for e in build_time_window(date(2026, 10, 23))}
        self.assertEqual(dates["tonight"], "2026-10-23")

    def test_idiom_offsets_default_window_days(self):
        # D1 default window_days = (1, 2, 3, 7); +2d now present.
        keys = {e.key for e in build_time_window(date(2026, 10, 23))}
        self.assertIn("+1d", keys)
        self.assertIn("+2d", keys)
        self.assertIn("+3d", keys)   # kept for compatibility (R4)
        self.assertIn("+7d", keys)

    def test_unsupported_idiom_raises(self):
        with self.assertRaises(ValueError):
            build_time_window(date(2026, 10, 23), idioms=("next_week",))


class EnumPoolTest(unittest.TestCase):
    def test_enum_with_separator_rejected(self):
        # '|' is forbidden even quoted (R5/D2)
        with self.assertRaises(ValueError):
            build_input(demo_tool(enum=("ECONOMY", "BUSINESS | DELUXE")),
                        "ping", now=date(2026, 10, 23))
        with self.assertRaises(ValueError):
            build_input(demo_tool(enum=('"BUSINESS | DELUXE"', "ECONOMY")),
                        "ping", now=date(2026, 10, 23))
        with self.assertRaises(ValueError):
            build_input(demo_tool(enum=("A|B", "ECONOMY")),
                        "ping", now=date(2026, 10, 23))

    def test_enum_with_internal_whitespace_is_quoted(self):
        out = build_input(demo_tool(enum=("ECONOMY", "Premium Economy", "BUSINESS")),
                          "ping", now=date(2026, 10, 23))
        self.assertIn('"Premium Economy"', out.text)
        # unquoted single-token values remain unquoted
        self.assertIn('<|ctx_enum|> BUSINESS | ECONOMY | "Premium Economy"', out.text)

    def test_enum_quoted_parses_to_canonical_value(self):
        out = build_input(demo_tool(enum=("ECONOMY", "Premium Economy")),
                          "ping", now=date(2026, 10, 23))
        enum_entries = [e for e in out.entries if e.segment == "ctx_enum"]
        self.assertTrue(any(e.value == "Premium Economy" for e in enum_entries))
        # entry text is the canonical unquoted value (char-offset contract)
        quoted = next(e for e in enum_entries if e.value == "Premium Economy")
        self.assertEqual(out.text[quoted.char_start:quoted.char_end], '"Premium Economy"')

    def test_enum_leading_trailing_whitespace_rejected(self):
        with self.assertRaises(ValueError):
            build_input(demo_tool(enum=(" Economy", "ECONOMY")),
                        "ping", now=date(2026, 10, 23))
        with self.assertRaises(ValueError):
            build_input(demo_tool(enum=("ECONOMY ", "ECONOMY")),
                        "ping", now=date(2026, 10, 23))

    def test_enum_dedupe_and_order(self):
        out = build_input(demo_tool(enum=("BUSINESS", "ECONOMY", "BUSINESS")),
                          "ping", now=date(2026, 10, 23))
        self.assertEqual(out.text.count("BUSINESS"), 1)
        # sorted(unique())
        self.assertLess(out.text.index("BUSINESS"), out.text.index("ECONOMY"))


class MultiToolTest(unittest.TestCase):
    def test_packing_renders_two_tools_and_none_anchor(self):
        t1 = demo_tool()
        t2 = Tool("search_flights", "Search flights",
                  params=[Param("origin", "Origin city"), Param("destination", "")])
        out = build_input_multi([t1, t2], "find flights from Hanoi to Da Nang",
                                now=date(2026, 10, 23))
        self.assertIn("<|tool|> book_flight", out.text)
        self.assertIn("<|tool|> search_flights", out.text)
        self.assertIn("<|none|>", out.text)
        self.assertEqual(len(out.tools), 2)

    def test_anchors_are_tagged_with_tool(self):
        t1 = demo_tool()
        t2 = Tool("search_flights", "Search flights",
                  params=[Param("origin", "Origin city"), Param("destination", "")])
        out = build_input_multi([t1, t2], "hi", now=date(2026, 10, 23))
        anchors = [e for e in out.entries if e.kind == "param_anchor"]
        tools = {e.tool for e in anchors}
        self.assertIn("book_flight", tools)
        self.assertIn("search_flights", tools)
        self.assertIsNone(next(e.tool for e in out.entries
                               if e.value == "<|none|>"))

    def test_enum_union_across_tools(self):
        t2 = Tool("pick_class", "Pick class",
                  params=[Param("cls", "c", enum=["PREMIUM"])])
        out = build_input_multi([demo_tool(), t2], "upgrade",
                                now=date(2026, 10, 23))
        self.assertIn("PREMIUM", out.text)
        self.assertIn("ECONOMY", out.text)

    def test_single_tool_via_multi_matches_text(self):
        single = demo_input(sanitize=False)
        multi = build_input_multi([demo_tool()], single.utterance,
                                  recent_context=list(single.recent_context),
                                  now=date(2026, 10, 23),
                                  weekday_tokens=("next_mon",),
                                  sanitize=False)
        # identical except the <|none|> line
        self.assertIn("<|none|>", multi.text)
        self.assertNotIn("<|none|>", single.text)
        self.assertEqual(multi.text.replace("<|none|>\n", ""), single.text)
        # legend: the shared entries are identical in
        # kind/segment/value/key/text/selectability — the multi pack only adds
        # the <|none|> anchor and re-tags ctx entries; the schema anchors must
        # also agree on their (kind, tool, value) tagging.
        def sig(e):
            return (e.kind, e.segment, e.value, e.key, e.text, e.selectable)
        shared_multi = [sig(e) for e in multi.entries if e.value != "<|none|>"]
        self.assertEqual(shared_multi, [sig(e) for e in single.entries])
        def anchor(e):
            return (e.kind, e.tool, e.value)
        # schema anchors only — exclude <|none|>, which is unique to the multi
        # form (it parses as a param_anchor with tool=None, asserted by the
        # text check)
        self.assertEqual(
            [anchor(e) for e in multi.entries
             if e.kind in ("bridge_text", "param_anchor") and e.value != "<|none|>"],
            [anchor(e) for e in single.entries
             if e.kind in ("bridge_text", "param_anchor")],
        )


class GateTest(unittest.TestCase):
    def _resolve(self, mapping):
        def res(p: Param):
            if p.name in mapping:
                return True, mapping[p.name]
            return False, None
        return res

    def test_hit_when_all_required_resolve(self):
        out = demo_input()
        mapping = {"flight_id": "VN123", "passenger_name": "Anh Luong",
                   "email": "anh.luong@corp.io"}
        result = gate_tool_call(out, self._resolve(mapping))
        self.assertTrue(result.hit)
        self.assertFalse(result.abstained)
        self.assertEqual(result.payload["flight_id"], "VN123")

    def test_abstain_when_required_missing(self):
        out = demo_input()
        mapping = {"flight_id": "VN123"}  # passenger_name + email missing
        result = gate_tool_call(out, self._resolve(mapping))
        self.assertFalse(result.hit)
        self.assertTrue(result.abstained)
        self.assertIn("passenger_name", result.missing_required)
        self.assertIn("email", result.missing_required)

    def test_optional_unresolved_is_ok(self):
        out = demo_input()
        mapping = {"flight_id": "VN123", "passenger_name": "Anh Luong",
                   "email": "anh.luong@corp.io"}
        result = gate_tool_call(out, self._resolve(mapping))
        self.assertTrue(result.hit)  # `class` optional+absent → defaulted

    def test_optional_with_default_is_materialized(self):
        tool = Tool("opts", "Options", params=[
            Param("a", "req", required=True),
            Param("b", "opt w/ default", required=False, default="BUSINESS",
                  has_default=True),
            Param("c", "opt no default", required=False),
        ])
        out = build_input(tool, "x", now=date(2026, 10, 23))
        res = gate_tool_call(out, lambda p: (False, None))  # nothing resolves
        self.assertTrue(res.abstained)
        res2 = gate_tool_call(out, lambda p: (True, "x") if p.name == "a" else (False, None))
        self.assertTrue(res2.hit)
        self.assertEqual(res2.payload["b"], "BUSINESS")   # schema default applied
        self.assertNotIn("c", res2.payload)               # no default -> omitted

    def test_multi_tool_gate_abstains_on_second_tool(self):
        # C1: the gate must cover ALL packed tools, not just tools[0].
        t2 = Tool("search_flights", "Search flights",
                  params=[Param("origin", "Origin city", required=True),
                          Param("destination", "Destination city", required=True)])
        out = build_input_multi([demo_tool(), t2], "find flights",
                                now=date(2026, 10, 23))
        # only tool 1 resolves fully -> tool 2's required params must abstain
        mapping = {"flight_id": "VN123", "passenger_name": "Anh",
                   "email": "anh@corp.io"}
        result = gate_tool_call(out, self._resolve(mapping))
        self.assertFalse(result.hit)
        self.assertTrue(result.abstained)
        self.assertIn("search_flights.origin", result.missing_required)
        self.assertIn("search_flights.destination", result.missing_required)

    def test_multi_tool_gate_hit_namespaces_payload(self):
        t2 = Tool("search_flights", "Search flights",
                  params=[Param("origin", "Origin city", required=True)])
        out = build_input_multi([demo_tool(), t2], "find flights",
                                now=date(2026, 10, 23))
        mapping = {"flight_id": "VN123", "passenger_name": "Anh",
                   "email": "anh@corp.io", "origin": "Hanoi"}
        result = gate_tool_call(out, self._resolve(mapping))
        self.assertTrue(result.hit)
        self.assertFalse(result.abstained)
        # multi pack -> payload keys are namespaced <tool>.<param>
        self.assertEqual(result.payload["book_flight.flight_id"], "VN123")
        self.assertEqual(result.payload["search_flights.origin"], "Hanoi")

    def test_abstained_reports_missing_required(self):
        out = demo_input()
        result = gate_tool_call(out, lambda p: (False, None))
        self.assertTrue(result.abstained)
        self.assertEqual(set(result.missing_required),
                         {"flight_id", "passenger_name", "email"})


class OffsetMappingTest(unittest.TestCase):
    def test_offset_mapping_roundtrip_no_straddle(self):
        # a BPE tokenizer would split on spaces; here a simple char-based
        # tokenizer keeps every entry boundary on a token boundary
        class CharTok:
            def tokenize(self, s):
                return list(s)          # one char per token
            def convert_tokens_to_ids(self, toks):
                return [ord(t) for t in toks]
            def __call__(self, text, return_offsets_mapping=False,
                         add_special_tokens=False):
                toks = self.tokenize(text)
                pos = 0
                off = []
                for t in toks:
                    start = text.index(t, pos)
                    off.append((start, start + 1))
                    pos = start + 1
                enc = {"input_ids": self.convert_tokens_to_ids(toks),
                       "offset_mapping": off}
                return enc

        out = demo_input()
        # the length guard is not what this test probes: give it headroom
        ids, per_token = tokenize_offset_mapping(out, CharTok(), max_len=10_000)
        self.assertEqual(len(ids), len(per_token))
        self.assertEqual(len(ids), len(out.text))
        # the "<|cls|>" marker chars are None (not inside an entry)
        self.assertIsNone(per_token[0])  # '<'
        # every entry start/end is a token boundary by construction

    def test_straddling_token_raises_bad_boundary(self):
        # a tokenizer that fuses across the <|tool|> entry boundary must abstain
        class FuseTok:
            def tokenize(self, s):
                return [s]              # one giant token => boundary violation
            def convert_tokens_to_ids(self, toks):
                return [0] * len(toks)
            def __call__(self, text, return_offsets_mapping=False,
                         add_special_tokens=False):
                enc = {"input_ids": self.convert_tokens_to_ids(self.tokenize(text)),
                       "offset_mapping": [(0, len(text))]}
                return enc

        out = demo_input()
        with self.assertRaises(SanitizeError) as cm:
            tokenize_offset_mapping(out, FuseTok(), max_len=10_000)
        self.assertEqual(cm.exception.reason, "bad_boundary")

    def test_shared_boundary_attributes_to_starting_entry(self):
        # Regression: when two entries are adjacent (entry A ends exactly where
        # entry B starts), a token whose start lands on that shared offset must
        # be attributed to B (the STARTING entry), never to A (the ending one).
        # The old code registered BOTH char_start and char_end via setdefault,
        # so `bounds[shared]` kept the first writer (A) and the boundary check
        # (which only catches straddling, not mis-attribution) let it through.
        # A char tokenizer exercises this on every char: each char boundary is
        # [prev].end == [next].start.
        class CharTok:
            def tokenize(self, s):
                return list(s)          # one char per token
            def convert_tokens_to_ids(self, toks):
                return [ord(t) for t in toks]
            def __call__(self, text, return_offsets_mapping=False,
                         add_special_tokens=False):
                toks = self.tokenize(text)
                pos = 0
                off = []
                for t in toks:
                    start = text.index(t, pos)
                    off.append((start, start + 1))
                    pos = start + 1
                enc = {"input_ids": self.convert_tokens_to_ids(toks),
                       "offset_mapping": off}
                return enc

        # "ab" [0, 2) is immediately followed by "cd" [2, 4): offset 2 is BOTH
        # entry_a.char_end AND entry_b.char_start. The trailing '<' is a
        # bridge/delimiter char inside no entry.
        entry_a = Entry(kind="string", segment="utterance", value="ab",
                        char_start=0, char_end=2, selectable=True, text="ab")
        entry_b = Entry(kind="string", segment="utterance", value="cd",
                        char_start=2, char_end=4, selectable=True, text="cd")
        out = SpanTemplateOutput(text="abcd<", entries=[entry_a, entry_b],
                                 tool=Tool("t", ""), utterance="abcd<",
                                 time_window=(), recent_context=())
        ids, per_token = tokenize_offset_mapping(out, CharTok(), max_len=10_000)
        self.assertEqual(len(ids), len(out.text))
        # char tokenizer => one token per char, so token i starts at char i;
        # the token starting on the shared offset must be B, never A.
        self.assertIs(per_token[entry_b.char_start], entry_b)
        self.assertIsNot(per_token[entry_b.char_start], entry_a)
        self.assertIs(per_token[entry_a.char_start], entry_a)
        # bridge/delimiter chars between entries still map to None
        self.assertIsNone(per_token[4])  # '<'


if __name__ == "__main__":
    unittest.main()