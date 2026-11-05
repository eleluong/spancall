"""P0/P1 normalization + sanitization (docs/pipeline_io.md Part A, proposal R7).

Pure, deterministic functions: same input -> same bytes. ``build_input`` /
``build_input_multi`` run these *before* rendering; any failure raises
``SanitizeError`` with a reason code so the router can abstain (``marker_injection``,
``empty``, ``too_long``), matching the "every exit is an abstain with a reason"
design rule.

Why rejection is enough: if user text can never contain a marker string, every
marker in the rendered text is structural, and the parser's one-pass legend is
unambiguous. Defense in depth: ``MARKER_TOKENS`` is the single authoritative
vocabulary, also used for tokenizer special-token setup (pipeline_io P6).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Authoritative reserved-marker vocabulary (proposal R7 + pipeline_io P6).
# Keep in sync with template.jinja / template_multi.jinja / parser.py.
#
# ``MARKER_TOKENS`` are the *bare* marker names — the canonical keys for the
# R7 injection regex. ``SPECIAL_MARKER_TOKENS`` are the *atomic* ``<|...|>``
# single-token forms registered on the tokenizer (Qwen-native convention,
# matching Qwen3's own ``<|im_start|>``-style special tokens) so a modern BPE
# keeps each structural marker atomic — a raw ``<|``/``|>`` would otherwise
# byte-split and straddle an entry boundary (docs/pipeline_io.md P6.3,
# long-context R8 validation gate).
MARKER_TOKENS: Tuple[str, ...] = (
    "CLS", "SEP", "EOS",      # structural
    "TOOL", "PARAM", "NONE",  # schema anchors
    "NULL", "UNRESOLVED",     # CTX meta values
    "BAR", "EQ", "Q",         # P6 atomic delimiters (adopted in templates/parser)
)

SPECIAL_MARKER_TOKENS: Tuple[str, ...] = tuple(
    f"<|{t.lower()}|>" for t in MARKER_TOKENS
) + (
    "<|ctx_meta|>", "<|ctx_time|>", "<|ctx_enum|>", "<|ctx_recent|>",
)

# Matches any reserved marker, case-insensitively, on an NFKC copy so that
# fullwidth forms (［ＳＥＰ］, ＜ｓｅｐ＞) are caught too. Two forms are
# detected: the atomic ``<|X|>`` grammar used by the current templates and the
# legacy bracketed ``[X]`` form, plus the ``CTX_*`` segment markers. The NFKC
# copy normalizes any fullwidth brackets/angle-brackets/letters in the candidate.
_MARKER_RE = re.compile(
    r"(?:<\|("
    + "|".join(re.escape(t) for t in MARKER_TOKENS)
    + r"|CTX_[A-Z]+)\|>"
    + r"|\[("
    + "|".join(re.escape(t) for t in MARKER_TOKENS)
    + r"|CTX_[A-Z]+)\])",
    re.IGNORECASE,
)

# C0/C1 controls to strip (keep \n and \t — they are collapsed by whitespace
# normalization anyway), zero-width chars, and bidi override/embedding controls.
_CONTROL_NON_KEEP = re.compile(
    r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F"
    r"\u200B-\u200F\u202A-\u202E\u2060\uFEFF]"
)

_WS_CHARS = " \t\n\r\f\v"
_WORD_RE = re.compile(r"\S+")

# Utterance length cap (R7 too_long abstain). Lifted (None) so the
# template can ingest arbitrary-length long-form tool-call narratives and
# trajectories without an arbitrary character cap (long_context_fix.md).
# The authoritative overflow guard is the backbone-driven token budget
# (max_seq_len / L_max) post-tokenization.
U_MAX_DEFAULT: Optional[int] = None


class SanitizeError(ValueError):
    """Raised when an untrusted input cannot be rendered safely.

    ``.reason`` is the abstain reason code (``marker_injection``, ``empty``,
    ``too_long``).
    """

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"sanitize abstain: {reason}"
                         + (f" ({detail})" if detail else ""))
        self.reason = reason


@dataclass(frozen=True)
class SanitizeResult:
    """Output of P0/P1 for one untrusted string.

    ``orig`` and ``norm`` are offset-linked via ``norm2orig`` so Q4 can always
    return the user's exact original characters: ``orig[norm2orig[i]]`` is the
    original char that normalized index ``i`` came from.
    """

    orig: str
    norm: str
    norm2orig: List[int]
    nfc_check: str
    reason: Optional[str] = None  # None -> safe

    def check(self) -> "SanitizeResult":
        if self.reason is not None:
            raise SanitizeError(self.reason)
        return self


def sanitize_utterance(text: str, max_chars: Optional[int] = U_MAX_DEFAULT) -> SanitizeResult:
    """P0/P1 for the user utterance. Raises via ``check()``; else returns norm.

    Steps: NFKC detection copy -> reserved-marker rejection -> control-char
    stripping (offset-preserving) -> whitespace-run collapse with ``norm2orig``
    -> empty/length checks. Casing is never changed.
    """
    nfc = unicodedata.normalize("NFKC", text)

    if _MARKER_RE.search(nfc):
        return SanitizeResult(text, text, list(range(len(text))), nfc,
                              reason="marker_injection")

    cleaned = _CONTROL_NON_KEEP.sub(" ", text)
    norm, mapping = _collapse_ws_with_map(cleaned)

    if not norm.strip():
        return SanitizeResult(text, norm, mapping, nfc, reason="empty")
    if max_chars is not None and len(norm) > max_chars:
        return SanitizeResult(text, norm, mapping, nfc, reason="too_long")
    return SanitizeResult(text, norm, mapping, nfc, reason=None)


def check_recent_value(value: str) -> Optional[str]:
    """R7 for recent-context values/keys: reserved marker -> reason code.

    Recent rows are injected verbatim into the template, so they are untrusted
    exactly like the utterance (pipeline_io P3.5). Returns the reason code or
    ``None`` if safe.
    """
    if _MARKER_RE.search(unicodedata.normalize("NFKC", value)):
        return "marker_injection"
    return None


def words(text: str) -> List[Tuple[int, int, str]]:
    """Whitespace-separated words with absolute ``(start, end, word)`` offsets."""
    return [(m.start(), m.end(), m.group(0)) for m in _WORD_RE.finditer(text)]


def _collapse_ws_with_map(text: str) -> Tuple[str, List[int]]:
    """Collapse whitespace runs to one space; map each norm index to an orig index.

    Leading/trailing runs are dropped entirely (no dangling space at the edges),
    so the rendered utterance line never starts/ends with whitespace — keeping the
    parser's char-offset contract exact (entry covers ``text[start:end]`` verbatim).
    """
    out: List[str] = []
    mapping: List[int] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _WS_CHARS:
            j = i
            while j < n and text[j] in _WS_CHARS:
                j += 1
            if i == 0 or j == n:
                i = j            # drop leading/trailing whitespace runs
                continue
            out.append(" ")
            mapping.append(i)    # first orig index of the run
            i = j
        else:
            out.append(ch)
            mapping.append(i)
            i += 1
    return "".join(out), mapping