"""Template provenance stamping (Treatise S6 of docs/lean_eval_diagnosis_walkthrough.md).

Why this exists
---------------
The v1 eval was invalidated by a silent skew: the processed data was built on
Nov 2 against one template revision, and ``span_template/`` changed on Nov 10-11.
Every stored span offset therefore addressed a render that no longer existed
(measured drift +30 from the revision alone, +9 more from the single/multi
grammar mismatch). Nothing in the pipeline noticed — the audit only happened
months later, by hand.

This module makes that skew *loud and mechanical*:

- :func:`template_fingerprint` returns a deterministic hash of every file that
  can change a render (the two Jinja templates, the render/parse module, the
  parser, and the sanitizer).
- :func:`stamp_payload` / :func:`write_stamp` record the fingerprint next to a
  processed dataset.
- :func:`check_stamp` compares a recorded stamp against the live tree and
  returns a structured verdict. Callers (data prep, training, notebook
  pre-flight) decide whether a mismatch is fatal or a warning — but it can no
  longer pass unseen.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Files whose bytes determine the rendered prompt AND the offsets inside it.
# Kept explicit (not a glob) so adding a render-affecting file is a deliberate
# edit to this list rather than a silent inclusion of, say, a test fixture.
_TEMPLATE_ROOT = Path(__file__).resolve().parent
_STAMPED_FILES: tuple[str, ...] = (
    "template.py",
    "template.jinja",
    "template_multi.jinja",
    "parser.py",
    "sanitize.py",
)

STAMP_FILENAME = "template_stamp.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def file_hashes(root: Optional[Path] = None) -> Dict[str, str]:
    """``{relative filename: sha256}`` for every render-affecting file."""
    base = Path(root) if root is not None else _TEMPLATE_ROOT
    out: Dict[str, str] = {}
    for name in _STAMPED_FILES:
        p = base / name
        if p.is_file():
            out[name] = _sha256_file(p)
    return out


def template_fingerprint(root: Optional[Path] = None) -> str:
    """One hash over all render-affecting files, order-independent.

    Hashes a sorted ``name=sha256`` listing rather than the concatenated bytes,
    so renaming a file changes the fingerprint even when content is identical.
    """
    hashes = file_hashes(root)
    blob = "\n".join(f"{k}={v}" for k, v in sorted(hashes.items()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def stamp_payload(
    extra: Optional[Dict[str, Any]] = None,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """The JSON body written next to a processed dataset.

    ``root`` overrides the template directory (tests, or stamping a dataset
    against a vendored copy); ``None`` means the live ``span_template/`` tree.
    """
    payload: Dict[str, Any] = {
        "template_fingerprint": template_fingerprint(root),
        "files": file_hashes(root),
    }
    if extra:
        payload.update(extra)
    return payload


def write_stamp(
    data_dir: str | os.PathLike[str],
    extra: Optional[Dict[str, Any]] = None,
    root: Optional[Path] = None,
) -> str:
    """Write ``template_stamp.json`` into ``data_dir``; returns its path."""
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / STAMP_FILENAME
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stamp_payload(extra, root), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return str(path)


def read_stamp(data_dir: str | os.PathLike[str]) -> Optional[Dict[str, Any]]:
    """Recorded stamp for ``data_dir``, or ``None`` when absent/corrupt."""
    path = Path(data_dir) / STAMP_FILENAME
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


@dataclass
class StampVerdict:
    """Outcome of comparing a recorded stamp against the live template tree."""

    ok: bool
    status: str                      # ok | missing | template_changed | unreadable
    recorded: Optional[str] = None
    current: Optional[str] = None
    changed_files: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)

    def message(self) -> str:
        if self.status == "ok":
            return f"template stamp OK ({self.current[:12]})"
        if self.status == "missing":
            return (
                "no template_stamp.json in this dataset: provenance unknown. "
                "Offsets may address an older template revision."
            )
        if self.status == "unreadable":
            return "template_stamp.json could not be read: provenance unknown."
        bits = [f"recorded={str(self.recorded)[:12]}", f"current={str(self.current)[:12]}"]
        if self.changed_files:
            bits.append("changed=" + ",".join(self.changed_files))
        if self.missing_files:
            bits.append("missing=" + ",".join(self.missing_files))
        return (
            "TEMPLATE CHANGED since this dataset was built ("
            + " ".join(bits)
            + "). Span offsets may address a render that no longer exists; "
            "re-align before training or evaluating."
        )


def check_stamp(data_dir: str | os.PathLike[str], root: Optional[Path] = None) -> StampVerdict:
    """Compare the dataset's recorded stamp against the live template tree."""
    current = template_fingerprint(root)
    recorded_doc = read_stamp(data_dir)
    if recorded_doc is None:
        return StampVerdict(ok=False, status="missing", current=current)
    recorded = recorded_doc.get("template_fingerprint")
    if not isinstance(recorded, str) or not recorded:
        return StampVerdict(ok=False, status="unreadable", current=current)

    old_files = recorded_doc.get("files") or {}
    new_files = file_hashes(root)
    changed = sorted(
        n for n in new_files
        if isinstance(old_files, dict) and old_files.get(n) not in (None, new_files[n])
    )
    missing = sorted(
        n for n in old_files if isinstance(old_files, dict) and n not in new_files
    )
    ok = recorded == current
    return StampVerdict(
        ok=ok,
        status="ok" if ok else "template_changed",
        recorded=recorded,
        current=current,
        changed_files=[] if ok else changed,
        missing_files=[] if ok else missing,
    )