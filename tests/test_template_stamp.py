"""Tests for span_template.stamp — the template-provenance gate (plan W3 / S6).

The v1 eval was invalidated by a data/template skew nobody detected. These tests
pin the detector's contract: it must distinguish a matching dataset, a dataset
built against an older template, an unstamped dataset, and a corrupt stamp.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from span_template import stamp as stamp_mod


def _fake_template_tree(root: Path, *, marker: str = "v1") -> Path:
    """Minimal tree containing every stamped file name with controllable bytes."""
    root.mkdir(parents=True, exist_ok=True)
    for name in stamp_mod._STAMPED_FILES:
        (root / name).write_text(f"# {name} {marker}\n", encoding="utf-8")
    return root


def test_fingerprint_is_stable_for_unchanged_files(tmp_path: Path) -> None:
    root = _fake_template_tree(tmp_path / "tpl")
    a = stamp_mod.template_fingerprint(root)
    b = stamp_mod.template_fingerprint(root)
    assert a == b
    assert len(a) == 64


def test_fingerprint_changes_when_any_stamped_file_changes(tmp_path: Path) -> None:
    root = _fake_template_tree(tmp_path / "tpl", marker="v1")
    before = stamp_mod.template_fingerprint(root)
    (root / "template.jinja").write_text("# template.jinja v2\n", encoding="utf-8")
    after = stamp_mod.template_fingerprint(root)
    assert before != after


def test_fingerprint_ignores_unlisted_files(tmp_path: Path) -> None:
    """A new unrelated file must not invalidate existing datasets."""
    root = _fake_template_tree(tmp_path / "tpl")
    before = stamp_mod.template_fingerprint(root)
    (root / "README.md").write_text("docs\n", encoding="utf-8")
    assert stamp_mod.template_fingerprint(root) == before


def test_write_and_check_round_trip_is_ok(tmp_path: Path) -> None:
    root = _fake_template_tree(tmp_path / "tpl")
    data = tmp_path / "data"
    data.mkdir()
    stamp_mod.write_stamp(data, extra={"split": "eval"}, root=root)
    verdict = stamp_mod.check_stamp(data, root=root)
    assert verdict.ok is True
    assert verdict.status == "ok"
    assert verdict.recorded == verdict.current
    assert (data / stamp_mod.STAMP_FILENAME).is_file()


def test_check_reports_missing_stamp(tmp_path: Path) -> None:
    root = _fake_template_tree(tmp_path / "tpl")
    data = tmp_path / "data"
    data.mkdir()
    verdict = stamp_mod.check_stamp(data, root=root)
    assert verdict.ok is False
    assert verdict.status == "missing"
    assert "provenance unknown" in verdict.message()
    assert verdict.current is not None


def test_check_detects_template_changed_and_names_the_file(tmp_path: Path) -> None:
    """The exact v1 failure: dataset built, then the template edited."""
    root = _fake_template_tree(tmp_path / "tpl", marker="v1")
    data = tmp_path / "data"
    data.mkdir()
    stamp_mod.write_stamp(data, root=root)

    (root / "template_multi.jinja").write_text("# template_multi.jinja v2\n", encoding="utf-8")
    verdict = stamp_mod.check_stamp(data, root=root)
    assert verdict.ok is False
    assert verdict.status == "template_changed"
    assert "template_multi.jinja" in verdict.changed_files
    assert "TEMPLATE CHANGED" in verdict.message()
    assert "re-align" in verdict.message()


def test_check_detects_a_removed_stamped_file(tmp_path: Path) -> None:
    root = _fake_template_tree(tmp_path / "tpl")
    data = tmp_path / "data"
    data.mkdir()
    stamp_mod.write_stamp(data, root=root)
    (root / "sanitize.py").unlink()
    verdict = stamp_mod.check_stamp(data, root=root)
    assert verdict.ok is False
    assert verdict.status == "template_changed"


def test_check_reports_unreadable_stamp(tmp_path: Path) -> None:
    root = _fake_template_tree(tmp_path / "tpl")
    data = tmp_path / "data"
    data.mkdir()
    (data / stamp_mod.STAMP_FILENAME).write_text("{not json", encoding="utf-8")
    verdict = stamp_mod.check_stamp(data, root=root)
    assert verdict.ok is False
    assert verdict.status == "missing"

    (data / stamp_mod.STAMP_FILENAME).write_text(json.dumps({"nope": 1}), encoding="utf-8")
    verdict = stamp_mod.check_stamp(data, root=root)
    assert verdict.ok is False
    assert verdict.status == "unreadable"
    assert "could not be read" in verdict.message()


def test_stamp_payload_records_per_file_hashes(tmp_path: Path) -> None:
    root = _fake_template_tree(tmp_path / "tpl")
    payload = stamp_mod.stamp_payload({"source": "test"}, root=root)
    assert payload["template_fingerprint"] == stamp_mod.template_fingerprint(root)
    assert set(payload["files"]) == set(stamp_mod.file_hashes(root))
    assert payload["source"] == "test"
    # A different tree must not share the same fingerprint.
    other = _fake_template_tree(tmp_path / "tpl2", marker="v9")
    assert stamp_mod.template_fingerprint(other) != payload["template_fingerprint"]


def test_real_tree_fingerprint_is_deterministic() -> None:
    """The live repo tree must hash consistently across calls."""
    assert stamp_mod.template_fingerprint() == stamp_mod.template_fingerprint()
    hashes = stamp_mod.file_hashes()
    assert "template.jinja" in hashes
    assert "template_multi.jinja" in hashes
    assert "template.py" in hashes