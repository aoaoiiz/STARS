from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from creative_video_exp.provenance import prepare_protocol_manifest
from creative_video_exp.utils import append_jsonl, write_json, write_jsonl
from run_experiment import _load_existing_results


def _formal_manifest(fingerprint: str = "formal-fingerprint") -> dict:
    return {
        "protocol_fingerprint": fingerprint,
        "scientific_payload": {"models": {"mode": "server_full_matrix"}},
    }


class RunOutputSafetyTests(unittest.TestCase):
    def test_formal_nonempty_output_requires_resume(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            output = Path(raw_dir)
            marker = output / "existing.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "without --resume"):
                prepare_protocol_manifest(output, _formal_manifest(), resume=False)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertFalse((output / "protocol_manifest.json").exists())

    def test_formal_empty_output_accepts_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            output = Path(raw_dir)
            manifest = _formal_manifest()
            prepare_protocol_manifest(output, manifest, resume=False)
            self.assertEqual(
                json.loads(
                    (output / "protocol_manifest.json").read_text(encoding="utf-8")
                ),
                manifest,
            )

    def test_resume_nonempty_output_requires_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            output = Path(raw_dir)
            (output / "results.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "without protocol_manifest"):
                prepare_protocol_manifest(output, _formal_manifest(), resume=True)

    def test_resume_matching_manifest_allows_empty_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            output = Path(raw_dir)
            manifest = _formal_manifest()
            write_json(output / "protocol_manifest.json", manifest)
            (output / "results.jsonl").write_text("", encoding="utf-8")
            prepare_protocol_manifest(output, manifest, resume=True)

    def test_resume_rejects_malformed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            output = Path(raw_dir)
            (output / "protocol_manifest.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "cannot be validated"):
                prepare_protocol_manifest(output, _formal_manifest(), resume=True)

    def test_atomic_json_preserves_previous_file_on_serialization_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "metrics.json"
            path.write_text('{"status":"previous"}', encoding="utf-8")
            with self.assertRaises(TypeError):
                write_json(path, {"unsupported": {1, 2}})
            self.assertEqual(path.read_text(encoding="utf-8"), '{"status":"previous"}')
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_atomic_jsonl_preserves_previous_file_on_late_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "results.jsonl"
            previous = '{"sample_key":"previous"}\n'
            path.write_text(previous, encoding="utf-8")
            with self.assertRaises(TypeError):
                write_jsonl(path, [{"sample_key": "new"}, {"bad": {1}}])
            self.assertEqual(path.read_text(encoding="utf-8"), previous)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_append_jsonl_serializes_before_mutating_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "results.jsonl"
            previous = '{"sample_key":"previous"}\n'
            path.write_text(previous, encoding="utf-8")
            with self.assertRaises(TypeError):
                append_jsonl(path, {"bad": {1}})
            self.assertEqual(path.read_text(encoding="utf-8"), previous)

    def test_append_jsonl_rejects_incomplete_existing_tail(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "results.jsonl"
            previous = '{"sample_key":"partial"}'
            path.write_text(previous, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "incomplete final row"):
                append_jsonl(path, {"sample_key": "new"})
            self.assertEqual(path.read_text(encoding="utf-8"), previous)

    def test_resume_rejects_incomplete_checkpoint_tail(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "results.jsonl"
            path.write_text('{"sample_key":"partial"}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "incomplete JSONL row"):
                _load_existing_results(path)

    def test_resume_rejects_invalid_json_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "results.jsonl"
            path.write_text('{}\nnot-json\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "line 2"):
                _load_existing_results(path)


if __name__ == "__main__":
    unittest.main()
