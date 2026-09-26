from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.file_type_router import detect_file_type
from agent.graph import (
    build_agent_graph,
    extract_slices_node,
    qc_check_node,
    route_after_detect,
)
from agent.prompts import get_prompts_for_type


class RouterContractTests(unittest.TestCase):
    def test_cross_media_folder_is_mixed(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "record.docx").touch()
            Path(d, "page.png").touch()
            file_type, _ = detect_file_type(d)
            self.assertEqual(file_type, "mixed")

    def test_docx_is_handoff_not_missing_node(self):
        self.assertEqual(route_after_detect({"detected_file_type": "docx"}), "unsupported")

    def test_record2_is_explicitly_not_migrated(self):
        self.assertEqual(route_after_detect({"detected_file_type": "image_record2"}), "unsupported")
        with self.assertRaises(KeyError):
            get_prompts_for_type("image_record2")

    def test_graph_compiles_with_pinned_langgraph_api(self):
        self.assertIsNotNone(build_agent_graph())


class RetryContractTests(unittest.TestCase):
    def test_successful_retry_clears_old_error_and_only_processes_pending(self):
        state = {
            "detected_file_type": "image_record1",
            "slice_dirs": {"ok": "/tmp/ok", "retry": "/tmp/retry"},
            "pending_images": ["/data/retry.png"],
            "raw_results": {"ok": [{"x": 1}]},
            "errors": {"retry": "old error"},
            "llm_backend": "vllm",
            "llm_base_url": "http://127.0.0.1:8000/v1",
            "model": "test",
        }
        with patch("agent.graph.make_llm_client", return_value=object()), \
             patch("agent.graph._extract_one_image", return_value=[{"x": 2}]) as extract:
            out = extract_slices_node(state)
        self.assertEqual(extract.call_count, 1)
        self.assertNotIn("retry", out["errors"])
        self.assertEqual(out["raw_results"]["ok"], [{"x": 1}])
        self.assertEqual(out["raw_results"]["retry"], [{"x": 2}])

    def test_system_errors_are_not_retried(self):
        out = qc_check_node({
            "errors": {"__route": "handoff"},
            "retry_count": {},
            "image_files": [],
        })
        self.assertEqual(out["pending_images"], [])
        self.assertEqual(out["phase"], "done")


if __name__ == "__main__":
    unittest.main()
