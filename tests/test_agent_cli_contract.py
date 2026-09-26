from __future__ import annotations

import unittest
from pathlib import Path


def load_cli():
    path = Path(__file__).resolve().parents[1] / "agent" / "cli.py"
    source = path.read_text(encoding="utf-8")
    source = source.replace("from .graph import run_agent\n", "run_agent = None\n")
    namespace = {"__name__": "vision_cli_contract_test", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


class CliEnvelopeTests(unittest.TestCase):
    def test_success_envelope(self):
        ns = load_cli()
        env = ns["build_envelope"](
            "/tmp/p1",
            {"merged_results": [{"x": 1}], "final_output_path": "/tmp/out.json", "errors": {}, "detected_file_type": "image_record1"},
        )
        self.assertEqual(env["schema_version"], "icu-agent/v1")
        self.assertEqual(env["pipeline"], "vision")
        self.assertEqual(env["status"], "succeeded")

    def test_route_error_is_failed(self):
        ns = load_cli()
        env = ns["build_envelope"]("/tmp/p1.docx", {"errors": {"__route": "handoff"}})
        self.assertEqual(env["status"], "failed")


if __name__ == "__main__":
    unittest.main()
