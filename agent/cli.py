from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .graph import run_agent

SCHEMA_VERSION = "icu-agent/v1"


def _status_from_state(state: Dict[str, Any]) -> str:
    errors = state.get("errors") or {}
    if any(str(key).startswith("__") for key in errors):
        return "failed"
    if errors:
        if state.get("merged_results") or state.get("final_output_path"):
            return "partial"
        return "failed"
    if state.get("merged_results") or state.get("final_output_path"):
        return "succeeded"
    return "failed"


def build_envelope(input_path: str, state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline": "vision",
        "status": _status_from_state(state),
        "input_path": os.path.abspath(input_path),
        "result": {
            "merged_results": state.get("merged_results", []),
            "final_output_path": state.get("final_output_path", ""),
        },
        "errors": state.get("errors", {}),
        "meta": {
            "detected_file_type": state.get("detected_file_type", "unknown"),
            "retry_counts": state.get("retry_count", {}),
        },
    }


def write_envelope(path: str | Path, envelope: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Vision/PDF ICU Agent adapter")
    parser.add_argument("input_path")
    parser.add_argument("--output-dir", default="agent_output")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--backend", choices=["ollama", "vllm"], default="vllm")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args(argv)

    try:
        state = run_agent(
            input_path=args.input_path,
            output_dir=args.output_dir,
            model=args.model,
            llm_backend=args.backend,
            llm_base_url=args.base_url,
        )
        envelope = build_envelope(args.input_path, state)
    except Exception as exc:
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "pipeline": "vision",
            "status": "failed",
            "input_path": os.path.abspath(args.input_path),
            "result": {"merged_results": [], "final_output_path": ""},
            "errors": {"__exception": f"{type(exc).__name__}: {exc}"},
            "meta": {},
        }

    write_envelope(args.result_json, envelope)
    return 0 if envelope["status"] in {"succeeded", "partial"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
