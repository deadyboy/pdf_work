# Agent convergence contract

This PR is the **vision/PDF pipeline adapter**, not a second universal orchestrator.

## Shared contract with docx_work

- Coarse input types: `docx | image | pdf | mixed | unknown` at the top-level orchestrator.
- LLM backend names: `ollama | vllm`.
- Bounded retry at the smallest failed unit; successful work is retained and old errors are cleared.
- System routing/merge failures use reserved `__*` error keys and are not retried as images.
- LangGraph is constrained to the verified `1.2.x` API line.

## Pipeline-specific state stays local

| Concern | docx_work | pdf_work |
|---|---|---|
| State | patient + field | input + image/slice |
| Core graph | field extraction queue | slice -> vision extract -> merge -> QC |
| Retry unit | field | image |
| Output | field-keyed structured record | ordered merged rows |

`PatientState` and `ImageProcessingState` should not be unified. A future shared package should contain only thin contracts (routing, backend selection, error/retry policy), never the clinical extraction state.

## Capability boundary in this draft

- Executed by this graph: `image_record1`, `image_jin`, and PDFs that classify to one of those types.
- Explicit handoff: `docx`, cross-media `mixed`, `image_mixed`.
- Explicitly not migrated yet: `image_record2`. The existing record-2 business pipeline uses five slices and five prompts; the previous Agent code incorrectly reused the three-slice record-1 mapping, so this PR now fails closed instead of silently producing wrong extraction.
