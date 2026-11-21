# JACK — Jarring Alignment & Candor Kernel

**Status (2026-11-21): Runnable experimental from-scratch statistical language model; personality fidelity NOT qualified.**

JACK is an attempt to construct a **model of Patrick**, using Patrick-authorized conversation history, writing, life history, and other attributable records. It is **not** intended to be a generic, perpetually abrasive truth-teller. Rigorous candor, questioning easy consensus, humor, emotional complexity, and the ability to reconsider a position are parts of the target, not substitutions for Patrick.

This repository originally described JACK as an abrasive cognitive runtime that would expose blind spots and resist sycophantic drift. That remains an aspiration; this release makes a narrower, reproducible implementation claim.

## What exists

- `jack_sources.py`: read-only, provenance-aware local intake of native ChatGPT exports, speaker-stamped PDF-text exports, user text, verified public text, and DOCX historical material. Assistant responses are **not** silently attributed to Patrick. For mapped conversation exports with a `current_node`, only the selected ancestry is ingested; missing/cyclic ancestry is rejected. Without `current_node`, a complete, single unbranched parent chain may be inferred; branched, disconnected, or unstructured mappings are skipped instead of being misattributed. Previously built private weight artifacts are not automatically recalculated after an intake change. Medical and third-party records stay in separate evidence classes.
- `jack_model.py`: a **new, independent, closed-form word-level trigram language model**. It calculates lexical frequency weights, conditional word-transition weights, and historical-topic weights from the local evidence. It generates text with those calculated weights, using only Python's standard library.
- `tests/`: synthetic regression cases for source custody, role attribution, calculated weights, deterministic generation, refusal to build without Patrick-attributed voice, and model integrity.

**No Qwen, Ollama, other pretrained backbone, optimizer, gradient updates, neural tensors, commercial service, or AI API is needed.** This is **not a neural transformer or a trained neural LLM**. Mathematical frequency estimation is used to calculate a statistical language model directly from source data; no iterative fitting or model-weight training process is run.

Historical records, including difficult personal history when Patrick includes them, influence **topic weights**. They do not masquerade as his first-person speech or as clinical conclusions about his personality. Public information must be linked to an authenticated source: same-name search results are not automatically Patrick.

## Reproduce locally (private sources only)

Requires Python 3.10+; no third-party Python packages. Place a source manifest **outside this Git repository**, in a private directory under your control. A sample structural manifest (replace with your own local paths; do not commit filled-in copies):

```json
{
  "schema": "JACK_SOURCE_MANIFEST_V1",
  "sources": [
    {
      "id": "my_export",
      "path": "C:\\private\\conversations.json",
      "kind": "chatgpt_export",
      "provenance": "user_role_in_export",
      "sensitivity": "private"
    },
    {
      "id": "life_history",
      "path": "C:\\private\\history.docx",
      "kind": "history_docx",
      "provenance": "self_report",
      "sensitivity": "restricted"
    }
  ]
}
```

PowerShell, from a checkout of this repository:

```powershell
$vault = Join-Path $env:LOCALAPPDATA 'JACK\\private\\20261121'
python -B jack_sources.py --manifest "$vault\\source_manifest_private.json" --output "$vault\\jack_evidence_v1.json"
python -B jack_model.py build --evidence "$vault\\jack_evidence_v1.json" --vault "$vault" --output "$vault\\jack_model_v1.json"
python -B jack_model.py inspect --model "$vault\\jack_model_v1.json"
python -B jack_model.py sample --model "$vault\\jack_model_v1.json" --question "What do you think about easy answers?" --seed 11
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

The outputs must be in the private manifest/model vault. They are created with **no-overwrite** semantics; use a new output filename to rebuild. Neither tool uploads your sources. The model JSON **does contain calculated representations of private source information**, so it must stay private too. Don't put it in a public repo, issue, CI artifact, or shared log.

## Closed-form weights

For Patrick-attributed text, conditional next-word weights are normalized source-weighted counts:

```text
P(next | previous_two) =
    weighted_count(previous_two, next)
    / sum_next weighted_count(previous_two, next)
```

The model uses lower-confidence multipliers for PDF-extracted visual speaker labels, preserving the distinction from native ChatGPT message roles. History adds separately weighted topic frequencies; it is excluded from the voice-transition generator. Every saved model includes source digest receipts and a SHA-256 of its calculated weights.

The output is an **experimental generated text sample**, not a reliable answer about an actual person. No sampling output has permission to act for Patrick or access his accounts.

## Limits and scope

- The available pilot corpus represents **only an inspected subset** of Patrick's history. No claim has been made to ingest every conversation, every trauma record, every database, or all public records. Some medical PDFs are image-only; OCR and complete transcription are separate work.
- With a small personal corpus and a word-trigram architecture, output may be short, repetitive, or incoherent. This version has **not** demonstrated robust reasoning, high-quality dialogue, personality fidelity, generalization, or identity continuity.
- Past records may contradict later corrections. Provenance remains attached to source evidence; this first version does not resolve all historical conflicts.
- Personal history and trauma are included on Patrick's instruction, with clinical statements distinct from his own accounts. They are not used to make unsolicited diagnoses.
- JACK is a **model of Patrick**, not Patrick himself, a legal agent for him, or a holder of his private experiences.

**Public-source rule:** Source code, synthetic fixtures, and this README belong here. Private conversations, medical files, complete biographical archives, calculated model weights, and their derived artifacts belong in controlled local storage.

## Work and verification

The first Lappy private source-calculated run produced a standalone model, and synthetic regression tests are executable from `tests/`. The precise private source inventory, partial extraction coverage, host validation, artifact hashes and test results belong in private receipts—not in this public README.

Next meaningful qualification is a blind comparison against Patrick's withheld real replies and independent review of mistaken attributions and source privacy. Internal green tests do not prove that JACK convincingly mimics Patrick.

**No release to a public weight registry, model promotion, cloud inference route, or service deployment is authorized by this repository.**
