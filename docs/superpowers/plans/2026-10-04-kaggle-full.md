# Kaggle Full Notebook Implementation Plan

> **For agentic workers:** Execute inline in this session with test-first checkpoints; no agents, commits, or GPU research runs.

**Goal:** Separate self-contained Kaggle notebook: setup, backbones, training ablations plus combination, inference, locked three-seed final, export in one Run All.

**Architecture:** Existing notebooks stay unchanged. New full_workflow.py restores prior experiment outputs and orchestrates existing APIs. New build_full_notebook.py packages source with existing bootstrap/setup/EDA cells and unchanged eval.py.

**Tech Stack:** Python, unittest, nbformat, PyTorch, existing lab modules.

### Task 1: Tests
- [ ] Add code/tests/test_full_workflow.py: notebook syntax and payload identity; restoration/conflicts; validation-only combination; stage ordering and lock-before-test; restart without re-selection; consent and seeds.
- [ ] Run new tests and confirm missing implementation failures.

### Task 2: Workflow
- [ ] restore_outputs(source, target): require B01–B05 config/summary; copy experiment artifacts only, compare existing files by checksum and reject conflicts before writes. Omit old environment/EDA/split-check caches.
- [ ] combined_config(configs): best nonbaseline augmentation and loss by validation; T09 interaction recipe with provenance, never test-based selection.
- [ ] run_all(base, allow_final_test, seeds): consent/three seeds first, backbone group, ablations/T09, inference, locked validation selection, final/export. Reuse existing lock without re-selection. Resume incomplete epoch checkpoints.
- [ ] Make baseline reuse ignore resume control flag via existing validate_saved_config.

### Task 3: Packaging
- [ ] Build a new notebook from existing standalone cells without writing original notebooks. Embed maintained modules and full_workflow, preserving eval.py bytes.
- [ ] Configuration: explicit final-test consent, optional previous output directory, CUDA required. Setup/EDA/selftests then run_all then original evaluator cell.
- [ ] Generate code/lab_day2_kaggle_full.ipynb; validate nbformat and compile all code cells/payloads.

### Task 4: Verification
- [ ] Run new/existing unittest suites and selftest; inline review restoration, test leakage, locking, checkpoint reuse and packaging.
- [ ] Update code/README.md with full run instructions and session/time/storage limits.
- [ ] Report actual checks, not Kaggle/GPU research results. Preserve eval.py/starter/original notebooks; no commit.
