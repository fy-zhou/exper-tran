# KDD 2027 baseline reproduction — handoff

## Current state (read this first)

This branch is based on `experiment/a100-precommercial-v8-joint-selection-24h`, **not** `main` (which contains no usable experiment code). Existing `src/primary_ml_cka` is the intended integration layer. Do **not** merge directly to `main` without reviewing differences.

Implemented and unit-tested by code inspection only:
- An independent constrained optimizer interface for **PGD** and **MI-FGSM**, in `src/primary_ml_cka/baselines/iterative.py`.
- The optimizer receives a user-supplied **baseline-specific scalar targeted objective**. The current Pull–Push loss must NOT be passed off as a classical PGD or MI-FGSM baseline. Without a documented objective and source of reference images, these modules are optimization primitives, **not end-to-end reproduced baselines**.
- A method-status registry in `configs/baselines/kdd2027_registry.yaml`.
- Unit tests in `tests/unit/test_kdd_baseline_iterative.py`.

**Not implemented**: Co-Attack, SGA, M-Attack, FOA-Attack, IAU-FOA; original loss functions, upstream dependency pinning, end-to-end MLLM evaluation, and any measured baseline TASR. Do not report any new result as completed.

## Mandatory audit before using old Ours outputs

1. Identify the exact branch+commit, resolved configs, and artifacts yielding manuscript Ours TASR. This checkout includes earlier vehicle-10 and CKA-based experiments, and is NOT automatically the manuscript's final 10-class Pull–Push protocol.
2. Confirm whether Ours used 50 or 100 optimization steps, and verify actual image IDs, source/target pairs, prompt, models, epsilon, preprocessing, attack-step loss, proxy success gate, and reference-image count.
3. Freeze and version a machine-readable image manifest. For every method, log eligible count, attempted count, successful targeted outputs, excluded count/reason. Keep denominator consistent.
4. Do not reuse any publication result when its origin or protocol is uncertain.
5. Ensure target model is not called for adversarial optimization or tuning. Frozen target-model evaluation only.

## Priority implementation order

1. Audit existing dataset/model/evaluation contract, create a provenance report.
2. Complete **PGD** and **MI-FGSM**: define a documented, model-compatible targeted image-only objective; verify attack gradients, model preprocessing, and strict L-inf budget on 2 images. Label this `adapted_image_only` where appropriate.
3. Reproduce **M-Attack** from the official repository using its actual loss and transforms; test compatibility; do not silently substitute our code or objectives.
4. Reproduce **FOA-Attack** (global/local features + original alignment) after M-Attack.
5. Audit Co-Attack/SGA compatibility. Their joint text-image methods may not have faithful image-only variants: record `incompatible` if that is the case.
6. Audit IAU-FOA or O-Attack paper, license, code and feasibility; only run with a verifiable implementation.
7. Pilot on non-test images; freeze configs; evaluate the exact same test images as Ours.
8. Report per-pair and per-target TASR, N, confidence intervals, gradient/model-forward budget, runtime, and all exclusions. Ours can lose; never manipulate baseline quality or selection.

## Acceptance checks

Run in a Python 3.11 environment with the repository dependencies:

```bash
python -m pytest -q tests/unit/test_kdd_baseline_iterative.py
python -m ruff check src/primary_ml_cka/baselines tests/unit/test_kdd_baseline_iterative.py
```

Before expensive GPU jobs, test 2-4 images with one documented checkpoint and assert the baseline loss descends under a small perturbation budget. Preserve every source commit, configuration and raw response.

## What to deliver next

- `docs/kdd2027/OURS_EXPERIMENT_AUDIT.md`
- `docs/kdd2027/BASELINE_COMPATIBILITY.md`
- `docs/kdd2027/BASELINE_PROVENANCE.md` with upstream repos, commit SHAs, citations, license, original algorithm details and adaptation notes.
- Per-method configs and end-to-end runners with resumable output manifests.
- Tests that reject unimplemented methods and invalid budget/gradient settings.
- Pilot results separate from held-out full results, without any fabricated numbers.
- Minimal commits on this branch; do not modify paper sources or overwrite original results.

**Fairness is mandatory:** preserve published methods' key components, do not add unauthorized enhancements or intentionally weaken them, no target-informed hyperparameter sweeps, and clearly mark faithful vs adapted vs optimizer-control comparisons.
