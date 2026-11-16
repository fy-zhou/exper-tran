# Baseline compatibility and execution scope

| Method | This implementation | Fidelity / limitation |
|---|---|---|
| PGD | Targeted sign descent, random start, L∞ projection | **Adapted image-only** loss: mean projected CLIP cosine distance to one target reference image. Not a classifier-CE reproduction; no Pull–Push substitution. |
| MI-FGSM | Per-image normalized gradient, momentum1, sign descent | Same explicit adapted loss; no random start. Abs-mean normalization differs from L1 sum by a constant pixel count, leaving sign/momentum trajectory equivalent for a fixed canvas. |
| M-Attack | Pinned official `EnsembleFeatureExtractor` and `EnsembleFeatureLoss`; official iterative sign-ascent updates and source/target crops | Upstream core, task adapted to frozen class-target images, BF16, image-derived seeds and our prompt. Original 300 steps reported separately from matched50. Default B16+B32+LAION-G retained. |
| FOA-Attack | Pinned official global/local extractor, OT loss and dynamic ensemble history; fixed K10 from official generation entrypoint | Upstream core, same task/precision adaptations. KMeans dependency API repair explicitly disclosed. No target-feedback K3→K5 loop from `FOAttack.py`. Original300 and matched50 separate. |
| Co-Attack | Audited, not substituted | Original Co-Attack jointly perturbs text/images for retrieval/entailment/grounding. Its image-only mode is a **different** algorithm; not a faithful unchanged targeted ten-class comparison. |
| SGA | Audited, not substituted | Set-level image/text retrieval guidance and text perturbation require paired captions, retrieval positives/ranks and VLP evaluation. An image-only class-target rewrite would need an explicitly separate adaptation protocol. |
| IAU-FOA | Official source pinned for review, not yet reproduced | Public code and self-citation exist; publication identifier and explicit license grant are not verified. No result claimed. |

Only PGD/MI/M/FOA are accepted by the executable method dispatcher. Unsupported
names raise an error before GPU inference. No baseline is silently replaced with
another method. No external evaluation/API entrypoint is executed.

Fairness: same images, epsilon, canvas, prompts, denominator and target revisions;
full BF16 in every role. Optimizer controls use50 steps. M/FOA additionally have
native300-step runs. Runtime and actual model forwards/backwards are retained:
matching iteration count alone does **not** match compute. Baseline models/losses
are not modified to mimic Ours. Baseline failure is not evidence of Ours superiority.

Upstream FOA's source uses local token normalization along **token axis dim1**.
The adapter preserves it. It also calls the global diagnostic before its cropped
local objective; this changes dynamic loss history, so the call is retained.
Transport plans are detached as in source. Its `iter_limit=100` KMeans argument is
unsupported by the published dependency0.3. The adapter implements the requested
bounded Euclidean Lloyd iterations, same NumPy initialization, detached assignments
and differentiable centroid means. Empty clusters/nonconvergence fail explicitly;
there is no random retry, cluster-count search, dropped local term or target tuning.
This engineering repair and FP32 feature statistics mean the run is not claimed
bitwise-identical to an unspecified upstream software environment.

Frozen test: V8 final150. Debug smoke: two old development images; pilot: old
screen10 (already used historically, **not untouched**). New outputs are isolated
under `/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction/`. Large model
weights reside only in `/ephemeral/kdd2027/hf-cache/`; results do not.
