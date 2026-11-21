# Ours first: native COCO captioning and LLaVA open QA

The user clarified that each dataset must retain its original task and that valid Ours measurements take priority over baseline expansion. There is no requirement to manufacture a higher Ours score or exclude a stronger, applicable baseline. ImageNet correctedFOA300/M-Attack300 remain strong main comparisons; near-zero MF variants are additional controls, not replacements selected to favor Ours.

## What changes and what stays fixed

COCO evaluates descriptions; LLaVA evaluates the original open questions. Neither becomes ten-class classification or a prescribed wrong-answer hit test. The feature attack still moves toward an independently fixed target image's semantics, but task effectiveness is assessed by native output quality rather than classification ASR.

The original ImageNet48-image **class** banks cannot be silently reused as if they describe arbitrary COCO scenes. This study explicitly introduces a **label-free reference-neighborhood adapter**:

- A fixed512-image COCO pool, disjoint by original image ID from all150 COCO and30 LLaVA test images. It contains the180 previously frozen target-reference images plus332 seed42-selected public COCO val2017 images. No target-model predictions enter pool selection.
- A fixed, pinned BF16 CLIP-B32 selector embeds this pool, source image and the already frozen target-reference image. Select48 target-nearest references, then48 source-nearest references excluding those48. All images in each bank are distinct and both banks are disjoint. Selection is completed before target evaluation and is identical for all proxy families.
- Each participating proxy/layer forms its own unit-normalized mean prototype from the48 selected reference embeddings. This is a neighborhood prototype, **not a claim that48 images share a human class label**. Reference homogeneity and scene complexity are limitations; successful transfer is not assumed.

Keep the frozen Uniform4 recipe: quarter depths25/50/75/100%, native spatial-token masked mean, .5/.5 cosine Pull/Push, layer-loss average within each proxy, equal per-image L1-normalized proxy image gradients, momentum1, DI .8–1.0 resize/pad,50 steps, alpha1/255, epsilon16/255, canvas224, random start and deterministic seed42-derived per-image streams. No language/output-head loss, patch term or target feedback is added. This is an explicitly declared cross-task reference construction adaptation, not an unchanged rerun of the ImageNet protocol.

|Evaluated target|Legal Ours proxy pool|
|---|---|
|Qwen27B|CLIP-L/14 + InternVL2B + Gemma E2B|
|InternVL14B|CLIP-L/14 + Qwen2B + Gemma E2B|
|Gemma31B|CLIP-L/14 + Qwen2B + InternVL2B|

All floating model parameters stay audited BF16/unquantized, including unused language parameters on CPU. Only visual modules participate in attack gradients. Final targets load in separate processes after adversarial PNGs are frozen.

## Native metrics and limitations

**COCO150:** original concise-caption prompt, five human captions/image, source-caption CIDEr, paired clean/adversarial outputs and per-image scores. Report image-bootstrap quality-drop intervals at fixed reference/IDF population. A positive clean-minus-adversarial score means deterioration. Compare only identical cohorts/IDF populations, and do not relabel CIDEr decrease as targeted ASR. Retain truncation/token records.

**LLaVA30/90:** preserve all original questions, categories and reference answers, with the explicit224px input adaptation. ROUGE-L remains diagnostic only. Add a frozen, local, blinded paired image-aware judge: Gemma31B judges Qwen/InternVL answers; Qwen27B judges Gemma answers. For each question the judge sees the clean image, original question/reference and randomly ordered candidate answers, without method/clean/adv labels. Score independent factual accuracy/relevance/coverage from0 to10. Deterministic candidate order is shared across methods. Record exact judge revision, BF16 audit, prompt/source hashes, raw response and strict parse. Missing/invalid scores remain N/A. Bootstrap by image rather than treating three questions as independent.

These are **adapted open-weight judge scores, not official GPT4 benchmark results or human ground truth**. The two judges differ across evaluated targets, so scores should be compared within a target, not casually pooled as one universal accuracy scale. Official GPT4 judge stays uncalled; no paid API/subscription CLI is invoked. Judge outputs never tune attack settings. Human review and official judging remain limitations for a final benchmark claim.

## Priority and execution

1. CPU preparation: pinned proxy weights on temporary disk, disjoint reference images/manifests on persistent root.
2. Wait for existing ImageNet SSA-CWA owner; do not interrupt it or create competing GPU controllers.
3. Freeze all reference-bank selections.
4. Two-image/two-step engineering attacks for all three legal pools: actual layer taps, finite/nonzero gradients, dtype and decoded PNG bounds. These are excluded from final results.
5. Ours pilot: COCO10 and LLaVA3/9 questions, all three targets,50 steps and full task scoring.
6. Ours full: COCO150 and LLaVA30/90, all three targets, reusing matching pilot PNGs. Advancement depends on engineering/scoring completeness, **not selecting a method by final-target scores**. If quality does not deteriorate, report that failed generalization honestly rather than tune on those same images.
7. Only after Ours full raw answers and task metrics are complete, resume the frozen PGD/MI/M-Attack/FOA baseline controls. Score LLaVA baselines with exactly the same open-judge protocol after each condition. Optional300-step rows must fit the original authorized24h deadline; no weakening/shrinking to claim completion.

Detached controller: `kdd2027-native-ours`.

```bash
PYTHONPATH=src /home/ubuntu/exper-tran/.venv-a100/bin/python -u \
  scripts/kdd2027/native_tasks/campaign.py
```

The previously waiting baseline-only controller is stopped, its state retained as `QUEUED_AFTER_OURS_TASK_VALIDATION`. Reuse `reproduction_owner.lock` and the existing per-worker GPU lock. No IAU, M-Attack-V2, GPT CLI or commercial route is started. Dataset/runtime errors stop that branch with logs and preserved receipts. A daemon does not invent algorithm fixes.

Persistent output root: `/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction/native_tasks_ours_first/`. Compact reports/raw structured evidence export to `docs/kdd2027/results/native_tasks_ours_first/`; all PNGs/reference images remain on root disk. The full ImageNet/old results and paper are untouched. Full completion and positive attack effect are separate claims; neither is asserted from code or smoke tests.
