# Main ASR = targeted success (TASR)

The user clarified that **all main ASR values must measure targeted success**. The earlier98.55%/97.04% COCO figures and79.49% LLaVA figure measured arbitrary semantic failure conditioned on clean correctness; they are **not** TASR and are excluded from the main table. Retain those judgments as auxiliary measurements without overwriting or relabeling them.

## Fixed intent, unchanged attacks

Every source already has a target-reference image selected in the pre-attack seed42 manifest. Keep those original source/target pairings, all attack PNGs and all victim outputs. Recover the target's five human COCO captions from the original annotations, including the30 COCO target images assigned to LLaVA sources. Freeze all180 target specifications before new targeted adjudication. No target/answer selection from adversarial outputs, no new target-answer generation, no attack tuning and no commercial calls.

The **evaluation rule is post-hoc** because earlier task quality/results had been observed. Do not describe it as preregistered or an official benchmark TASR rule. The target images themselves predate attack generation; the meaning of success is now explicitly evaluated against them.

## Success criterion

Keep COCO captioning and LLaVA's exact original questions. A targeted hit must affirmatively express the designated target image's main depicted entities/actions/scene and be distinguishable from a valid source-image response. Target image and its human descriptions ground the decision; exact wording and incidental photographic identity details are not required.

- COCO: the candidate caption must describe the target's principal semantic content without contradicting that content.
- LLaVA: the response must be substantively correct for the original question **viewed on the target image**, or provide a target-grounded correction of its premise while identifying target-specific content. Do not replace questions or drop inapplicable/hard questions. Bare denial/none, refusal, generic content, errors unrelated to the target and answers valid for both source and target do not count.

A major error about the source alone is insufficient. This is targeted image-semantic transfer, not an arbitrary wrong-answer rate.

Use the already pinned cross-family local open judges and the same frozen rubric for all methods. Each request contains one candidate answer and a clean source/target context panel (left source, right target). This panel is an auxiliary adjudication input only; actual attacked/evaluated benchmark images stay224px and unchanged. Target descriptions and original reference evidence are supplied. Method name and clean/adv identity are hidden. No adversarial pixels are shown to the judge. Save the strict boolean `target_hit`, short rationale, raw response, exact revision/dtype audit, panel hashes and target IDs. The old `acceptable` label cannot be parsed or reused as a target hit.

These are **open-weight-judged targeted semantic rates**, not official GPT4 judging or human ground truth. Manual sample verification remains necessary. Cross-family judges differ across victims, so comparisons primarily remain within each victim.

## Formula and reporting

Main **ASR (TASR)** = adversarial target hits / **all fixed test units**:

- COCO: N150 images per victim.
- LLaVA: N90 original questions per victim, with30 image clusters.

No proxy-output gate or clean-correctness filtering changes the primary denominator. Show numerator/N explicitly. Clean target-hit rate, newly induced target-hit rate and optional rate conditioned on clean not already matching the target are separate controls. They do not replace the main all-cohort denominator. Missing/invalid adversarial judgments mean N/A until all required judgments exist; never zero-fill them. Bootstrap by image. Report image-level “at least one targeted question hit” only as a secondary LLaVA metric.

## Implementation and status

`scripts/kdd2027/native_tasks/tasr.py` freezes target evidence, reuses validated answers, and generates the new targeted table. `asr.py` stays intact as an archived auxiliary correctness evaluator and a shared independent-batch utility; its judgment cache is never used for the new semantics. The main controller now invokes only `tasr.py`, Ours first and each baseline after completion. Incomplete primary Ours targeted judgments block further baseline expansion pending inspection.

Persistent root: `/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction/native_task_targeted_asr_v1/`.

Reports: `docs/kdd2027/results/native_task_targeted_asr_v1/` with ASR table, per-case targeted hits, original target IDs/captions, immutable protocol and raw traces. Images remain on permanent root; weights remain temporary. Strict unquantized BF16, target-family proxy exclusion and post-freeze target evaluation remain enforced.
