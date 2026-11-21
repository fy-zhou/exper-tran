# Non-GPT task expansion: approximately 8–10 hours

User requested an 8–10-hour non-GPT run after the November21 restart. The walltime target is an estimate, not a fabricated minimum runtime. No commercial requests, including a LLaVA GPT judge, are admitted. No Ours method is changed.

## Queue and budget

The current ImageNet MF-it/SSA-CWA queue retains priority. A new resumable task queue waits for the **same** `reproduction_owner.lock`; every inference child also uses the existing GPU lock. It cannot compete with or duplicate the active queue. Start time/deadline are persisted on root disk; the ten-hour budget includes time waiting for ImageNet. Required work already admitted finishes its current phase; future work remains resumable when the window ends. Runtime may exceed the target if an active phase takes longer than estimated. There is no idle padding to manufacture eight hours of compute.

|Priority|Dataset/task|Images / questions|Methods|Steps|Targets|
|---|---|---|---|---:|---|
|P0|Finish ImageNet classification|150 each|MF-it, SSA-CWA|released300 / native10 outer|Qwen27B, InternVL14B, Gemma31B|
|P1|COCO captioning|150 /150|PGD, M-Attack, corrected FOA|50 each|same three|
|P1|LLaVA-Bench(COCO) open questions|30 /90|PGD, M-Attack, corrected FOA|50 each|same three|
|P2, if remaining budget covers estimate plus reserve|COCO and LLaVA|same full cohorts|MI-FGSM|50|same three|

The six P1 method/dataset rows give 18 target-evaluation cells, 450 attacked images, 1,350 adversarial captions and 810 adversarial question answers. Clean answers are reused only for identical model revision, clean image hash, prompt and generation limit. Previous ten/three-image prefixes are imported only after validating source locks, exact scientific settings, deterministic per-image seed, image/reference identities, PNG hashes, and response identities. Native50 task trials are explicitly **not** claimed to be original M-Attack/FOA300 reproductions. The existing ImageNet300 comparison remains separately available.

Measured prior median attack times at50 steps: PGD9.5s, M-Attack17s, FOA24.3s/image. Three full COCO attacks therefore take about2.1h before target inference. LLaVA90-question generation is more expensive than ten-option classification, especially the first clean-answer pass. Remaining ImageNet about2h plus full COCO and LLaVA give an approximately8–10h campaign; no unverified promise of full completion is made. Optional MI is admitted only if its estimate fits. No precision/model/sample/step reduction is allowed under memory pressure.

## Attack and evaluation contracts

Each source image has a disjoint, predeclared COCO target image, selected without model-output screening. The baseline optimizes its own published/adapted image-pair feature objective through frozen CLIP surrogates. This is targeted **image-representation transfer**; no invented target answer is supplied to the evaluator. Pixel canvas224, L-infinity16/255, seed42 with per-image derivation, pinned unquantized BF16 model weights, original baseline updates, and post-freeze-only target evaluation remain unchanged. All six core rows use a matched50 optimization-step budget; actual forward/backward costs still differ by method and are retained.

**COCO:** seed42 selection of150 distinct val2017 images, excluding all LLaVA image IDs, with five original human captions each. Prompt: `Provide a concise, single-sentence caption describing this image.` Max128 new tokens, same decoding for clean/adv. Primary descriptive measurements: clean versus adversarial CIDEr against source annotations; also CIDEr against the target-image captions. The same150-image reference/IDF population is used across all complete methods. Earlier10-image CIDEr values are not directly comparable to150-image absolute scores because the IDF population differs. These metrics measure caption quality and target-caption alignment, **not classification ASR**. Record raw units, complete/partial denominator, paired outputs and original IDs.

**LLaVA-Bench(COCO):** all30 original images and90 original questions, preserving conversation/detail/complex categories and reference answers from pinned upstream revision. Identical questions are asked separately on clean and adversarial224px images, max1024 new tokens. This224px input is an explicit protocol adaptation. Preserve all raw answers. ROUGE-L versus the original reference is only a lexical-overlap diagnostic: it cannot establish correctness, attack ASR or semantic failure, and longer/paraphrased answers can change it. Official GPT4-judge results remain `N/A / NOT_CALLED_NON_GPT_CAMPAIGN`. No unofficial metric is relabeled as an official LLaVA score. Human review or a separately approved, clearly labeled judging protocol is needed for strong task-success claims.

## Run and recover

Already started in detached tmux `kdd2027-nongpt10h`, waiting for the active owner when necessary:

```bash
PYTHONPATH=src /home/ubuntu/exper-tran/.venv-a100/bin/python -u \
  scripts/kdd2027/nov21/dataset_expansion.py run --hours 10
```

The exact same command resumes the original deadline and frozen outputs. It does not reset a time window or repeat validated image inference. Completed failures/missing queries remain explicit. No GPT launcher is invoked. The final raw-completeness check is distinct from LLaVA judge availability.

Persistent full outputs: `/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction/nov21_dataset_expansion/`. Weights remain only on `/ephemeral`. Compact reports and structured evidence are exported to `docs/kdd2027/results/nov21_dataset_expansion/` and pushed on completion; PNGs remain on persistent root disk. Reports preserve actual complete/partial counts and do not turn missing results into zeros.

## User-authorized extension:24 hours (supersedes the ten-hour budget above)

The user extended the experiment window to one day. Deadline is **2026-11-22 18:01:01 UTC**, measured from the original campaign start; it is not silently reset on restart. The existing waiting controller alone was replaced; active ImageNet SSA-CWA was left running. Historical outputs and the attack/evaluation implementation are unchanged. Only prepared dataset-prefix imports existed before this scheduling update; their original November21 receipts remain preserved and are revalidated/imported into the new source-versioned run. There were no new dataset attacks to discard.

Priority after the three-method matched50 core: complete MI-FGSM50 on COCO150 and LLaVA30/90 (both, all three targets), then attempt native300 M-Attack and correctedFOA on both full cohorts. Native300 optional runs are admitted only if a conservative generation/evaluation estimate plus reporting reserve fits. Primary matched50 rows are kept separate from native300 rows. Optional missing300 cells remain queued/N/A, never zero or claimed complete. Maximum expanded task matrix:12 method/dataset/step conditions ×3 targets =36 target cells. More hours do not authorize target-informed tuning or GPT calls.

Detached controller: `kdd2027-nongpt24h`. Resume with the same command and `--hours 24`; the persisted authorized deadline is retained. BF16, epsilon, references, manifests and deterministic per-image seeds remain fixed. CUBLAS workspace is explicitly set to the existing baseline-queue value. Results are source-versioned; reports ignore obsolete preparation-only versions rather than merge them.
