# November21 extension: quick targeted controls, GPT and distinct task trials

The November21 user instruction authorizes2–3 additional usable ImageNet baselines, GPT evaluation of existing baselines, and initial evaluation on COCO captioning/LLaVA open questions. The historical experiment outputs, source locks and paused IAU/V2 runs stay untouched. All added methods are selected for threat-model/compute compatibility **before target scores**, not because previous adaptations scored poorly.

## Additional ImageNet conditions

| Condition | Source and exact revision | Preserved objective/update | Budget and adaptation |
|---|---|---|---|
| AttackVLM MF-ii, transfer only | [paper](https://arxiv.org/html/2305.16934v2), [official code](https://github.com/yunqing-me/AttackVLM), `bdc5c7d34340bdf266e7dbcc3f5583d9621f263a` | Native projected CLIP image-reference cosine ascent; zero initialization, pixel step1 and bounded sign update | Released-code default300 (paper uses100); CLIP-B32; common epsilon16/255 instead of paper examples8/255; HF BF16 execution adaptation |
| AttackVLM MF-it, transfer only | Same paper | Match image embedding to fixed target text embedding, per paper's image-to-text feature objective; same optimizer |Released-code default300 steps (paper100); CLIP-B32; class text fixed before evaluation; **paper-equation implementation, not a claimed imported official MF-it training function** |
| SSA-CWA / CLIP ten-class adaptation | [paper](https://arxiv.org/abs/2303.09105), [official code](https://github.com/huanranchen/AdversarialAttacks), `dbac7e7e32844f440876b57e6c0d3bbb95040a80` | Exact public `SSA_CommonWeakness`, targeted CE, original DCT/inverse DCT and gradient/inner/outer updates | Native10 outer steps,20 spectrum samples/model/step,3 CLIPs,600 source backward computations/image; sigma16,rho0.5,mu1,inner_step_size250,outer_ksi16/255/5; CLIP zero-shot10-class logits replace original classifier architecture |

The AttackVLM paper explicitly uses100-step PGD in Section4.1, whereas the released transfer script defaults to300. This queue keeps the released300-step setting and reports that discrepancy; it must not call300 the original paper budget.

These are **three conditions from two papers**, not three independent papers. AttackVLM victim-query refinement is excluded. CWA's targeted classifier goal matches ImageNet classification, unlike the old Co/SGA joint untargeted retrieval adaptation. Its public SSA variant has the reverse-step block commented out; preserve the actual released core rather than inventing a paper-only stage. All original active formulas/constants are retained. Native step counts are not equivalent compute budgets. Additional conditions are marked adapted where architecture, class-target reference construction, epsilon or BF16 differs.

No expensive generator pretraining, target-query optimization, new proxy composition search, IAU300 or V2K10 campaign is started. Chain of Attack was reviewed but not admitted to the quick queue: its actual method requires ClipCap and iterative caption regeneration; a cosine-only substitute would not be its faithful reproduction.

Two historically used, non-test engineering ImageNet images passed GPU gradient/bounded-PNG smoke for each new condition. All loaded proxy parameters (including CLIP text encoders/projections) were recursively audited BF16/unquantized before first forward. SSA transforms/losses and image gradients useFP32; model parameters remainBF16. CPU tests check native DCT round-trip/gradients and MF-ii update direction/bounds. Full150 uses the existing5×30 GT cohort, common ten-option rubric/parser, exact large target revisions and unconditional target hits/150. No proxy-output gate or target feedback during generation.

## GPT queue: existing frozen attacks

Requested model is `gpt-6.1-sol`, `low`, via the already authorized ChatGPT subscription CLI; local catalog image support and login were reverified. No paid API, model fallback, login change or quota bypass. Evaluate PGD50, MI-FGSM50, M-Attack300, correctedFOA300 and AnyAttack on the existing150 images. Generate no new attack PNGs for this evaluation.

First, balanced10 (two images/direction) checks transport and joins, then expand all five to150 without selecting attacks/parameters from the responses. This prefix is not an independent holdout. One clean answer may be shared only under exact same image/prompt/client/model identity and a preserved verified raw trace. Each image request is a fresh ephemeral, tool-disabled, read-only isolated CLI session; the label is not disclosed beyond the fixed ten-option rubric. Raw JSONL, returned text, parser status, thread ID, hashes and timings are retained. The private server snapshot is undisclosed: do not claim a backend model version or weight precision beyond accepted requested slug/effort.

An initial180s timeout left a manually verified empty zero-byte answer and no scored result; its evidence is preserved under failed_attempts. The transport-only wrapper now captures timeout stdout/stderr with600s allowance and owns/kills its own process group. A scored response is never retried. A genuine quota/auth/unknown incomplete-return failure blocks calls for review; completed responses resume without re-querying.

For a fair current-date comparison, one pre-existing frozen **Ours Uniform4** commercial configuration (generic qwen-style CLIP/InternVL2B/GemmaE2B pool) follows the five-baseline queue. It reuses the same150 clean requests. Its earlier GPT50-assisted selection history stays disclosed; current GPT responses do not select a method or change any attack parameter. Historical GPT numbers are not silently relabeled as current-date paired results.

## Two different new tasks, bounded initial trials

| Dataset/task | Cohort | Methods | Targets | Measurements |
|---|---|---|---|---|
| COCO captioning |10 deterministic images from the previously frozen150 manifest | M-Attack, correctedFOA, PGD; all matched50-step **task trials**, not native300 reproductions | Qwen27B / InternVL14B / Gemma31B | Paired concise captions; official pycocoevalcap CIDEr on this10-image subset; target-caption CIDEr. Raw scale and subset IDF population disclosed, no classification ASR |
| LLaVA-Bench(COCO) open VQA |3 original images /9 unmodified original questions, from the archived30/90 benchmark | Same three50-step trials | Same three targets | Paired raw answers; original reference answer ROUGE-L proxy. **Official GPT4 judge score unavailable; never report ROUGE-L as VQA correctness or an official benchmark score** |

Canvas224, epsilon16/255, fixed prompts/decoding, disjoint target reference images and stable GT manifests remain. Retain original LLaVA question categories/text. No target-answer-based eligibility or source sampling. The new tasks are not converted into ImageNet ten-class ASR. Ours48-reference prototype construction is not silently replaced with a single image and called an unchanged full Ours benchmark result; this first bounded task extension evaluates image-pair baselines only. Full COCO/LLaVA and a separately frozen Ours reference-bank extension remain future work, not claimed completed here.

## Execution, outputs and recovery

Existing single GPU owner/admission locks are reused. Local attacks/evaluations run in fresh separate processes; CPU subscription CLI runs independently without occupying the A100. Complete each attack cohort before loading a final target. Model parameters are strictlyBF16, no quantization fallback; snapshots are pinned and stored on `/ephemeral/kdd2027/hf-cache/hub`. Weights were restored after hibernation. Persistent PNGs, manifests, request records, model audits, losses, logs and state remain under:

`/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction/nov21/`

- GPU queue: `scripts/kdd2027/nov21/local_queue.py`, tmux `kdd2027-nov21-gpu`.
- CLI baseline queue: `gpt_baselines.py`, tmux `kdd2027-nov21-gpt`.
- Same-date Ours control: `gpt_ours_control.py`, waits for baseline queue first.
- CPU report: `report.py --watch --push`, summarizes27 open-weight evaluation cells + five GPT150 baselines + frozen Ours control. It reports partial/blocked truthfully and never starts inference.

The code never restarts historical63-cell, IAU or V2 queues. Valid completed image receipts and exact CLI responses are reused. Interrupted or parser-invalid outputs remain recorded; missing results are not zeros. GPT quota/transport blocking does not cancel independent GPU tasks. Root-disk old image corpora stay available even if ephemeral weights vanish again.

Preliminary projections from smoke: each MF condition roughly tens of minutes; SSA-CWA about1–2hours of generation; target evaluation and distinct-task trials add runtime. These are measured projections, not completion guarantees. Report actual throughput after native full-budget images and do not truncate steps/images to meet an arbitrary estimate. The initial task trials intentionally have smaller predeclared cohorts; full150 ImageNet remains unchanged.
