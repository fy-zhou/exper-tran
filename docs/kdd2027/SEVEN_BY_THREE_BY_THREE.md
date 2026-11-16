# Seven baselines × three tasks × three target models

User scope:63 formal evaluation conditions. Seven methods: PGD, MI-FGSM,
Co-Attack, SGA, M-Attack, FOA-Attack, IAU-FOA. Three tasks retain distinct metrics;
this is not63 identical targeted-ASR experiments.

| Dataset | Frozen source cohort | Evaluation |
|---|---|---|
| ImageNet | Existing150 images, five directions×30 | Wrong-target hits/N, per direction, clean correctness |
| COCO2017 | Deterministic150 of5000 validation images; disjoint from LLaVA image IDs and target references | Clean/adv official COCO CIDEr implementation on this fixed subset, plus target-caption alignment; not a classification ASR |
| LLaVA-Bench(COCO) | All30 official images, all90 original questions | Clean/adv raw answers, categories; official GPT4 judging remains pending without an authorized configured route. ROUGE-L is only an explicitly labeled overlap proxy |

Each task evaluates Qwen27B, InternVL14B and Gemma31B. Three targets share a
frozen image whenever family rules permit. All models, including auxiliary BERT,
are unquantized BF16 and audited before inference. Classification parsing is
unchanged. New tasks use their original caption/question text, not ten options.
Canvas224 and deterministic decoding are disclosed evaluation adaptations;
LLaVA original images/questions are archived separately. Caption generation uses
max128 new tokens; VQA max1024, with truncation metadata.

Co/SGA imports the official joint text/image generation cores rather than inventing
a cosine replacement. Auxiliary adversarial texts are saved. Evaluating the image
on an MLLM with its original task prompt is an **image-only transfer adaptation**,
not the papers' original joint retrieval benchmark. Their original untargeted goal
must remain explicit; low wrong-target ASR does not establish inferiority for
untargeted robustness. Common eps16/255,50-step ports are disclosed alongside
M/FOA/IAU native300 settings; iterations alone do not match compute.

IAU native5 uses B16+B32+LAION-G+InternVL3-1B+DINO-Base for Qwen/Gemma.
For InternVL target only, a predeclared **family-disjoint4 adaptation** omits the
illegal InternVL surrogate, preserving the core loss/augmentation. It is not called
the original five-proxy configuration. No composition is chosen from target scores.

## Dependency correction, not attack tuning

IAU requirements identify KMeans commit
`f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805`, which supports `iter_limit` and
reinitializes empty clusters. PyPI0.3 lacks these behaviors. Old FOA runs used a
bounded adapter that failed on empty clusters. They remain preserved and must be
labeled as an earlier dependency version. The extension uses the author-pinned
compatible dependency, records the exact source, and **reruns FOA rather than
reusing mismatched old trajectories**. This is not target-score-driven tuning.
PGD/MI/M exact matching ImageNet attacks and target records are hash-verified and
reused, including the already complete native300 M results.

## Continuous queue

```bash
bash scripts/kdd2027/run_7x3x3.sh prepare --minimum-hours 8
bash scripts/kdd2027/run_7x3x3.sh start --minimum-hours 8
bash scripts/kdd2027/run_7x3x3.sh status
```

The controller waits on the existing reproduction owner; it does not kill or
compete with the old FOA run. It tests new Co/SGA/FOA/IAU cores and new caption/VQA
chains on two images before admitting full jobs. Then it runs balanced IAU pilots,
reuses existing controls, covers the new tasks, and completes expensive native300
conditions. Attack/evaluation are separate processes so resize monkeypatches,
proxy weights and attack-only kernel settings cannot enter the target evaluator.
Ordinary restarts reuse valid per-image receipts and exact raw responses.

Eight hours is a **minimum requested work window**, not a promise of all63 results
in eight hours. The full native300 IAU work alone may exceed it substantially.
No sleep/duplicate experiment is inserted to fill time; tasks continue until done,
stopped by the user, or blocked. A code/dataset/judge failure is reported, never
relabeled as completion. Heartbeats, logs and matrix live on permanent root disk:
`outputs/kdd2027_baseline_reproduction/extension/`. Sources/weights use `/ephemeral`.
CPU/static tests do not establish that newly integrated GPU cores have passed;
the live queue records their actual smoke outcomes before full admission.

New raw data preparation is complete: COCO5000 with captions/annotations, official
LLaVA30/90, native answers, source URLs and hashes. Before any extension GPU run,
the provisional derived-JPEG cohort was archived and replaced with lossless PNG
canvases (same source IDs); old experiments are untouched.
