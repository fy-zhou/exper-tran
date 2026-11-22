# CrossVLAD5x2 and LLaVA-Bench10 protocol smoke

Approved scope: Ours then three meaningful baseline candidates on each task,
three fixed large open-weight targets. This is exploratory smoke, not final ASR.

```bash
bash scripts/kdd2027/protocol_smoke10/run.sh --start
bash scripts/kdd2027/protocol_smoke10/run.sh --status
```

Execution is detached in tmux `kdd2027-protocol-smoke10-v2` and shares the existing
`reproduction_owner.lock` and `gpu_owner.lock`. A second GPU controller is not
introduced. Rerunning resumes frozen validated PNGs/evaluations; logs are persistent.
Outputs: `/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction/protocol_smoke10_v2/`.
Weights and raw downloads only are on `/ephemeral`.

## Frozen cohort

CrossVLAD: bicycle->motorcycle, car->bus, cat->dog, chair->couch, tv->laptop;
two images/direction. Official labels are not reauthored. Sampling is seed42
per direction, before any model output. The initial proposed laptop->tv was
not in the downloaded release; use its published reverse direction instead.

The official3000-sample release contains58 distinct directed pairs; the paper
lists79 possible pairs. Report actual released counts, not an invented79-pair run.
The released set also includes cases not meeting the paper's stated caption/area
criteria. Version1 was superseded BEFORE any new GPU attack; its manifest is
preserved under `protocol_smoke10/`. Version2 rechecks >=3/5 caption mentions
(frozen aliases), 10%-50% object-box area, and original COCO source-present /
target-absent annotations. No target output participates in filtering.

For Ours, sample48GT references for each side of each direction from independent
COCO val2017 images, excluding all released CrossVLAD sources, historical COCO150,
and LLaVA30. Banks exclude the opposite category; the union has449 images.
Reference IDs, aliases, annotation hashes, all sampling rules and exact official
labels are frozen. This is an independent class-bank construction, not an official
CrossVLAD-provided prototype. Whole-image spatial features are retained.

LLaVA: existing manifest first10 original images /30original questions, all three
question categories retained. There is no official five-direction split here.
Existing Ours50-step neighborhood48 outputs are reused only after exact source,
reference-bank, row, model-revision, receipt and raw-evaluation integrity checks.
They remain labeled as the historical neighborhood adapter, not new class48 attacks.

## Methods and budgets

- Ours:1:1 Pure PP, four quartile native layers, per-proxy L1 normalized equal
  gradient fusion, DI,50steps, seed42, alpha1/255, epsilon16/255,224canvas.
  Family-disjoint3proxy pools are unchanged; target access only after PNG freeze.
- M-Attack: pinned official feature core, native300step budget, fixed target-class
  exemplar. No target-output retry or selection.
- FOA: pinned core and corrected pinned KMeans dependency, native300steps,
  same exemplar as M-Attack. Native crop/transport/history semantics retained.
- SSA-CWA: official optimizer core,10outer steps /20spectrum samples, CLIP ensemble
  COCO80 zero-shot classification head. This is an explicit task adaptation,
  supported by prior ImageNet nontrivial results but NOT a claim of prior COCO
  effectiveness. On LLaVA, the attacked class is the largest GT object in the
  previously frozen target image, chosen without model output.

PGD/MI-FGSM and weak AttackVLM variants are not used to fill the three main
candidate slots. No new strong result is promised; a zero or failed candidate is
reported honestly. Reference counts, text guidance, model pools and compute differ;
this smoke is not a fully resource-matched ranking. In particular, SSA is a
classifier-to-caption transfer control, not a faithful caption-specific attack.
All open model floating parameters must pass BF16/unquantized audits.

## Scoring and genuine external blocker

CrossVLAD Captioning: exact official lower-case substring rule, target category
present AND source category absent. Report clean hits, adv hits /10 and each /2.
Natural caption prompt /224canvas are disclosed adaptations for chat MLLMs;
Florence's `<CAPTION>` control token is not sent to other architectures. No CTSR
or full four-task benchmark claim is made. Preserve raw text so substring/negation
limitations remain inspectable.

LLaVA COCO: pin original released questions, GPT4 reference answers, reviewer
`eval_gpt_review.py`, and `rule.json` at commit
`c121f0432da27facab705978f83c4ada465e46fd`. Construct exactly reference-first
official prompts, temperature0.2, historical model `gpt-4`. The separate later
In-the-Wild reviewer uses `gpt-4-0314`; do not confuse the two scripts. Summaries
use100*mean(candidate score)/mean(reference score), by category and overall.

No OpenAI API credentials/approved paid route or verified historical GPT4 access
is configured. The subscription CLI is not silently substituted. Request packets
are prepared/deduplicated, but official scores stay N/A / BLOCKED until this
external dependency exists. No paid request is made by this campaign.

## Primary sources

- [CrossVLAD paper](https://arxiv.org/abs/2507.07709),
  [official code/data](https://github.com/Gwill-Z/CRAFT), revision
  `0f457d73b049d62111d39c69d893f2f7d10574b9`.
- [LLaVA official COCO reviewer](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/eval/eval_gpt_review.py).
- [M-Attack](https://github.com/VILA-Lab/M-Attack),
  [FOA](https://github.com/jiaxiaojunQAQ/FOA-Attack),
  [SSA-CWA](https://github.com/huanranchen/AdversarialAttacks).

Tests: `PYTHONPATH=src .venv-a100/bin/python -m pytest -q
scripts/kdd2027/protocol_smoke10/test_protocol.py` (use actual environment path).
