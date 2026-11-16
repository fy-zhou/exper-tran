# Upstream provenance — 2026-11-16

All repositories were fetched from their authors' public repositories. Code is
kept in external clean checkouts, not copied into Ours or bundled into this repo.
`prepare_upstreams.py` pins commits; workers recheck HEAD and tracked-file status.
The small adapter imports only reviewed feature/loss modules, never API scripts.

| Method | Primary source | Pinned upstream commit | License evidence |
|---|---|---|---|
| PGD | [Madry et al.](https://arxiv.org/abs/1706.06083), targeted constrained optimization adaptation | Independent constrained optimizer; not imported | Local project code |
| MI-FGSM | [Dong et al.](https://arxiv.org/abs/1710.06081), momentum update | Independent constrained optimizer; explicit image-only objective | Local project code |
| M-Attack | [Paper](https://arxiv.org/abs/2503.10635), [official code](https://github.com/VILA-Lab/M-Attack) | `9e0cc64a22dff799fa84c12a4e51934d1e849e29` | MIT LICENSE present |
| FOA-Attack | [NeurIPS2025 paper](https://papers.neurips.cc/paper_files/paper/2025/hash/540a3bf4863672cdaebfed69fa2ce335-Abstract-Conference.html), [official code](https://github.com/jiaxiaojunQAQ/FOA-Attack) | `6aa9ad463a1e812b7f77ff2e5c4b1712e7c78c4b` | README Apache2 badge; no standalone LICENSE located. Do not redistribute source as if grant independently verified. |
| Co-Attack | [Paper](https://arxiv.org/abs/2206.09391), [official code](https://github.com/adversarial-for-goodness/Co-Attack) | `96f2c0ebf743f600860ed7aa78c14b38c8883521` | GPL3 LICENSE present |
| SGA | [Paper](https://arxiv.org/abs/2307.14061), [official code](https://github.com/Zoky-2020/SGA) | `f49e6926cd13148e49bdbc2c3e537660e096230c` | MIT LICENSE present |
| IAU-FOA | [Author repository](https://github.com/jiaxiaojunQAQ/IAU-FOA) | `749f349da249bd933ace4d33776ee59249b8d2a3` | No explicit LICENSE found; paper publication identity not independently verified |

Reviewed M core:
`generate_adversarial_samples.py::fgsm_attack`,
`surrogates/FeatureExtractors/Base.py::{EnsembleFeatureExtractor,EnsembleFeatureLoss}`.
Local-to-local target-image cosine matching, source/target RandomResizedCrop224
with area[.5,.9], mean loss across B16/B32/LAION-G, iterative delta sign ascent;
zero initialization, alpha1/255, epsilon16/255. Official default300 steps.
The upstream function named `pgd_attack` uses Adam; it is **not** substituted for
our separately labeled sign-PGD optimizer control.

Reviewed FOA core:
`generate_adversarial_samples_foa_attack.py::fgsm_attack` (fixed K10),
`Base.py::{EnsembleFeatureExtractor_ot,EnsembleFeatureLoss_OT_foa_attack}`.
Global last-hidden CLS cosine plus .2 times clustered local Sinkhorn alignment,
uniform marginals, entropic epsilon.1, at most100 Sinkhorn iterations with
mean-r change threshold.01; detached transport. Per-model previous-score ratio,
softmax temperature1 scaled to number of models. Preserve both global and local
calls and update order. See compatibility document for KMeans API repair.
`FOAttack.py` is deliberately not executed: commercial response-conditioned
cluster escalation violates our evaluation-only target protocol and its FGSM
function also lacks a return in this pinned revision.

Verified model pins:

| Alias | Repository | Revision |
|---|---|---|
| B16 | openai/clip-vit-base-patch16 | `57c216476eefef5ab752ec549e440a49ae4ae5f3` |
| B32 | openai/clip-vit-base-patch32 | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| Laion | laion/CLIP-ViT-G-14-laion2B-s12B-b42K | `4b0305adc6802b2632e11cbe6606a9bdd43d35c9` |

Load complete CLIP parameters in BF16 and audit recursively before first inference,
including unused text tower. Projected pooled embeddings for M/PGD/MI; unprojected
last hidden CLS/patch embeddings for FOA, following source. Resize→clamp/255→center
crop→CLIP normalization; **no Ours straight-through PNG quantization** inserted.
Loss/statistical FP32 is recorded as an adaptation for strict BF16 weights.
Targets inherit exact V6 model revision pins and canonical prompt/parser.

Runtime environment used for tests: Python3.11, torch2.6.0+cu124,
transformers5.14.1, kmeans-pytorch0.3 (only newly installed dependency).
The upstream complete requirements file was **not** installed into the frozen
MLLM environment; it includes incompatible older Transformers and unrelated API
clients. Tests compare M loss gradients and complete crop/update loop against the
pinned official function; FOA transport and centroid gradient checks are separate.
