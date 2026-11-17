# Prepared-only M-Attack-V2 / AnyAttack reproduction adapters

No new attack, target evaluation, weight download or model initialization was run for this preparation. CPU/static checks only. Existing FOA source hashes/queue are untouched. These files are **not yet GPU-validated reproductions**; do not add them to an automatic campaign until non-test scientific smoke passes under future authorization.

## What is implemented

- `reproduction.py`: pinned clean-upstream verification; exact reviewed upstream core classes imported through a restricted AST loader, avoiding upstream API/logging/config entrypoints. Adapter cannot silently replace their losses with Pull/Push.
- V2: official weighted primary+two-reference alignment, native Adam learning rate0.005/Adam beta1=0.9,300steps/epsilon16,10-pass gradient accumulation, source random crops and mild target resize/flip/rotation; four pinned CLIP backbones including LAION-B32/G14. HF unprojected pooler features match the official extractor, rather than the old projected-cosine adapter. Default FP16 autocast/scaler is disabled to meet strict BF16 weight protocol.
- AnyAttack: official OpenAI CLIP-B32 encoder and exact official Decoder class/state-dict, target-image-conditioned noise, clamp16/255 and pixel clip. Only explicit checksum-verified local weights may load; no implicit download or fallback. BF16 activation bridges handle the upstream Decoder.float() FC input and OpenAI LayerNorm FP32 accumulation; stored parameters remain BF16 and recursively audited before use.
- `prepare_or_generate.py`: default non-inference source/dependency/manifest check. An explicit future `--generate` creates only frozen adversarial PNGs/atomic receipts, under the shared GPU admission locks; deterministic per-image seed, input/ref hashes, exact sources, bounded decoded pixels, resume and precision evidence. It does not evaluate final targets or call commercial APIs.

## Upstream discrepancies requiring honest validation

Pinned V2 public code implements multipass Adam and weighted pooler alignment. Do not invent an extra standalone patch-momentum routine just because it is discussed in the paper. Exact correspondence to the paper's full method remains an audit item. The official parallel runner forwards configured K10, whereas the single-file runner omits K and uses the attack function default5; this adapter follows parallel/config K10, explicitly. It rejects missing/duplicate retrieved targets instead of silently padding primary images, and records this stricter preparation requirement.

No official local retrieval cache for the frozen150 target references has been built yet. V2 requires two source-only retrieved references with recorded provenance, disjoint from all attacked images. Its extra reference information is distinct from the existing single-target-image baselines and Ours48-reference banks.

AnyAttack requires an actual released fine-tuned decoder checksum and the official OpenAI B32 JIT checkpoint. No checkpoint has been chosen based on final-target feedback, downloaded or loaded. Do not substitute an HF CLIP projection and call it the exact OpenAI encoder. Decoder architecture/dtype compatibility and original generation fidelity still require a later GPU smoke.

The preparation check currently reports missing `einops`/`omegaconf` in the selected environment; do not change dependencies while FOA is active. All adapters fail rather than substitute models or precision. No integration into the existing63-cell frozen queue or evaluator has been performed; later use needs explicit evaluator receipt mapping and a separately frozen study version.

## CPU-only checks (safe during FOA)

```bash
CUDA_VISIBLE_DEVICES='' /home/ubuntu/exper-tran/.venv-a100/bin/python -m unittest discover -s scripts/kdd2027/prepared/tests -v
CUDA_VISIBLE_DEVICES='' /home/ubuntu/exper-tran/.venv-a100/bin/python scripts/kdd2027/prepared/prepare_or_generate.py --method m_attack_v2
CUDA_VISIBLE_DEVICES='' /home/ubuntu/exper-tran/.venv-a100/bin/python scripts/kdd2027/prepared/prepare_or_generate.py --method anyattack
```

The12 checks validate source compilation/pins, import isolation, reference hashes/count/disjointness, no silent padding, and publication guards against partial/missing evidence. They do not test attack effectiveness, gradients or GPU numerics. No full reproduction success is claimed.

Packet schema: `images` is a nonempty list of unique `image_id`, `clean`, `clean_sha256`, and `target_references` (`path`, `sha256`). V2 additionally requires `retrieval_provenance`; it uses exactly3 distinct references, AnyAttack exactly1. Paths must point to canonical224 RGB images. Future generation requires `--packet`, `--output`, explicit `--generate`, plus checksum-locked local AnyAttack weights when applicable. Do not execute generation now.

## FOA-stage publication

`../publish_foa_stage.py --wait --push` is CPU-only. It waits for corrected FOA150 plus all three target evaluations, or for the narrow runner's explicit failure record; exports an honest complete/partial report, all structured raw evidence, full local archive, validates PNGs, and commits/pushes only its report directory. It does not run or restart experiments. The watcher lives in detached tmux `kdd2027-foa-publish`; errors remain logged and local evidence survives failed push. After this stage no new GPU job starts automatically.
