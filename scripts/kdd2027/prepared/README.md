# Active update — 2026-11-17

The user subsequently authorized running all newly selected baselines except IAU-FOA. This supersedes preparation-only instructions below. AnyAttack two-image smoke and full150 generation passed; V2 two-image smoke passed and native300/K10 generation started. Both full-model BF16 audits passed before inference. Target evaluations remain after frozen PNGs, under the unchanged canonical ImageNet rubric.

One `remaining_parallel.py` admission owner took over the previously owned FOA continuation using verified PID signatures; completed FOA receipts are unchanged/reused. Initial admission-wrapper failures occurred before new-method inference, were repaired and preserved. FOA restarts may recompute an interrupted image. `admitted_worker.py` delegates only the two exact existing admission locks from its verified direct parent, with measured-free-memory admission,8GiB reserve and per-job CUDA caps. At mostFOA+one generator execute; initializations are staggered. Target evaluation gets exclusive GPU priority. `protect_foa.py` only signals verified V2 children of that owner and throttles if last-three FOA image times exceed188s vs the measured156s solo baseline. GPU utilization, not filled VRAM alone, determines useful concurrency. No six-hour full-campaign guarantee is made.

V2's original weighted loss collapses the batch axis with `.squeeze()` at batch1. A recorded AST repair retains the batch axis via reshape; CPU tests verify identical original/fixed loss and pixel gradients for batch2, plus finite batch1 gradients. All original loss weights, transforms, native300 steps and K10 remain. This engineering repair is explicit, not a new attack loss. Source-only CLIP-B32 cosine retrieval selects two neighbors within each existing disjoint48-image class bank, rather than the upstream external COCO retrieval pool: this is a task/reference adaptation, recorded per image. AnyAttack uses released `coco_cos.pt`, HF revision4c7172a1ada67ab01e6606ddda4ccacb1362ea4a, SHA256bed1f25d46b4eaef0c3d4eb39c3853b2241991cc048b3b2fff21b7b8c38e0070; no target-feedback fine-tuning or checkpoint search.

New weights are in /ephemeral; extra dependencies are isolated under `/ephemeral/kdd2027/prepared-deps`, not installed over activeFOA dependencies. New logs, manifests, raw responses and checkpoints are on persistent root `outputs/kdd2027_baseline_reproduction/prepared_remaining`. The FOA-stage uploader explicitly excludes these concurrently written files from its stable snapshot. They require a separate result export after evaluation. Historical scientific sources/outputs are not overwritten. Fourteen preparation/admission/loss/publication tests pass, with Ruff clean.

The original preparation documentation follows as historical context.

---

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

## Latest stop scope — 2026-11-17 08:26 UTC

The user paused M-Attack-V2 and requested completion of only corrected FOA and AnyAttack, then end experiments. V2's8 frozen images and interruption metadata are preserved under `prepared_remaining/user_stop_v2.json`; no V2 resume/evaluation or IAU run is authorized. The running owner observed the V2 child terminate and is now waiting for FOA only; its remaining pending list contains AnyAttack. Future owner starts also honor the persisted stop marker. The raw old controller may list the terminated V2 worker as failed; final reporting classifies it USER_PAUSED, not scientific failure.

`publish_final_requested.py` is a CPU-only detached waiter. After the existing owner finishes FOA and AnyAttack evaluation, it validates frozen PNGs, raw row counts/receipt links, BF16 audits; writes the percentage comparison including preserved Ours, all structured evidence and a full persistent local image/checkpoint archive; commits/pushes only its report directory. It launches no inference and does not resume any method. If evaluation fails, the report is PARTIAL_OR_INVALID rather than invented completion. The earlier FOA-stage upload is retained independently.
