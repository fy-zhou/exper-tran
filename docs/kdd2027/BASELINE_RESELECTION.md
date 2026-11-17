# Task-matched baseline selection — 2026-11-17

Selection uses attack goal, perturbable modality and target access, not observed success rate. Co-Attack/SGA's original joint text-image retrieval threat model does not match the main targeted image-only transfer task; their archived adaptations are supplemental and not evidence of Ours superiority. No baseline is removed solely because it scored poorly. The earlier63-cell matrix remains historical, paused and unmodified.

| Method | Role | Current evidence |
|---|---|---|
| PGD | Basic optimizer control with explicitly adapted target-image cosine | ImageNet150 complete |
| MI-FGSM | Momentum control with the same adapted objective | ImageNet150 complete |
| M-Attack | Direct targeted image-to-image feature-transfer comparator | ImageNet150 at50/300 steps complete |
| FOA-Attack | Global/local alignment and transport comparator | Corrected-dependency full150 running |
| IAU-FOA | Visual-invariance/transport comparator |10-image300-step pilot only; expensive full run pending |
| M-Attack-V2 | Newer targeted local/detail alignment comparator | Official source selected; not implemented/tested/run |
| AnyAttack | Pretrained targeted generator, separate resource track | Official source/checkpoints available; local compatibility not established |

M-Attack, FOA, IAU and V2 form the direct method-comparison group if validated. AnyAttack has extra pretraining and possibly task fine-tuning, so training cost/data and inference cost are reported separately. PGD/MI are optimizer controls, not faithful reproductions of classical classifier-CE attacks. Do not pretend all seven have identical information or compute budgets.

## Primary-source evidence

- [M-Attack official code](https://github.com/VILA-Lab/M-Attack), [paper](https://arxiv.org/abs/2503.10635): targeted black-box LVLM transfer with source/target-image feature matching. Existing task/BF16 adaptations remain disclosed.
- [FOA paper](https://arxiv.org/abs/2505.21494), [official code](https://github.com/jiaxiaojunQAQ/FOA-Attack): targeted feature alignment. Current rerun changes the incompatible KMeans dependency, not the declared feature objective. Only corrected fully validated results replace old partial records in the main table.
- [IAU-FOA paper](https://arxiv.org/abs/2610.06977): targeted visual-invariance-augmented alignment. Native5 is legal only for Qwen/Gemma targets; InternVL uses the distinctly labeled family-disjoint4 adaptation. A native/adapted mixed row must disclose different proxy resources.
- [M-Attack-V2 paper v2](https://arxiv.org/abs/2602.17645v2), [official implementation](https://github.com/VILA-Lab/M-Attack-V2), inspected HEAD `8faca6d95e1caa190d676575489ab81bb31e4d03`: MCA, ATA, patch momentum and patch-diverse ensemble. Auxiliary target images require provenance and disjointness; preserving the method may require a reference-count track rather than silently collapsing ATA to one reference. Do not execute the upstream paid API pipeline.
- [AnyAttack CVPR2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Anyattack_Towards_Large-scale_Self-supervised_Adversarial_Attacks_on_Vision-language_Models_CVPR_2025_paper.pdf), [official implementation](https://github.com/jiamingzhang94/AnyAttack), inspected HEAD `91b209120c61261a51a45ce217dae7f1daac081d`: pretrained targeted generator with downstream tasks. Official README provides checkpoints and recommends `coco_cos.pt` when task choice is uncertain; this is not proof it is optimal or task-equivalent here. Audit released checkpoint, conditioning, adaptation, perturbation bounds and BF16 before admission. No target-informed fine-tuning.
- [AttackVLM official instructions](https://github.com/yunqing-me/AttackVLM/blob/main/README.md): distinguish transfer-only from victim-query refinement. Only transfer-only could be a later alternative; query refinement violates this campaign's target-isolation contract.
- [InstructTA paper](https://arxiv.org/abs/2312.01886): assumes the victim visual encoder is known. This differs from our current family-disjoint unknown-target setting and is not silently imported into the main comparison.

## Task-specific evaluation and fairness

ImageNet: original150 fixed images, target class and unconditional hits/150. Captioning: retain COCO caption task; define targeted semantic redirection against frozen target-reference captions separately from clean CIDEr degradation. VQA: retain original LLaVA questions; evaluate raw paired answers and explicitly distinguish semantic targeting from official answer-quality judging. No caption/VQA score becomes ten-class ASR.

All methods retain epsilon16/255, frozen cohorts, canonical image canvas and prompts, BF16 parameters, target-family exclusions and post-freeze target evaluation. When native resolution or method-specific auxiliary references conflict, document a separate adaptation/native track rather than silently changing the method. Compare50-step controls and native budgets separately; report actual backward/forward calls, runtime, memory, references and training cost. Ours uses48 references/class; reference-count inequality is a known information-budget difference, not resolved by sharing the test images.

The new selection file is `configs/baselines/kdd2027_selected_baselines.yaml`. It is intentionally outside all active scientific-source hashes. It neither rewrites the historical queue nor schedules new GPU work. Current authorized Co/FOA continuation finishes; Co remains archived supplemental evidence. V2/AnyAttack require implementation audit and non-test smoke before a full run. No complete reproduction or new ASR is claimed for them.
