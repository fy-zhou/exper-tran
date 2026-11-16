# Published Ours artifact audit — 2026-11-16

This audits the preserved **V8 publication artifacts**, not an assertion that an
unsupplied manuscript contains the same rows. Do not label earlier vehicle10,
V6 conditional-TASR, or V7 margin+patch outputs as the final Ours method.

Source branch: `experiment/a100-precommercial-v8-joint-selection-24h`.
The baseline branch was fetched from `pe0-p1e/exper-tran`, starting at
`7007cdb400c9cad93af2199f47821272487a7747`. The old preflight records an initial
V8 checkout at `4b20d5d8f79e0502a588e21afe9ce5323cf6be0a`; it is not proof that
all later campaign files were identical. Preserve scientific-source hashes in
`study_lock.json` and per-condition receipts rather than assigning all results
to that initial checkout. No Ours source or result is modified by this baseline work.

Verified local evidence:

- Root: `/home/ubuntu/exper-tran/outputs/a100_precommercial_v8_joint_selection24h/`.
- `paper_closeout/final_method_lock.json`: nine conditions = three methods
  (uniform4, selected4, FDA4) × three target families.
- `manifests/final150.json`: 150 fixed GT-selected images, five transitions ×30.
- Read-only verification of **1,350 attack receipts** passed: complete status,
  **50 steps**, no target access during attack, valid clean/adv SHA256 and
  saved-PNG L∞ ≤16/255. These are not 100-step results.
- Machine-readable audit: new output root `audits/ours.json`; regenerate with
  `PYTHONPATH=src python scripts/kdd2027/audit_ours.py --output <new-audit.json>`.

Exact final attack contract: 224px canonical canvas; epsilon16/255; alpha1/255;
50 steps; momentum1; random start; seed42 with image/stream hashing; DI enabled;
strict BF16 unquantized model weights; FP32 input/gradient arithmetic. Pure
1:1 cosine target-prototype pull and source-prototype push. Forty-eight disjoint
references/class; per-layer unit-normalized class mean; valid spatial-token mean
excluding CLS/padding. Four native taps/proxy, layer losses averaged before
per-image per-proxy L1 gradient normalization and equal proxy fusion. No ten-class
margin, patch loss, language-head scoring or proxy-success eligibility gate.

Targets: Qwen3.5-27B, InternVL3.5-14B-HF, Gemma4-31B-it. Corresponding proxy
families exclude the whole target MLLM family. Their actual layer IDs and model
revisions are in the preserved method lock and model audits, not guessed from
normalized depth labels. GPT50 **was used for method selection** by user request;
it must not be described as an untouched commercial holdout. Final150 was frozen
separately; do not tune baselines on its target outputs.

Directions: goldfish→monarch butterfly; pineapple→acoustic guitar;
laptop→espresso; volcano→rocking chair; soccer ball→school bus.
The exact canonical ten-option prompt is imported read-only from
`experiments/2026-09-a100-final-v5/src/common.py::classification_prompt` with
V6 `config/runner_template.yaml`. Evaluation reuses `TransformersTargetGenerator`
and `parse_exact_label` without changing label numbering, generation or parser.

Primary metric: **unconditional targeted hits / all150 GT images**, also reported
per direction /30. Clean-correct conditional ASR is secondary; zero conditional
denominator is N/A. Historical proxy-conditioned TASR is a different estimand.
Failed/missing evaluations never become zero ASR or sample exclusions.

Baseline references: one deterministic target image (minimum image ID in each
existing disjoint 48-image class bank), identical across methods and budgets.
This is a disclosed adaptation of image-pair methods to a class-target benchmark;
it is neither a class prototype nor a target-output-selected reference.
