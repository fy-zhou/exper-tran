# November21 targeted baseline and task-extension campaign

Status at 2026-11-21T19:54:00Z: {'RAW_COMPLETE_OFFICIAL_JUDGE_PENDING': 9, 'COMPLETE': 18}. Missing results are N/A, never zeros.

ImageNet new methods: AttackVLM MF-ii and MF-it (two variants of one paper), and the official SSA-CWA optimizer adapted to a CLIP ten-class zero-shot classifier. Full150 source cohort and all three fixed large targets. Target output never enters generation.

COCO:10 fixed images, concise captioning, clean/adv CIDEr on this subset and target-caption alignment. LLaVA:3 fixed original benchmark images,9 original open questions; raw clean/adv answers and reference ROUGE-L diagnostic. These are initial task trials, not full benchmark results. ROUGE-L is not official VQA correctness; GPT4 judge remains pending.

|Task|Method|Target|N|Status|ASR|
|---|---|---|---:|---|---:|
|imagenet|attackvlm_mfii|Qwen3.5-27B|150|COMPLETE|0.00%|
|imagenet|attackvlm_mfii|InternVL3.5-14B-HF|150|COMPLETE|1.33%|
|imagenet|attackvlm_mfii|Gemma 4 31B-it|150|COMPLETE|0.00%|
|imagenet|attackvlm_mfit|Qwen3.5-27B|150|COMPLETE|0.00%|
|imagenet|attackvlm_mfit|InternVL3.5-14B-HF|150|COMPLETE|0.67%|
|imagenet|attackvlm_mfit|Gemma 4 31B-it|150|COMPLETE|0.67%|
|imagenet|ssa_cwa_clip|Qwen3.5-27B|150|COMPLETE|62.67%|
|imagenet|ssa_cwa_clip|InternVL3.5-14B-HF|150|COMPLETE|59.33%|
|imagenet|ssa_cwa_clip|Gemma 4 31B-it|150|COMPLETE|34.67%|
|coco_captioning|m_attack|Qwen3.5-27B|10|COMPLETE|N/A|
|coco_captioning|m_attack|InternVL3.5-14B-HF|10|COMPLETE|N/A|
|coco_captioning|m_attack|Gemma 4 31B-it|10|COMPLETE|N/A|
|coco_captioning|foa_attack|Qwen3.5-27B|10|COMPLETE|N/A|
|coco_captioning|foa_attack|InternVL3.5-14B-HF|10|COMPLETE|N/A|
|coco_captioning|foa_attack|Gemma 4 31B-it|10|COMPLETE|N/A|
|coco_captioning|pgd|Qwen3.5-27B|10|COMPLETE|N/A|
|coco_captioning|pgd|InternVL3.5-14B-HF|10|COMPLETE|N/A|
|coco_captioning|pgd|Gemma 4 31B-it|10|COMPLETE|N/A|
|llava_bench_coco|m_attack|Qwen3.5-27B|3|RAW_COMPLETE_OFFICIAL_JUDGE_PENDING|N/A|
|llava_bench_coco|m_attack|InternVL3.5-14B-HF|3|RAW_COMPLETE_OFFICIAL_JUDGE_PENDING|N/A|
|llava_bench_coco|m_attack|Gemma 4 31B-it|3|RAW_COMPLETE_OFFICIAL_JUDGE_PENDING|N/A|
|llava_bench_coco|foa_attack|Qwen3.5-27B|3|RAW_COMPLETE_OFFICIAL_JUDGE_PENDING|N/A|
|llava_bench_coco|foa_attack|InternVL3.5-14B-HF|3|RAW_COMPLETE_OFFICIAL_JUDGE_PENDING|N/A|
|llava_bench_coco|foa_attack|Gemma 4 31B-it|3|RAW_COMPLETE_OFFICIAL_JUDGE_PENDING|N/A|
|llava_bench_coco|pgd|Qwen3.5-27B|3|RAW_COMPLETE_OFFICIAL_JUDGE_PENDING|N/A|
|llava_bench_coco|pgd|InternVL3.5-14B-HF|3|RAW_COMPLETE_OFFICIAL_JUDGE_PENDING|N/A|
|llava_bench_coco|pgd|Gemma 4 31B-it|3|RAW_COMPLETE_OFFICIAL_JUDGE_PENDING|N/A|

## Caption task trial (raw pycocoevalcap CIDEr units, same10-image IDF set)

|Method|Target|Clean CIDEr|Adv CIDEr|
|---|---|---:|---:|
|foa_attack|Gemma 4 31B-it|0.5161|0.2814|
|foa_attack|Qwen3.5-27B|0.3449|0.4175|
|foa_attack|InternVL3.5-14B-HF|1.2996|0.6452|
|m_attack|Gemma 4 31B-it|0.5161|0.2519|
|m_attack|Qwen3.5-27B|0.3449|0.3432|
|m_attack|InternVL3.5-14B-HF|1.2996|0.9535|
|pgd|Gemma 4 31B-it|0.5161|0.4737|
|pgd|Qwen3.5-27B|0.3449|0.5192|
|pgd|InternVL3.5-14B-HF|1.2996|1.1161|

## Subscription CLI targeted classification

Requested gpt-6.1-sol, low effort, same frozen150 images, existing ChatGPT login only; private server snapshot undisclosed. Balanced10 is a transport/sample check from the final cohort, not an independent development set. No attack parameter is selected using it. One unscored180s timeout left a verified empty answer file; its evidence was archived once. 600s transport capture/owned process-group cleanup now apply. Scored results are never retried. Auth/quota failures stop calls; no API/model fallback. A current-date frozen Uniform4 control runs only after the five-baseline queue, sharing clean answers under identical request identities.

|Method|N|Completed pairs|Hits|ASR|
|---|---:|---:|---:|---:|
|anyattack|10|10|0|0.00%|
|anyattack|150|101|1|N/A|
|foa_attack|10|10|6|60.00%|
|foa_attack|150|150|91|60.67%|
|m_attack|10|10|5|50.00%|
|m_attack|150|150|83|55.33%|
|mi_fgsm|10|10|0|0.00%|
|mi_fgsm|150|150|9|6.00%|
|pgd|10|10|0|0.00%|
|pgd|150|150|3|2.00%|

All open model floating parameters are BF16/unquantized; model-load audits, frozen manifests, raw answers, request attempts, source locks and loss traces remain on persistent root `/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction/nov21`. Weights are cached only on /ephemeral. Native step counts and forward/backward budgets differ; source reference counts, text/image guidance and surrogate identities are disclosed in the plan. This is not a fully compute/information-matched ranking.

IAU and V2 stay paused. Old campaigns, Ours attacks, paper and original results are untouched. No LLaVA official score or full-dataset result is invented.
