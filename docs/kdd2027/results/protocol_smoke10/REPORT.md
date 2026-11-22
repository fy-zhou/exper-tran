# CrossVLAD5x2 and original LLaVA10 smoke

Exploratory smoke, not a final benchmark. CrossVLAD uses official labels and caption criterion.
Natural Describe-this-image prompt and224canvas are disclosed chat-MLLM/input adaptations.
Five directions x2 identical sources; all3 large target families; Ours uses48GT references/class.
M/FOA use one fixed exemplar from the same target class bank; SSA uses all80COCO class text prompts.
These have different reference/text/compute resources; do not describe this as fully matched.
Ours50steps; native-budget M/FOA300; SSA-CWA10outer steps/20spectrum samples.
CrossVLAD official substring criterion retained, including its linguistic limitations.
LLaVA10images/30original questions: historical COCO eval_gpt_review.py / rule.json / GPT4 reference.
Original LLaVA Ours frozen outputs reused after receipt/source/bank checks.
Official GPT4 access BLOCKED; no open judge or newer CLI is substituted. No ASR for LLaVA.

|Task|Method|Target|Steps|Returned|Hits/N or official score|Status|
|---|---|---|---:|---:|---|---|
|crossvlad_captioning|ours|InternVL3.5-14B-HF|50|0/10|N/A|PARTIAL|
|crossvlad_captioning|ours|Qwen3.5-27B|50|0/10|N/A|PARTIAL|
|llava_bench_coco|ours|Gemma 4 31B-it|50|30/30|N/A|BLOCKED_OFFICIAL_GPT4_ACCESS|
|llava_bench_coco|ours|InternVL3.5-14B-HF|50|30/30|N/A|BLOCKED_OFFICIAL_GPT4_ACCESS|
|llava_bench_coco|ours|Qwen3.5-27B|50|30/30|N/A|BLOCKED_OFFICIAL_GPT4_ACCESS|
