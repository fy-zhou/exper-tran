# CrossVLAD 5 × 17

CrossVLAD uses official labels and caption criterion.
Natural Describe-this-image prompt and224canvas are disclosed chat-MLLM/input adaptations.
Five directions x17 identical sources; all3 large target families; Ours uses48GT references/class.
M/FOA use one fixed exemplar from the same target class bank; SSA uses all80COCO class text prompts.
These have different reference/text/compute resources; do not describe this as fully matched.
Attack budgets: Ours 50 updates; M-Attack 300; FOA 300; SSA-CWA 10 outer updates with 20 spectrum samples per update.
FOA uses the 300-step setting to avoid truncating its optimization at 50 steps. CrossVLAD has no matched FOA 50-step row, so this comparison does not estimate the effect of changing its step count.
The table compares performance under each method's stated budget. Step counts are not compute-equivalent; see ATTACK_BUDGETS.md.
CrossVLAD official substring criterion retained, including its linguistic limitations.

|Task|Method|Target|Steps|Returned|Hits/N or official score|Status|
|---|---|---|---:|---:|---|---|
|crossvlad_captioning|foa_attack|Qwen3.5-27B|300|85/85|47/85 (55.3%)|RAW_COMPLETE|
|crossvlad_captioning|foa_attack|InternVL3.5-14B-HF|300|85/85|45/85 (52.9%)|RAW_COMPLETE|
|crossvlad_captioning|foa_attack|Gemma 4 31B-it|300|85/85|38/85 (44.7%)|RAW_COMPLETE|
|crossvlad_captioning|m_attack|Qwen3.5-27B|300|85/85|41/85 (48.2%)|RAW_COMPLETE|
|crossvlad_captioning|m_attack|InternVL3.5-14B-HF|300|85/85|43/85 (50.6%)|RAW_COMPLETE|
|crossvlad_captioning|m_attack|Gemma 4 31B-it|300|85/85|35/85 (41.2%)|RAW_COMPLETE|
|crossvlad_captioning|ours|Gemma 4 31B-it|50|85/85|57/85 (67.1%)|RAW_COMPLETE|
|crossvlad_captioning|ours|InternVL3.5-14B-HF|50|85/85|55/85 (64.7%)|RAW_COMPLETE|
|crossvlad_captioning|ours|Qwen3.5-27B|50|85/85|53/85 (62.4%)|RAW_COMPLETE|
|crossvlad_captioning|ssa_cwa_clip|Qwen3.5-27B|10|85/85|54/85 (63.5%)|RAW_COMPLETE|
|crossvlad_captioning|ssa_cwa_clip|InternVL3.5-14B-HF|10|85/85|40/85 (47.1%)|RAW_COMPLETE|
|crossvlad_captioning|ssa_cwa_clip|Gemma 4 31B-it|10|85/85|25/85 (29.4%)|RAW_COMPLETE|
