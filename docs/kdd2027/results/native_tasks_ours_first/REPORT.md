# Ours first: native COCO captioning and LLaVA open QA

Ours uses a declared label-free48-reference-neighborhood adapter. It is not the original ImageNet class-bank protocol.
All model parameters BF16, target-family-disjoint proxies,50 steps, no target feedback in generation.
Positive quality drop means degradation. No classification ASR is inferred. Missing/invalid scores are not zero.
LLaVA open-weight cross-family paired judge is adapted, blinded and explicitly not official GPT4 scoring.
CIDEr depends on the evaluated reference population; compare methods only at matched N/IDs.

|Dataset|Target|Images|Returned/planned|Primary metric valid|Clean observed|Adv observed|
|---|---|---:|---:|---|---:|---:|
|coco_captioning|Gemma 4 31B-it|10|10/10|True|0.5160691839681533|0.06053816207462239|
|coco_captioning|Gemma 4 31B-it|150|150/150|True|0.5360942863067093|0.05397750606021918|
|coco_captioning|InternVL3.5-14B-HF|10|10/10|True|1.2996178725239569|0.28201423937863057|
|coco_captioning|InternVL3.5-14B-HF|150|150/150|True|1.4366307970694399|0.12806283143736225|
|coco_captioning|Qwen3.5-27B|10|10/10|True|0.3449329034839505|0.11319803756204613|
|coco_captioning|Qwen3.5-27B|150|150/150|True|0.6124385119892448|0.07181501479840409|
|llava_bench_coco|Gemma 4 31B-it|30|90/90|True|9.355555555555556|0.5111111111111111|
|llava_bench_coco|Gemma 4 31B-it|3|9/9|True|9.555555555555555|0.0|
|llava_bench_coco|InternVL3.5-14B-HF|30|90/90|True|8.7|1.9|
|llava_bench_coco|InternVL3.5-14B-HF|3|9/9|True|8.222222222222221|0.6666666666666666|
|llava_bench_coco|Qwen3.5-27B|30|90/90|True|9.533333333333333|0.18888888888888888|
|llava_bench_coco|Qwen3.5-27B|3|9/9|True|9.666666666666666|0.0|
