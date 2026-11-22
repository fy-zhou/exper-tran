# ASR (targeted semantic success / TASR)

ASR here means targeted success against the originally fixed target image. Arbitrary wrong answers do not count.
Primary denominator: every fixed test image (COCO150) or original question (LLaVA90). Missing judgments are N/A.
Fixed cross-family open-weight semantic adjudication; post-hoc rule, not official benchmark or human-ground-truth scoring.
The prior non-targeted semantic failure rate is auxiliary and must not be relabeled as these results.

|Task|Method|Steps|Target|Scored/planned|Target hits/N|ASR (targeted)|Status|
|---|---|---:|---|---:|---:|---:|---|
|coco_captioning|ours_uniform4_neighborhood48|50|Gemma 4 31B-it|150/150|5/150|3.33%|COMPLETE|
|coco_captioning|ours_uniform4_neighborhood48|50|InternVL3.5-14B-HF|150/150|23/150|15.33%|COMPLETE|
|coco_captioning|ours_uniform4_neighborhood48|50|Qwen3.5-27B|150/150|15/150|10.00%|COMPLETE|
|llava_bench_coco|ours_uniform4_neighborhood48|50|Gemma 4 31B-it|90/90|14/90|15.56%|COMPLETE|
|llava_bench_coco|ours_uniform4_neighborhood48|50|InternVL3.5-14B-HF|90/90|11/90|12.22%|COMPLETE|
|llava_bench_coco|ours_uniform4_neighborhood48|50|Qwen3.5-27B|90/90|10/90|11.11%|COMPLETE|
