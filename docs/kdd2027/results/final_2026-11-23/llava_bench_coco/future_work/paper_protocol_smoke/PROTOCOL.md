# Target-anchored LLaVA-Bench (COCO) pilot

Inspired by section 5.2 and Table 6 of https://arxiv.org/pdf/2505.01050#page=14.

Sample two of the 30 official COCO target images with seed 42. For each of their three original questions, select a different official image as the source input by seeded random draw. Use the target image's official question and reference answer unchanged.

Compare three conditions using the same question, reference answer, and judge. Clean is the different unperturbed source image. Adv is that source's newly generated Uniform4 perturbation directed toward the paired target. True target is the unperturbed target image, a positive control. The perturbations are frozen and validated before target inference. The existing 48-image neighborhood of each actual image supplies its prototype. Use BF16 Gemma 31B as the experimental target, with target-family exclusion in the proxy ensemble.

GPT-6 Luna medium returns a binary judgment of whether the candidate at least partially addresses the question relative to the target's official reference answer. Each mode has six planned judgments, two per question type. Candidate method and condition are hidden. The paper uses a different attack, target-model set, and GPT-4o judge. This small pilot checks the target-anchored evaluation and does not support a comparison of attack efficacy with Table 6. Its images and questions need not overlap the previous six-image rubric-only smoke, so those results are not a paired effect of changing the question source.
