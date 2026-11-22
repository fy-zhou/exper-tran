# Dataset tasks and targeted-goal design: literature review and proposal

Status: literature-grounded design discussion, **not a new implemented experiment**. Existing outputs and frozen targets remain unchanged. The completed native-task campaign is exploratory; do not relabel its results under a newly favorable target after inspecting outputs. No new GPU attack, judge, commercial call or dataset rewrite was launched for this review.

## The three tasks in plain language

|Data/task|Input|Normal output|What a targeted goal must specify|
|---|---|---|---|
|Current ImageNet diverse10 classification|One image, fixed ten-class rubric|A class label|One predetermined wrong class|
|MS COCO captioning|One image and a description instruction|A natural-language description|A predetermined target description, or an explicitly defined object/attribute replacement|
|LLaVA-Bench(COCO)|One image and one original question|An answer to that particular question|A predetermined wrong answer or semantic condition compatible with the question|

MS COCO is a broad annotated image dataset used for several tasks; this campaign uses image captioning. LLaVA-Bench(COCO) is a small question-answer benchmark built from COCO images, not the LLaVA model itself: the evaluated model may still be Qwen, InternVL or Gemma. Its original30 images have90 conversation/detail/reasoning questions. These give different tasks, not two independent visual domains. Native caption evaluation uses reference-description metrics; original LLaVA evaluation uses GPT4 answer-quality scoring. Neither automatically supplies our targeted-attack goal. Sources: [COCO evaluation implementation](https://github.com/tylin/coco-caption), [LLaVA original paper §5.1](https://arxiv.org/html/2304.08485v2).

## What the inspected primary papers actually do

### AttackVLM — predefined target responses

Section3.1 explicitly separates wrong output from matching a predefined target response. Section3.2 uses either target-text embeddings or a generated target image representing that text. Its quantitative setup uses a broad image-content prompt; qualitative VQA examples do not establish a complete question-compatible target annotation for the90-question LLaVA benchmark. [Primary paper §§3.1–3.2, Table2](https://arxiv.org/html/2305.16934v2).

### FOA-Attack — target-image caption alignment

Section4.1 randomly selects COCO target images and uses a generic description prompt. The same evaluated MLLM describes adversarial and target images; GPTScore compares those descriptions, with success above0.5. This is a well-defined caption-transfer setup even when source and target scenes are unrelated. Its reported ASRs cannot be compared directly with our image-panel boolean judge or50-step task trials. It does not by itself define target answers for arbitrary original VQA questions. [Primary paper §4.1](https://arxiv.org/html/2505.21494v1).

### Chain of Attack — separate target success and fooled-only output

The inspected arXiv-v1 formulation specifies a target reference caption. Its evaluation distinguishes matching that target from merely becoming unrelated to the clean response. That distinction is central to the earlier error-rate/TASR confusion in our campaign. The inspected version and code should be cross-checked against the final conference release before a faithful reproduction claim. [Primary paper §§3.1,3.3](https://arxiv.org/html/2411.15720v1).

### CrossVLAD / CRAFT — one object identity across tasks

The ICCV2025 work defines source→target object-category replacements, and evaluates captioning, detection, region categorization and localization. Caption success requires the target category and exclusion of the source category. Dataset construction controls object ambiguity and existing target-category presence. It **does not include open VQA**. Its object-change goal is closely relevant to our semantic-consensus story; cross-task semantic manipulation itself is established prior work. Borrowing target definitions is distinct from reproducing its region/text-alignment attack or matching its white-box/transfer threat model. [Primary paper §§3.1–3.3](https://arxiv.org/html/2507.07709v1), [official code](https://github.com/Gwill-Z/CRAFT).

## Proposed goals for our method — not measured results

A common semantic change can yield different task outputs. Illustrative source→target: cat→dog.

|Task|Example original output|Proposed targeted output/condition|
|---|---|---|
|Classification|cat|dog|
|Captioning|A cat sits on a sofa.|A dog sits on a sofa; the source cat is not asserted as still present.|
|Object-related QA, original question asks which animal is on the sofa|cat|dog, allowing declared equivalent wording|
|Unrelated QA, original question asks the sofa color|blue|Preserve blue as a non-target utility check; do not force the answer dog.|

Examples explain the principle; they are not new official LLaVA questions or results. Preserve actual benchmark questions. Define the targeted entity/attribute/answer slot from annotations and question semantics before looking at victim results. Use deterministic aliases/semantic parsing for simple object, color or count answers, with negation and object binding handled explicitly. Open output does not inherently require a model judge. Long descriptions/reasoning may require frozen semantic constraints and blinded human/judge validation.

For captioning, two legitimate goals must stay separate: (A) match a full target description/image, as in caption-transfer work; (B) replace one specified visual concept while leaving other scene semantics unconstrained or preserved. A cow being introduced is not the same success condition as faithfully reproducing a whole cow-leading scene. Do not select the looser goal after seeing which outputs score well.

For VQA, freeze the original question and the intended wrong answer set first. Then choose visual references that support that question-conditioned target. If all90 original questions must receive a TASR, all90 need reviewed target specifications. Alternatively, predeclare a targetable subset using question/annotation properties only, report its coverage and TASR, and retain full90-question native quality results. No success-based exclusions or hidden denominator changes. A completely fixed unrelated output string is another research objective (output control); it does not test the same visual-semantic mechanism.

## Implications for reference prototypes and fairness

Keep the attack loss visual-only if that is the research contract. A textual label used to specify or evaluate the desired concept does not require an output-head/text loss. Restore a justified reference construction:48 independently selected images expressing the same target concept, disjoint from attack images, rather than assuming any48 global nearest scenes form a coherent class. Attribute/count/causal-answer prototypes need separate validation; an object prototype does not automatically control every answer type.

Use identical frozen goal mappings and scoring for all baselines. Report supervision, reference count, object boxes, proxy identity and compute differences; a region-aware method should not receive undisclosed extra ground truth. Keep faithful published metrics separate from our adaptations. Use independent development/confirmation data for new target/protocol decisions; current evaluated images are not untouched holdouts. Targeted success is harder to engineer than a class label for rich answers, but is not intrinsically undefinable.

## Correction to the earlier diagnosis

Randomly choosing a target image is not inherently invalid for captioning. The concrete flaw was applying caption-style arbitrary image pairing to question-conditioned QA without specifying compatible target answers, together with an unvalidated class-bank→neighborhood-bank adaptation. Existing low ASR values mix genuine target misses, partially shifted semantics rejected by the chosen full-scene criterion, question-target incompatibility and possible judge error. Their contributions have not been quantified. They should not be used alone to conclude that the original ImageNet method lacks directed transfer.
