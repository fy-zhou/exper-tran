# LLaVA target-anchored smoke: global exemplar vs spatial alignment

Gemma 31B, two frozen target images, six official target questions, unchanged image pairs and target references. GPT-6 Luna medium uses the same paper-style partial-answer criterion for all conditions.

| Method | Clean | Adv | True target | Median generation seconds per adversarial image |
|---|---:|---:|---:|---:|
| Original Uniform4, neighborhood prototypes | 0/6 | 0/6 | 5/6 | 11.0 |
| Uniform4 + global exact exemplar, 25% | 0/6 | 1/6 | 5/6 | 10.5 |
| Uniform4 + ordered 2×2 region pooling, 25% | 0/6 | 0/6 | 5/6 | 12.6 |

Both candidate conditions generated and validated all 6 PNGs with the 16/255 pixel bound. Each method uses the same forward count, four layers, model pool, random seed, DI and 50-step budget. Spatial region pooling takes the matching quadrant from the exact source and target and preserves the 48-image global prototypes. The global exemplar control matches the exact same images and uses the same loss weight. BF16 audits, pinned sources, PNG hashes, answer traces, and judge records are preserved in each variant directory.

The one global-exemplar match is target question 10. The target shows a large black dog with a red collar lying on a kitchen floor beside two bowls. The candidate described a small reddish-brown and black puppy resting on patterned fabric. The judge counted partial overlap for a resting dog. The mismatch in color, setting and objects means this hit does not demonstrate preservation of those details.

Conclusion for this cohort: a positional 2×2 grid is insufficient. It has no observed match advantage over global prototypes or the exact-exemplar control. Neither candidate establishes fine-grained transfer. The sample has two independent target images and cannot establish a category-wide effect. Next, verify whether the needed details appear in the frozen local features and align only the object or object pair supporting the question before introducing any additional losses or tuning. An object-conditioned experiment would require independently chosen annotations and a matched control.
