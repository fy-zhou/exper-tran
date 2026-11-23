# LLaVA-Bench (COCO) quality decline

Official 30 images and 90 questions. The same 90 clean/adv question pairs per cell were rated with GPT-6 Luna medium. All methods in this table use 50 attack steps.

Relative score is 100 × sum of candidate ratings / sum of reference ratings. Point drop is clean minus adversarial relative score. Relative decline is point drop divided by clean relative score, expressed as a percentage. Larger declines indicate greater degradation of the original answer.

| Method | Target | Clean score | Adv score | Drop (points) | Relative decline | Pairs |
|---|---|---:|---:|---:|---:|---:|
| Ours | Qwen 27B | 65.91 | 14.59 | 51.31 | 77.86% | 90 |
| Ours | InternVL 14B | 77.96 | 30.21 | 47.75 | 61.25% | 90 |
| Ours | Gemma 31B | 85.39 | 17.42 | 67.97 | 79.60% | 90 |
| M-Attack | Qwen 27B | 65.91 | 34.37 | 31.54 | 47.86% | 90 |
| M-Attack | InternVL 14B | 77.96 | 41.34 | 36.61 | 46.97% | 90 |
| M-Attack | Gemma 31B | 85.39 | 50.43 | 34.96 | 40.94% | 90 |
| FOA | Qwen 27B | 65.91 | 35.89 | 30.01 | 45.54% | 90 |
| FOA | InternVL 14B | 77.96 | 45.30 | 32.66 | 41.90% | 90 |
| FOA | Gemma 31B | 85.39 | 53.20 | 32.20 | 37.70% | 90 |
| MI-FGSM | Qwen 27B | 65.91 | 52.48 | 13.42 | 20.37% | 90 |
| MI-FGSM | InternVL 14B | 77.96 | 63.61 | 14.35 | 18.41% | 90 |
| MI-FGSM | Gemma 31B | 85.39 | 76.29 | 9.11 | 10.66% | 90 |
| PGD | Qwen 27B | 65.91 | 58.31 | 7.60 | 11.53% | 90 |
| PGD | InternVL 14B | 77.96 | 70.04 | 7.92 | 10.15% | 90 |
| PGD | Gemma 31B | 85.39 | 81.78 | 3.62 | 4.24% | 90 |

See [summary CSV](quality_summary.csv) and [question-level scores](quality_per_question.csv). The Ours-only target-matching judgments are archived as [exploratory results](ours_match_summary.json). The matched target-question smoke and fine-grained color, position, and relation transfer are future work.
