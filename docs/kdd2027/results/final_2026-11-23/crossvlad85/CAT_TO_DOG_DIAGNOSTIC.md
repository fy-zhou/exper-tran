# CrossVLAD cat to dog caption diagnostic

The frozen 17-image direction uses the same source IDs across targets. The official lowercase substring score remains unchanged. The following word-boundary count is a post-hoc diagnostic using cat/cats/kitten/feline/tabby and dog/dogs/puppy/canine and selected common breed names.

| Target | Official hits | Literal cat only | Literal both | Literal neither | Word-boundary diagnostic hits |
|---|---:|---:|---:|---:|---:|
| Qwen 27B | 9/17 | 2 | 0 | 6 | 9/17 |
| InternVL 14B | 3/17 | 7 | 2 | 5 | 5/17 |
| Gemma 31B | 11/17 | 2 | 1 | 3 | 12/17 |

InternVL's 314140 and 234124 contain dog but fail because "indicating" contains "cat". Gemma's 311965 has the same problem with "scattered". The diagnostic parser change therefore removes two InternVL false exclusions and one Gemma false exclusion. InternVL remains at 5/17, with seven captions retaining only the source keyword and five showing neither keyword. Caption degradation and targeted object replacement are distinct outcomes. This analysis does not establish whether the remaining difference comes from the target model, the family-disjoint proxy set, or the reference distribution.
