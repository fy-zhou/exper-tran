# Ordered spatial pooling smoke

Hypothesis: pooling all vision tokens into one vector and taking a 48-image exemplar neighborhood smooths details needed by the target question. Test these two changes separately on the already frozen target-anchored six-question cohort. The original global-only result is 0/6 and remains frozen.

- Global exemplar control retains the original 48-image global prototypes and adds one 25% auxiliary term matching the actual paired source/target images in the same global feature space.
- Ordered quadrants substitutes region-to-corresponding-region cosine similarities in the same auxiliary term, keeping the weight, forward passes, original global prototypes, proxies, selected blocks, seed, DI, BF16, and 50 steps fixed. Qwen's merge-block order is inverted before splitting into quadrants.
- The point of the contrast is the spatial index, rather than adding another optimizer, trained component, gradient weighting scheme, or target-model feedback. Neither branch explicitly represents colors or scene-graph relations. Quadrants are only a coarse layout and shifted objects or source-side augmentation can limit it.

Use the official target question and answer and a blinded GPT-6 Luna medium binary judgment. Report per-question flips with clean and true-target controls. If ordered pooling improves on the global exemplar control, repeat with shuffled region order and a separate color/location/relation cohort before any claims about mechanisms or generalization. These six questions cannot establish a location or relation result on their own.

Related work: FOA uses clustered local tokens and optimal transport (https://arxiv.org/abs/2505.21494); IAU-FOA extends alignment with adaptive marginals and augmentation (https://arxiv.org/abs/2610.06977); Everywhere Attack combines global and local views (https://ojs.aaai.org/index.php/AAAI/article/view/33061); Fine-Grained Detail Targeting studies the information and instability of crops (https://arxiv.org/abs/2602.17645). Ordered grids draw on spatial pyramid matching (https://inc.ucsd.edu/~marni/Igert/Lazebnik_06.pdf), so adding local features or using a grid is not itself a novel contribution.
