With the availability of large-scale datasets and sustained advances in computational resources,
deep learning has achieved substantial progress in computer vision. Nevertheless,
in data-scarce scenarios, model performance and generalization ability remain severely limited.
Existing few-shot learning approaches that rely on image data augmentation can partially mitigate sample insufficiency;
however, they often fail to fully exploit the intrinsic information contained in limited training data, resulting in restricted improvements in data diversity and representational richness.
To address these challenges, this study proposes a novel image data augmentation framework based on a three-dimensional mapping table.
The framework first utilizes a visual model pre-trained on the original dataset.
Subsequently, a co-evolutionary algorithm is employed to optimize the Three-Dimensional lookup table at the pixel level, thereby substantially expanding the diversity of generated samples.
In addition, a pruning strategy is introduced to eliminate redundant elements within the mapping table that contribute marginally to the mapping outcomes, effectively reducing encoding length and computational complexity.
The resulting optimized Three-Dimensional lookup table exhibits strong generalization capability and can effectively enhance unseen data.
Importantly, the proposed design ensures that the augmented samples preserve semantic consistency while presenting more diverse and informative feature distributions, thereby alleviating the adverse effects of data scarcity.
Extensive experiments conducted on multiple benchmark datasets demonstrate that the proposed framework consistently outperforms existing image data augmentation methods across a range of evaluation metrics, confirming its effectiveness and superiority.