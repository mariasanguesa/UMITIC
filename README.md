# Unsupervised Hierarchical Tissue Characterization in Multiplex and Hyperplexed Imaging
***Summary***: We present a fully unsupervised framework for the characterization of multiplex immunofluorescence images at single-cell resolution. Our approach begins with the integration of two advanced deep learning-based cell segmentation strategies, followed by a post-processing step to refine segmentation accuracy. We then employ a contrastive learning scheme to extract low-dimensional, label-free representations of individual cells. These embeddings are enriched with morphological descriptors and spatial features using a custom-designed Graph Neural Network that explicitly models cellular interactions. Building on these representations, we implement a hierarchical analysis pipeline: (i) classification of cell types based on phenotypic features, and (ii) neighborhood-level characterization incorporating spatial context. This multi-level strategy enables a detailed and interpretable understanding of both cellular identities and tissue architecture. By linking cellular phenotypes to their spatial environments, our framework provides a powerful tool for unbiased tissue profiling in digital. The pipeline is fully annotation-free and demonstrates scalability across datasets containing between 7 and 43 markers. To our knowledge, this is the first fully unsupervised method capable of analyzing hyperplex immunofluorescence data at this scale. 
<img src='https://github.com/mariasanguesa/UMITIC/images/Method_Overview.jpg' />

## Index (the usage of this code is explained step by step) 
[Requirements and installation](#Requirements-and-installation) • [Parameter configuration](#Parameter-configuration) • [Running](#Running) • [Cite](#Citation) 

## Requirements and installation

## Citation
Please cite this paper in case our method or parts of it were helpful in your work.
```diff
@article{x,
  title={x},
  author={x},
  journal={x},
  year={x}
}
```
