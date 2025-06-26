# Unsupervised Hierarchical Tissue Characterization in Multiplex and Hyperplexed Imaging
***Summary***: here we present a fully unsupervised pipeline designed for high-resolution, multi-level tissue profiling using multiplex or hyperplexed immunofluorescence imaging data. It integrates deep learning techniques for cell segmentation, representation learning, and spatial modeling, without relying on human annotations. 

![method overview](https://github.com/mariasanguesa/UMITIC/blob/main/images/method_overview.jpg)

---

## 🧬 Overview

We introduce a comprehensive unsupervised framework for characterizing tissue architecture at single-cell resolution across multiplex/hyperplex imaging modalities. The pipeline includes:

1. **Cell Segmentation**  
   Dual-model segmentation combining nuclear and cytoplasmatic features with refinement post-processing.

2. **Contrastive Representation Learning**  
   Self-supervised CNNs extract low-dimensional embeddings from single-cell crops without labels. 

3. **Spatial Embedding with GNNs**  
   A custom Graph Neural Network models intercellular relationships, integrating local spatial context.

4. **Hierarchical Analysis**  
   - **Cell-level clustering**: Marker expression and morphology-based classification of cell types.  
   - **Tissue-level analysis**: Spatial neighborhood clustering of the tissue.

This annotation-free, scalable method supports datasets containing 7 to 43+ markers. To our knowledge, it is the first **fully unsupervised** framework capable of processing hyperplex images at this scale.

🧠 Highlights: 

	✅ Fully unsupervised pipeline (no manual labels required)
	
	🧬 Integrates both phenotypic and spatial information
	
	🌍 Scalable to 40+ markers (hyperplex imaging)
	
	📊 Highly interpretable results

---

## Index 
[Requirements](#Requirements) • [Dataset preparation](#Dataset-preparation) • [Parameter configuration](#Parameter-configuration) • [Running](#Running) • [Cite](#Citation) 

## Requirements 
* Tested on Windows 10
* Tested on GPU: Nvidia GeForce RTX 6000. For GPU support, install the versions of CUDA that are compatible with Pytorch's versions.
* Python ≥ 3.8  (tested on 3.8.20)
  
* Requirements installation:
  ```bash
  pip install -r requirements.txt
  
## Dataset preparation
Prepare your dataset folder (DATASET_DIR) as follows:

```bash
DATASET_DIR/
├── Raw_Data/
│   ├── Images/
│   │   ├── image_1.tiff
│   │   ├── image_2.tiff
│   │   └── ...
│   └── Experiment_Information/
│       └── Channels.txt               		
```
In the 'Raw_Data/Images' folder we expect multiplex/hyperplex image data consisting of multi-channel '.tiff' files with one marker per channel.
In the 'Raw_Data/Experiment_Information' Channels.txt should contain the list of marker names. Each line corresponds to one image channel. Use "None" to ignore specific channels.

```bash
DAPI
CD45
None
PanCK  
```

## Parameter configuration
All experiment parameters are defined in 'UMITIC/src/utils/DatasetParameters.py'. Change it to your own configuration: 
```python
def parameters(path, debug):
    if 'DATASET_DIR' in path:        
        args['param1'] = value1
      	args['param2'] = value2
	...		
```
See example [file](https://github.com/mariasanguesa/UMITIC/blob/main/UMITIC/src/utils/DatasetParameters.py) for full configuration.

## Running
After preparing your data and setting parameters:
```bash
python main.py
```

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
