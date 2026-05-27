import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from Code.Cellular_Segmentation.Cellular_Segmentation import cellular_segmentation
from Code.utils.DatasetParameters import parameters

import gc
import numpy as np
gc.set_threshold(0, 0, 0)

def main(path):

    # Select Experiment parameters
    params = parameters(path) 
    
    # # Nuclear and cellular segmentation
    # cellular_segmentation(params)
    # gc.collect()

    # Run contrastive learning to transform single-cell crops into low-dimensional embeddings
    # from Code.Patch_Contrastive_Learning.Patch_Contrastive_Learning import patch_contrastive_learning
    # patch_contrastive_learning(params) 
    # gc.collect()

    # Run GNN for spatial-aware embedding generation
    # from Code.Spatial_Contrastive_Learning.Spatial_Contrastive_Learning import spatial_contrastive_learning
    # spatial_contrastive_learning(params)
    # gc.collect()
    
    # # # Assign a cell type to each cell
    # from Code.Clustering.Cell_Type_Assignment import cell_type_assignment
    # for i in np.arange(1.5, 3, 0.1):
    #     params['Leiden_Cell_Type_Resolution'] = float(f"{i:.1f}")
    #     cell_type_assignment(params)

    # Assign a neighborhood to each cell
    from Code.Clustering.Neighborhood_Assignment import neighborhood_assignment
    for j in np.arange(0.1, 1, 0.1):
        params['Leiden_Neighborhood_Resolution'] = float(f"{j:.1f}")  
        neighborhood_assignment(params)

if __name__ == "__main__":    
    path ="Z:/sanguesa.127969/CRC_2/"
    # path = "D:/msanguesa/STEP_Mama_HER2_good_recurrence_filtered/" 
    main(path) 