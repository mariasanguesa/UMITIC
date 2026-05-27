from Code.Cellular_Segmentation.Cellular_Segmentation import cellular_segmentation
from Code.Clustering.Cell_Type_Assignment import cell_type_assignment
from Code.utils.DatasetParameters import parameters
from Code.Patch_Contrastive_Learning.Patch_Contrastive_Learning import patch_contrastive_learning
from Code.Spatial_Contrastive_Learning.Spatial_Contrastive_Learning import spatial_contrastive_learning
from Code.Clustering.Neighborhood_Assignment import neighborhood_assignment
from Code.Clustering.Cell_Type_Assignment import cell_type_assignment

import gc
import numpy as np
gc.set_threshold(0, 0, 0)

def main(path):

    # Select Experiment parameters
    params = parameters(path) 
    
    # Nuclear and cellular segmentation
    cellular_segmentation(params)
    gc.collect()

    # Run contrastive learning to transform single-cell crops into low-dimensional embeddings
    patch_contrastive_learning(params) 
    gc.collect()

    # Run GNN for spatial-aware embedding generation
    spatial_contrastive_learning(params)
    gc.collect()
    
    #Assign a cell type to each cell
    cell_type_assignment(params)

    # Assign a neighborhood to each cell
    neighborhood_assignment(params)

if __name__ == "__main__":     
    path = "D:/msanguesa/mama_exp/primary_lrf/"     
    main(path) 