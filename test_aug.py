import torch
from Code.Patch_Contrastive_Learning.data_aug.contrastive_learning_dataset import (
    ContrastiveLearningDataset_PCL,
    visualize_original_plus_augs
)
args={}
args['path'] = "Z:/sanguesa.127969/code/mama_exp/good_recurrence/" 
args['GPU_INDEX'] = torch.device('cuda:0') # GPU index
args['Z_Score'] = True # Whether to apply z score per channel

args['num_DAPI_Ch'] = 0 # DAPI (nuclear) channel number
args['cyto_channels'] = [1,8,12,17] 

# Paramaters for single-cell embeddings generation
args['N_Workers'] = 4 # Number of data loading workers 
args['N_Crops_per_Image'] = 25 # Number of cells chosen per image
args['Crop_Size'] = 25 # Size of single-cell crops in pixels
args['CNN_Epochs'] = 100 # Number of total epoch to run
args['CNN_Architecture'] = 'resnet50' # or resnet18
args['CNN_Batch_Size'] = 40 # Number of images per iteration
args['CNN_Weight_Decay'] = 1e-4 # Weight Decay
args['CNN_Out_Dimensions'] = 128 # Feature dimensions of embedding vectors
args['CNN_Learning_Rate'] = 0.0003 # Learning Rate
args['CNN_Temperature'] = 0.07 # Softmax temperature         

# GNN setup
args['GNN_Out_Dimensions'] = 64 # Feature dimensions of spatial-aware embedding vectors
args['GNN_Dropout'] = 0.3 # Dropout rate applied during GNN training
args['GNN_Learning_Rate'] = 0.0001 # Learning rate for GNN 
args['GNN_Weight_Decay'] = 1e-4 # Weight Decay for GNN optimizer
args['GNN_N_Neighbors'] = [15, 5] # Number of neighbors to sample per GNN layer
args['GNN_N_Epochs'] = 25 # Total number of training epochs
args['GNN_Gamma'] = 0.3 # Learning rate decay factor
args['GNN_Batch_Size'] = 256 # Batch size used during GNN training
args['GNN_Step_Size'] = 10 # Number of epochs between learning rate decay
args['GNN_Temperature'] = 0.3 # Temperature parameter for contrastive loss

# Clustering parameters
args['Leiden_Percentage'] = 0.8  # Percentage of cells used for Leiden clustering
args['Leiden_Cell_Type_Resolution'] = 0.2 # Resolution parameter for Leiden clustering to detect cell types
args['Leiden_Neighborhood_Resolution'] = 0.5 
ds = ContrastiveLearningDataset_PCL(args, training=True)

_ = visualize_original_plus_augs(
    dataset=ds,
    idx=0,
    crop_idx=0,
    rgb_channels=(15, 17, 0),  # elige 3 marcadores representativos
    n_augs=6,
    seed=42
)