import torch

def parameters(path):
    args={}
    args['path'] = path # Base path 

    if 'mama_exp' in path:
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
        args['CNN_Batch_Size'] = 25 # Number of images per iteration
        args['CNN_Weight_Decay'] = 1e-4 # Weight Decay
        args['CNN_Out_Dimensions'] = 128 # Feature dimensions of embedding vectors
        args['CNN_Learning_Rate'] = 0.0003 # Learning Rate
        args['CNN_Temperature'] = 0.07 # Softmax temperature         
        
        # GNN setup
        args['GNN_Out_Dimensions'] = 64 # Feature dimensions of spatial-aware embedding vectors
        args['GNN_Dropout'] = 0.3 # Dropout rate applied during GNN training
        args['GNN_Learning_Rate'] = 0.0001 # Learning rate for GNN 
        args['GNN_Weight_Decay'] = 1e-4 # Weight Decay for GNN optimizer
        args['GNN_N_Neighbors'] = [10, 2] # Number of neighbors to sample per GNN layer
        args['GNN_N_Epochs'] = 100 # Total number of training epochs
        args['GNN_Gamma'] = 0.3 # Learning rate decay factor
        args['GNN_Batch_Size'] = 256 # Batch size used during GNN training
        args['GNN_Step_Size'] = 10 # Number of epochs between learning rate decay
        args['GNN_Temperature'] = 0.3 # Temperature parameter for contrastive loss

        # Clustering parameters
        args['Leiden_Percentage'] = 0.3 # Percentage of cells used for Leiden clustering
        args['Leiden_Cell_Type_Resolution'] = 0.2 # Resolution parameter for Leiden clustering to detect cell types
        args['Leiden_Neighborhood_Resolution'] = 0.5 # Resolution parameter for Leiden clustering to detect neighborhoods


    if 'CRC' in path:
        args['GPU_INDEX'] = torch.device('cuda:0') # GPU index
        args['Z_Score'] = True # Whether to apply z score per channel
    
        args['num_DAPI_Ch'] = 0 # DAPI (nuclear) channel number
        args['cyto_channels'] = [4, 14, 20, 25, 50] 

        # Paramaters for single-cell embeddings generation
        args['N_Workers'] = 2 # Number of data loading workers 
        args['N_Crops_per_Image'] = 25 # Number of cells chosen per image
        args['Crop_Size'] = 25 # Size of single-cell crops in pixels
        args['CNN_Epochs'] = 100 # Number of total epoch to run
        args['CNN_Architecture'] = 'resnet18' # or resnet18
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
        args['Leiden_Neighborhood_Resolution'] = 0.5 # Resolution parameter for Leiden clustering to detect neighborhoods


    
    if 'Mama' in path:
        args['GPU_INDEX'] = torch.device('cuda:1') # GPU index
        args['Z_Score'] = True # Whether to apply z score per channel
    
        args['num_DAPI_Ch'] = 0 # DAPI (nuclear) channel number
        args['cyto_channels'] = [15,5,17] 

        # Paramaters for single-cell embeddings generation
        args['N_Workers'] = 2 # Number of data loading workers 
        args['N_Crops_per_Image'] = 25 # Number of cells chosen per image
        args['Crop_Size'] = 20 # Size of single-cell crops in pixels
        args['CNN_Epochs'] = 100 # Number of total epoch to run
        args['CNN_Architecture'] = 'resnet18' # or resnet18
        args['CNN_Batch_Size'] = 25 # Number of images per iteration
        args['CNN_Weight_Decay'] = 1e-4 # Weight Decay
        args['CNN_Out_Dimensions'] = 128 # Feature dimensions of embedding vectors
        args['CNN_Learning_Rate'] = 0.0003 # Learning Rate
        args['CNN_Temperature'] = 0.07 # Softmax temperature         
        
        # GNN setup
        args['GNN_Out_Dimensions'] = 128 # Feature dimensions of spatial-aware embedding vectors
        args['GNN_Dropout'] = 0.3 # Dropout rate applied during GNN training
        args['GNN_Learning_Rate'] = 0.0001 # Learning rate for GNN 
        args['GNN_Weight_Decay'] = 1e-4 # Weight Decay for GNN optimizer
        args['GNN_N_Neighbors'] = [10, 2] # Number of neighbors to sample per GNN layer
        args['GNN_N_Epochs'] = 20 # Total number of training epochs
        args['GNN_Gamma'] = 0.07 # Learning rate decay factor
        args['GNN_Batch_Size'] = 32 # Batch size used during GNN training
        args['GNN_Step_Size'] = 10 # Number of epochs between learning rate decay
        args['GNN_Temperature'] = 0.07 # Temperature parameter for contrastive loss

        # Clustering parameters
        args['Leiden_Percentage'] = 0.2  # Percentage of cells used for Leiden clustering
        args['Leiden_Cell_Type_Resolution'] = 2.5 # Resolution parameter for Leiden clustering to detect cell types
        args['Leiden_Neighborhood_Resolution'] = 0.5 # Resolution parameter for Leiden clustering to detect neighborhoods


    if 'exp7_7plex' in path:
        args['num_DAPI_Ch'] = 6 # DAPI (nuclear) channel number
        args['cyto_channels'] = [0,1,2,3,4,5] 
        args['PCL_CNN_Architecture']='resnet50' # or resnet18
        args['PCL_N_Workers']=2 # Number of data loading workers 
        args['PCL_Epochs']=200 # Number of total epoch to run
        args['PCL_N_Crops_per_Image']=50# Number of crops chosen per image
        args['PCL_Patch_Size']=15 # Size of patches in pixels
        args['PCL_Stride']=0 # Size of stride between consecutive patches
        args['PCL_Alpha_L']=1.15 # size ratio between image crop and image patch
        args['PCL_Z_Score']=True # Whether to apply z score per channel/marker
        args['PCL_Batch_Size']= 60 # Number of images per iteration
        args['PCL_Weight_Decay']= 1e-4 # Weight Decay
        args['PCL_Out_Dimensions']= 128 # Feature dimensions of patch
        args['PCL_Learning_Rate']= 0.0003 # Learning Rate
        args['PCL_Temperature']= 0.07 # Softmax temperature    
        args['PCL_eliminate_Black_Background'] = True 
        args['PCL_eliminate_White_Background'] = False        
        args['PCL_Color_perturbation_Augmentation'] = False # Whether to use color normalization augmentation in H&E white-field microscopy images. Used to force PCL to learn morphological features and avoid learning color staining features. 
        args['PCL_GPU_INDEX']= torch.device('cuda:1')

    if 'LuO_vs_SC' in path:
        args['GPU_INDEX'] = torch.device('cuda:0') # GPU index
        args['Z_Score'] = True # Whether to apply z score per channel

        # Cellular segmentation parameters
        args['num_DAPI_Ch'] = 0 # DAPI (nuclear) channel number
        args['cyto_channels'] = [2,4,6] # Channels for membrane delineation

        # Paramaters for single-cell embeddings generation
        args['N_Workers'] = 10 # Number of data loading workers 
        args['N_Crops_per_Image'] = 50 # Number of cells chosen per image
        args['Crop_Size'] = 15 # Size of single-cell crops in pixels
        args['CNN_Epochs'] = 150 # Number of total epoch to run
        args['CNN_Architecture'] = 'resnet18' # or resnet18
        args['CNN_Batch_Size'] = 40 # Number of images per iteration
        args['CNN_Weight_Decay'] = 1e-4 # Weight Decay
        args['CNN_Out_Dimensions'] = 128 # Feature dimensions of embedding vectors
        args['CNN_Learning_Rate'] = 0.0003 # Learning Rate
        args['CNN_Temperature'] = 0.07 # Softmax temperature         

        # Clustering parameters
        args['Leiden_Percentage'] = 0.3  # Percentage of cells used for Leiden clustering
        args['Leiden_Cell_Type_Resolution'] = 0.3 # Resolution parameter for Leiden clustering to detect cell types

    if 'exp_github_7plex' in path:
        args['GPU_INDEX'] = torch.device('cuda:1') # GPU index
        args['Z_Score'] = True # Whether to apply z score per channel

        # Cellular segmentation parameters
        args['num_DAPI_Ch'] = 6 # DAPI (nuclear) channel number
        args['cyto_channels'] = [2,3,4,5] # Channels for membrane delineation

        # Paramaters for single-cell embeddings generation
        args['N_Workers'] = 2 # Number of data loading workers 
        args['N_Crops_per_Image'] = 10 # Number of cells chosen per image
        args['Crop_Size'] = 15 # Size of single-cell crops in pixels
        args['CNN_Epochs'] = 30 # Number of total epoch to run
        args['CNN_Architecture'] = 'resnet18' # or resnet18
        args['CNN_Batch_Size'] = 20 # Number of images per iteration
        args['CNN_Weight_Decay'] = 1e-4 # Weight Decay
        args['CNN_Out_Dimensions'] = 128 # Feature dimensions of embedding vectors
        args['CNN_Learning_Rate'] = 0.0003 # Learning Rate
        args['CNN_Temperature'] = 0.07 # Softmax temperature         
        
        # GNN setup
        args['GNN_Out_Dimensions'] = 128 # Feature dimensions of spatial-aware embedding vectors
        args['GNN_Dropout'] = 0.3 # Dropout rate applied during GNN training
        args['GNN_Learning_Rate'] = 0.0001 # Learning rate for GNN 
        args['GNN_Weight_Decay'] = 1e-4 # Weight Decay for GNN optimizer
        args['GNN_N_Neighbors'] = [10, 2] # Number of neighbors to sample per GNN layer
        args['GNN_N_Epochs'] = 2 # Total number of training epochs
        args['GNN_Gamma'] = 0.07 # Learning rate decay factor
        args['GNN_Batch_Size'] = 32 # Batch size used during GNN training
        args['GNN_Step_Size'] = 10 # Number of epochs between learning rate decay
        args['GNN_Temperature'] = 0.07 # Temperature parameter for contrastive loss

        # Clustering parameters
        args['Leiden_Percentage'] = 0.7  # Percentage of cells used for Leiden clustering
        args['Leiden_Cell_Type_Resolution'] = 0.3 # Resolution parameter for Leiden clustering to detect cell types
        args['Leiden_Neighborhood_Resolution'] = 0.5 # Resolution parameter for Leiden clustering to detect neighborhoods

    return args
