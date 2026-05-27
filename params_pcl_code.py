import torch

def parameters(path, debug):

    # args=DefaultParameters(path)
    args={}
    args['path'] = path
    if 'exp7_7plex' in path:
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
    
    if '_conference' in path:
        args['num_DAPI_Ch'] = 0
        args['cyto_channels'] = [1,2,3,4,5]
        args['PCL_CNN_Architecture']='resnet50' # or resnet18
        args['PCL_N_Workers']=2 # Number of data loading workers 
        args['PCL_Epochs']=200 # Number of total epoch to run
        args['PCL_N_Crops_per_Image']=20# Number of crops chosen per image
        args['PCL_Patch_Size']=20 # Size of patches in pixels
        args['PCL_Stride']=0 # Size of stride between consecutive patches
        args['PCL_Alpha_L']=1.15 # size ratio between image crop and image patch
        args['PCL_Z_Score']= True # Whether to apply z score per channel/marker
        args['PCL_Batch_Size']= 50 # Number of images per iteration
        args['PCL_Weight_Decay']= 1e-4 # Weight Decay
        args['PCL_Out_Dimensions']= 64 # Feature dimensions of patch
        args['PCL_Learning_Rate']= 0.0003 # Learning Rate
        args['PCL_Temperature']= 0.07 # Softmax temperature    
        args['PCL_eliminate_Black_Background'] = True 
        args['PCL_eliminate_White_Background'] = False        
        args['PCL_Color_perturbation_Augmentation'] = False # Whether to use color normalization augmentation in H&E white-field microscopy images. Used to force PCL to learn morphological features and avoid learning color staining features. 
        args['PCL_GPU_INDEX']= torch.device('cuda:1')
    
    if 'exp6_7plex' in path:
        args['PCL_CNN_Architecture']='resnet18' # or resnet18
        args['PCL_N_Workers']=8 # Number of data loading workers 
        args['PCL_Epochs']=200 # Number of total epoch to run
        args['PCL_N_Crops_per_Image']=30# Number of crops chosen per image
        args['PCL_Patch_Size']=15 # Size of patches in pixels
        args['PCL_Stride']=0 # Size of stride between consecutive patches
        args['PCL_Alpha_L']=1.15 # size ratio between image crop and image patch
        args['PCL_Z_Score']= True # Whether to apply z score per channel/marker
        args['PCL_Batch_Size']= 50 # Number of images per iteration
        args['PCL_Weight_Decay']= 1e-4 # Weight Decay
        args['PCL_Out_Dimensions']= 128 # Feature dimensions of patch
        args['PCL_Learning_Rate']= 0.0003 # Learning Rate
        args['PCL_Temperature']= 0.07 # Softmax temperature    
        args['PCL_eliminate_Black_Background'] = True 
        args['PCL_eliminate_White_Background'] = False        
        args['PCL_Color_perturbation_Augmentation'] = False # Whether to use color normalization augmentation in H&E white-field microscopy images. Used to force PCL to learn morphological features and avoid learning color staining features. 
        args['PCL_GPU_INDEX']= torch.device('cuda:1')

    if 'exp5_7plex' in path:
        args['PCL_CNN_Architecture']='resnet50' # or resnet18
        args['PCL_N_Workers']=20 # Number of data loading workers 
        args['PCL_Epochs']=300 # Number of total epoch to run
        args['PCL_N_Crops_per_Image']=50# Number of crops chosen per image
        args['PCL_Patch_Size']=15 # Size of patches in pixels
        args['PCL_Stride']=0 # Size of stride between consecutive patches
        args['PCL_Alpha_L']=1.15 # size ratio between image crop and image patch
        args['PCL_Z_Score']= False # Whether to apply z score per channel/marker
        args['PCL_Batch_Size']= 40 # Number of images per iteration
        args['PCL_Weight_Decay']= 1e-4 # Weight Decay
        args['PCL_Out_Dimensions']= 128 # Feature dimensions of patch
        args['PCL_Learning_Rate']= 0.0003 # Learning Rate
        args['PCL_Temperature']= 0.07 # Softmax temperature    
        args['PCL_eliminate_Black_Background'] = True 
        args['PCL_eliminate_White_Background'] = False        
        args['PCL_Color_perturbation_Augmentation'] = False # Whether to use color normalization augmentation in H&E white-field microscopy images. Used to force PCL to learn morphological features and avoid learning color staining features. 
        args['PCL_GPU_INDEX']= torch.device('cuda:1')

    if 'exp4_7plex' in path:
        args['PCL_CNN_Architecture']='resnet50' # or resnet18
        args['PCL_N_Workers']=20 # Number of data loading workers 
        args['PCL_Epochs']=250 # Number of total epoch to run
        args['PCL_N_Crops_per_Image']=10# Number of crops chosen per image
        args['PCL_Patch_Size']=15 # Size of patches in pixels
        args['PCL_Stride']=0 # Size of stride between consecutive patches
        args['PCL_Alpha_L']=1.15 # size ratio between image crop and image patch
        args['PCL_Z_Score']= False # Whether to apply z score per channel/marker
        args['PCL_Batch_Size']= 40 # Number of images per iteration
        args['PCL_Weight_Decay']= 1e-4 # Weight Decay
        args['PCL_Out_Dimensions']= 128 # Feature dimensions of patch
        args['PCL_Learning_Rate']= 0.0003 # Learning Rate
        args['PCL_Temperature']= 0.07 # Softmax temperature    
        args['PCL_eliminate_Black_Background'] = True 
        args['PCL_eliminate_White_Background'] = False        
        args['PCL_Color_perturbation_Augmentation'] = False # Whether to use color normalization augmentation in H&E white-field microscopy images. Used to force PCL to learn morphological features and avoid learning color staining features. 
        args['PCL_GPU_INDEX']= torch.device('cuda:1')

    if 'exp3_7plex' in path:
        args['PCL_CNN_Architecture']='resnet50' # or resnet18
        args['PCL_N_Workers']=20 # Number of data loading workers 
        args['PCL_Epochs']=250 # Number of total epoch to run
        args['PCL_N_Crops_per_Image']=10# Number of crops chosen per image
        args['PCL_Patch_Size']=15 # Size of patches in pixels
        args['PCL_Stride']=0 # Size of stride between consecutive patches
        args['PCL_Alpha_L']=1.15 # size ratio between image crop and image patch
        args['PCL_Z_Score']= False # Whether to apply z score per channel/marker
        args['PCL_Batch_Size']= 40 # Number of images per iteration
        args['PCL_Weight_Decay']= 1e-4 # Weight Decay
        args['PCL_Out_Dimensions']= 128 # Feature dimensions of patch
        args['PCL_Learning_Rate']= 0.0003 # Learning Rate
        args['PCL_Temperature']= 0.07 # Softmax temperature    
        args['PCL_eliminate_Black_Background'] = True 
        args['PCL_eliminate_White_Background'] = False        
        args['PCL_Color_perturbation_Augmentation'] = False # Whether to use color normalization augmentation in H&E white-field microscopy images. Used to force PCL to learn morphological features and avoid learning color staining features. 
        args['PCL_GPU_INDEX']= torch.device('cuda:1')

    if '23B10981_exp1' in path:
        args['PCL_CNN_Architecture']='resnet50' # or resnet18
        args['PCL_N_Workers']=20 # Number of data loading workers 
        args['PCL_Epochs']=200 # Number of total epoch to run
        args['PCL_N_Crops_per_Image']=30# Number of crops chosen per image
        args['PCL_Patch_Size']=15 # Size of patches in pixels
        args['PCL_Stride']=0 # Size of stride between consecutive patches
        args['PCL_Alpha_L']=1.15 # size ratio between image crop and image patch
        args['PCL_Z_Score']= True # Whether to apply z score per channel/marker
        args['PCL_Batch_Size']= 40 # Number of images per iteration
        args['PCL_Weight_Decay']= 1e-4 # Weight Decay
        args['PCL_Out_Dimensions']= 128 # Feature dimensions of patch
        args['PCL_Learning_Rate']= 0.0003 # Learning Rate
        args['PCL_Temperature']= 0.07 # Softmax temperature    
        args['PCL_eliminate_Black_Background'] = True 
        args['PCL_eliminate_White_Background'] = False        
        args['PCL_Color_perturbation_Augmentation'] = False # Whether to use color normalization augmentation in H&E white-field microscopy images. Used to force PCL to learn morphological features and avoid learning color staining features. 
        args['PCL_GPU_INDEX']= torch.device('cuda:0')

    if '23B10981_exp2' in path:
        args['PCL_CNN_Architecture']='resnet50' # or resnet18
        args['PCL_N_Workers']=10 # Number of data loading workers 
        args['PCL_Epochs']=250 # Number of total epoch to run
        args['PCL_N_Crops_per_Image']=15# Number of crops chosen per image
        args['PCL_Patch_Size']=15 # Size of patches in pixels
        args['PCL_Stride']=0 # Size of stride between consecutive patches
        args['PCL_Alpha_L']=1.15 # size ratio between image crop and image patch
        args['PCL_Z_Score']= True # Whether to apply z score per channel/marker
        args['PCL_Batch_Size']= 40 # Number of images per iteration
        args['PCL_Weight_Decay']= 1e-4 # Weight Decay
        args['PCL_Out_Dimensions']= 128 # Feature dimensions of patch
        args['PCL_Learning_Rate']= 0.0003 # Learning Rate
        args['PCL_Temperature']= 0.07 # Softmax temperature    
        args['PCL_eliminate_Black_Background'] = True 
        args['PCL_eliminate_White_Background'] = False        
        args['PCL_Color_perturbation_Augmentation'] = False # Whether to use color normalization augmentation in H&E white-field microscopy images. Used to force PCL to learn morphological features and avoid learning color staining features. 
        args['PCL_GPU_INDEX']= torch.device('cuda:1')

    if 'Supervised_step' in path:
        args['num_DAPI_Ch'] = 0
        args['cyto_channels'] = [0,1,2,3,4,5]
        args['PCL_CNN_Architecture']='resnet50' # or resnet18
        args['PCL_N_Workers']=20 # Number of data loading workers 
        args['PCL_Epochs']=200 # Number of total epoch to run
        args['PCL_N_Crops_per_Image']=20# Number of crops chosen per image
        args['PCL_Patch_Size']=12 # Size of patches in pixels
        args['PCL_Stride']=0 # Size of stride between consecutive patches
        args['PCL_Alpha_L']=1.15 # size ratio between image crop and image patch
        args['PCL_Z_Score']= True # Whether to apply z score per channel/marker
        args['PCL_Batch_Size']= 40 # Number of images per iteration
        args['PCL_Weight_Decay']= 1e-4 # Weight Decay
        args['PCL_Out_Dimensions']= 64 # Feature dimensions of patch
        args['PCL_Learning_Rate']= 0.0003 # Learning Rate
        args['PCL_Temperature']= 0.07 # Softmax temperature    
        args['PCL_eliminate_Black_Background'] = True 
        args['PCL_eliminate_White_Background'] = False        
        args['PCL_Color_perturbation_Augmentation'] = False # Whether to use color normalization augmentation in H&E white-field microscopy images. Used to force PCL to learn morphological features and avoid learning color staining features. 
        args['PCL_GPU_INDEX']= torch.device('cuda:1') # GPU where the training is carried out 

    # if 'panel33c' in path:
    #     args['PCL_CNN_Architecture']='resnet50' # or resnet18
    #     args['PCL_N_Workers']=20 # Number of data loading workers 
    #     args['PCL_Epochs']=200 # Number of total epoch to run
    #     args['PCL_N_Crops_per_Image']=50# Number of crops chosen per image
    #     args['PCL_Patch_Size']=20 # Size of patches in pixels
    #     args['PCL_Stride']=0 # Size of stride between consecutive patches
    #     args['PCL_Alpha_L']=1.15 # size ratio between image crop and image patch
    #     args['PCL_Z_Score']= True # Whether to apply z score per channel/marker
    #     args['PCL_Batch_Size']= 40 # Number of images per iteration
    #     args['PCL_Weight_Decay']= 1e-4 # Weight Decay
    #     args['PCL_Out_Dimensions']= 128 # Feature dimensions of patch
    #     args['PCL_Learning_Rate']= 0.0003 # Learning Rate
    #     args['PCL_Temperature']= 0.07 # Softmax temperature    
    #     args['PCL_eliminate_Black_Background'] = True 
    #     args['PCL_eliminate_White_Background'] = False        
    #     args['PCL_Color_perturbation_Augmentation'] = False # Whether to use color normalization augmentation in H&E white-field microscopy images. Used to force PCL to learn morphological features and avoid learning color staining features. 
    #     args['PCL_GPU_INDEX']= torch.device('cuda:1') # GPU where the training is carried out 

    if 'selected_patches_20img' in path:
        args['PCL_CNN_Architecture']='resnet50' # or resnet18
        args['PCL_N_Workers']=20 # Number of data loading workers 
        args['PCL_Epochs']=200 # Number of total epoch to run
        args['PCL_N_Crops_per_Image']=50# Number of crops chosen per image
        args['PCL_Patch_Size']=20 # Size of patches in pixels
        args['PCL_Stride']=0 # Size of stride between consecutive patches
        args['PCL_Alpha_L']=1.15 # size ratio between image crop and image patch
        args['PCL_Z_Score']= True # Whether to apply z score per channel/marker
        args['PCL_Batch_Size']= 40 # Number of images per iteration
        args['PCL_Weight_Decay']= 1e-4 # Weight Decay
        args['PCL_Out_Dimensions']= 128 # Feature dimensions of patch
        args['PCL_Learning_Rate']= 0.0003 # Learning Rate
        args['PCL_Temperature']= 0.07 # Softmax temperature    
        args['PCL_eliminate_Black_Background'] = True 
        args['PCL_eliminate_White_Background'] = False        
        args['PCL_Color_perturbation_Augmentation'] = False # Whether to use color normalization augmentation in H&E white-field microscopy images. Used to force PCL to learn morphological features and avoid learning color staining features. 
        args['PCL_GPU_INDEX']= torch.device('cuda:1') # GPU where the training is carried out 
    # if 'exp' in path:
    #     # # los canales empiezan en 0  
    #     # args['num_DAPI_Ch'] = 0
    #     # args['cyto_channels'] = [2,9,24,25]

    #     args['PCL_CNN_Architecture']='resnet50' # or resnet18
    #     args['PCL_N_Workers']=20 # Number of data loading workers 
    #     args['PCL_Epochs']=200 # Number of total epoch to run
    #     args['PCL_N_Crops_per_Image']=50# Number of crops chosen per image
    #     args['PCL_Patch_Size']=20 # Size of patches in pixels
    #     args['PCL_Stride']=0 # Size of stride between consecutive patches
    #     args['PCL_Alpha_L']=1.15 # size ratio between image crop and image patch
    #     args['PCL_Z_Score']= True # Whether to apply z score per channel/marker
    #     args['PCL_Batch_Size']= 40 # Number of images per iteration
    #     args['PCL_Weight_Decay']= 1e-4 # Weight Decay
    #     args['PCL_Out_Dimensions']= 128 # Feature dimensions of patch
    #     args['PCL_Learning_Rate']= 0.0003 # Learning Rate
    #     args['PCL_Temperature']= 0.07 # Softmax temperature    
    #     args['PCL_eliminate_Black_Background'] = True 
    #     args['PCL_eliminate_White_Background'] = False        
    #     args['PCL_Color_perturbation_Augmentation'] = False # Whether to use color normalization augmentation in H&E white-field microscopy images. Used to force PCL to learn morphological features and avoid learning color staining features. 
    #     args['PCL_GPU_INDEX']= torch.device('cuda:1') # GPU where the training is carried out 
        
    if '5Ch' in path or '10ch' in path or '17ch' in path or '25Ch' in path:
        # los canales empiezan en 0  
        args['num_DAPI_Ch'] = 0
        args['cyto_channels'] = [2,9,24,25]

        args['PCL_CNN_Architecture']='resnet50' # or resnet18
        args['PCL_N_Workers']=10 # Number of data loading workers 
        args['PCL_Epochs']=200 # Number of total epoch to run
        args['PCL_N_Crops_per_Image']=100 # Number of crops chosen per image
        args['PCL_Patch_Size']=20 # Size of patches in pixels
        args['PCL_Stride']=0 # Size of stride between consecutive patches
        args['PCL_Alpha_L']=1.15 # size ratio between image crop and image patch
        args['PCL_Z_Score']= True # Whether to apply z score per channel/marker
        args['PCL_Batch_Size']= 40 # Number of images per iteration
        args['PCL_Weight_Decay']= 1e-4 # Weight Decay
        args['PCL_Out_Dimensions']= 64 # Feature dimensions of patch
        args['PCL_Learning_Rate']= 0.0003 # Learning Rate
        args['PCL_Temperature']= 0.07 # Softmax temperature    
        args['PCL_eliminate_Black_Background'] = True 
        args['PCL_eliminate_White_Background'] = False        
        args['PCL_Color_perturbation_Augmentation'] = False # Whether to use color normalization augmentation in H&E white-field microscopy images. Used to force PCL to learn morphological features and avoid learning color staining features. 
        args['PCL_GPU_INDEX']= torch.device('cuda:0') # GPU where the training is carried out 
    if 'tonsil_43plex' in path or 'tonsil_43plex_sinfuncion' in path:
        # los canales empiezan en 0  
        args['num_DAPI_Ch'] = 0
        args['cyto_channels'] = [2,9,24,25]

        args['PCL_CNN_Architecture']='resnet18' # or resnet18
        args['PCL_N_Workers']=10 # Number of data loading workers 
        args['PCL_Epochs']=200 # Number of total epoch to run
        args['PCL_N_Crops_per_Image']=100 # Number of crops chosen per image
        args['PCL_Patch_Size']=20 # Size of patches in pixels
        args['PCL_Stride']=0 # Size of stride between consecutive patches
        args['PCL_Alpha_L']=1.15 # size ratio between image crop and image patch
        args['PCL_Z_Score']= True # Whether to apply z score per channel/marker
        args['PCL_Batch_Size']= 40 # Number of images per iteration
        args['PCL_Weight_Decay']= 1e-4 # Weight Decay
        args['PCL_Out_Dimensions']= 128 # Feature dimensions of patch
        args['PCL_Learning_Rate']= 0.0003 # Learning Rate
        args['PCL_Temperature']= 0.07 # Softmax temperature    
        args['PCL_eliminate_Black_Background'] = True 
        args['PCL_eliminate_White_Background'] = False        
        args['PCL_Color_perturbation_Augmentation'] = False # Whether to use color normalization augmentation in H&E white-field microscopy images. Used to force PCL to learn morphological features and avoid learning color staining features. 
        args['PCL_GPU_INDEX']= torch.device('cuda:0') # GPU where the training is carried out 
    return args
