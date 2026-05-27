# from tkinter import N

from Code.Patch_Contrastive_Learning.data_aug.cells_from_image import get_cells_from_image
from Code.utils import utilz
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from skimage import io
import random
import gc
import csv
from Code.Patch_Contrastive_Learning.preprocess_images import Mean_std_experiment #, Mean_std_experiment_fast
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if available
import matplotlib.pyplot as plt

class ContrastiveLearningDataset_PCL(Dataset):
    def __init__(self, args,training):
        self.training = training

        # Root folder
        self.root_folder = args['path']+'Raw_Data/Images/'

        # Image list
        self.image_list = os.listdir(self.root_folder)
        random.shuffle(self.image_list)

        # Open Channels.txt
        self.Channels, _ = utilz.load_channels(args['path']+'Raw_Data/')
        self.in_channels = len(self.Channels)        

        # Parameters
        self.args = args    
        self.crop_size = args['Crop_Size']
        # Alpha_l: size ratio between image crop and image patch
        self.alpha_L = 1
        self.n_crops_per_image = args['N_Crops_per_Image']    

        # Preprocessing
        if self.args['Z_Score']:
            if os.path.exists(self.root_folder[:-7]+'Experiment_Stats.csv'):
                f = csv.reader(open(self.root_folder[:-7]+'Experiment_Stats.csv'))
                self.mean = np.array(next(f),dtype=np.float32)
                self.std = np.array(next(f),dtype=np.float32)      
            else : 
                self.mean, self.std = Mean_std_experiment(obj=self, base_path=self.root_folder,image_paths=self.image_list) 
    
    def __len__(self):        
        return len(self.image_list)

    def load_image(self,idx):

        # Load CZI file
        if self.image_list[idx].endswith('.czi'):
            from aicspylibczi import CziFile
            import pathlib
            import cv2
            
            # Obtain header info from the czi
            factor = 5
            file = pathlib.Path(self.root_folder+self.image_list[idx])
            czi = CziFile(file)
            self.czidimensions = czi.get_mosaic_scene_bounding_box(0)    
            dims = czi.get_dims_shape()        
            
            # Obtain crop or the full image
            if self.training:
                xIn,yIn = np.random.randint(0,self.czidimensions.w-int(self.crop_size*factor*self.alpha_L)-1), np.random.randint(0,self.czidimensions.h-int(self.crop_size*factor*self.alpha_L)-1)
                image = czi.read_mosaic(C=0,region=(self.czidimensions.x+xIn,self.czidimensions.y+yIn,int(self.crop_size*factor*self.alpha_L),int(self.crop_size*factor*self.alpha_L)))
                new_size_x, new_size_y  = int(self.crop_size*self.alpha_L), int(self.crop_size*self.alpha_L)
            else:    
                image = czi.read_mosaic(C=0,region=(self.czidimensions.x,self.czidimensions.y,self.czidimensions.w,self.czidimensions.h))
                new_size_x, new_size_y  = int(self.czidimensions.w/factor), int(self.czidimensions.h/factor)
            
            # Postprocessing and image resizing
            image = image.squeeze()
            image = cv2.resize(image, dsize=(new_size_x, new_size_y), interpolation=cv2.INTER_CUBIC)
            self.wsiIm = czi

        elif self.image_list[idx].endswith('.ome.tif'): 
            image = io.imread(self.root_folder+self.image_list[idx])              
            image = image[:,:,self.Channels]
            if self.args['Z_Score'] and hasattr(self, 'mean'):
                image = image-np.expand_dims(np.expand_dims(self.mean,0),0)
                image = image/np.expand_dims(np.expand_dims(self.std,0),0)

        # Load TIFF file
        elif self.image_list[idx].endswith('.tif') or self.image_list[idx].endswith('.tiff'): 
            image = io.imread(self.root_folder+self.image_list[idx])                
            
            if np.argmin(image.shape)==0:
                image = image.transpose((1, -1, 0))                           
            
            image = image[:,:,self.Channels]
            if self.args['Z_Score'] and hasattr(self, 'mean'):
                image = (image - self.mean[np.newaxis, np.newaxis,:]) / self.std[np.newaxis, np.newaxis,:]
        else:
            image =  np.load(self.root_folder+self.image_list[idx],allow_pickle=True)  
           
        return image

    def get_crops(self, data,idx):
        images = get_cells_from_image(self.root_folder, self.image_list[idx], data, self.n_crops_per_image, self.crop_size,True)

        return images  

    def get_all_crops(self, data,idx):
        # Obtain indices considering crop size
        patches = []  
        patch_pos = []
        mean_ch = []     

        patches, patch_pos, diags, ecc, comp, pol, nucl, nucl_size, cell_size, mean_ch = get_cells_from_image(self.root_folder,self.image_list[idx],data,None, self.crop_size,False)
        # patches,patch_pos,diags = get_cells_from_image(self.root_folder,self.image_list[idx],data,None, self.crop_size,False)

        return patches, np.array(patch_pos), diags, ecc, comp, pol, nucl, nucl_size, cell_size, mean_ch

    # def perform_augmentation(self, crop):        
    #     # Define sequence of augmentations
    #     seq_easy = iaa.Sequential([
    #         iaa.CoarseDropout(0.02, size_percent=0.05, per_channel=0.4), # Drop 2% of all pixels, 15% of the original size,in 50% of all patches channels
    #         iaa.Add((-0.15, 0.15), per_channel=0.5), # Add noise per channel...
    #         iaa.AdditiveGaussianNoise(scale=(0.0, 0.1), per_channel=0.4),
    #         # iaa.GaussianBlur(sigma=(0, 1)), # blur patches with a sigma of 0 to 3.0
    #         # iaa.Fliplr(0.5), # horizontal flips
    #         # iaa.Flipud(0.5),
    #         iaa.Multiply((0.9, 1.1), per_channel=1), #Intensity scaling
    #         # iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, rotate=(0, 360),) # Scaling, rotate
    #     ], random_order=True)                
    
    #     # Flip lr
    #     if random.random()<0.5:
    #         for c in range(crop.shape[2]):
    #             crop[:,:,c] = np.fliplr(crop[:,:,c])

    #     # Flip ud
    #     if random.random()<0.5:
    #         for c in range(crop.shape[2]):
    #             crop[:,:,c] = np.flipud(crop[:,:,c])

    #     # Rotate
    #     if random.random()<0.5:
    #         for c in range(crop.shape[2]):
    #             crop[:,:,c] = np.rot90(crop[:,:,c],-1)
    #     # Rotate
    #     if random.random()<0.5:
    #         for c in range(crop.shape[2]):
    #             crop[:,:,c] = np.rot90(crop[:,:,c],1)
        
    #     # Rotate
    #     if random.random()<0.5:
    #         for c in range(crop.shape[2]):
    #             crop[:,:,c] = np.rot90(crop[:,:,c],1)

    #     return seq_easy(images=crop.astype(np.float32))  

    def perform_augmentation(self, crop):
        # una sola rotación en {0, 90, 180, 270} + flips opcionales
        k = random.randint(0, 3)
        if k:
            crop = np.rot90(crop, k=k, axes=(0, 1))

        if random.random() < 0.5:
            crop = np.fliplr(crop)

        if random.random() < 0.5:
            crop = np.flipud(crop)
        mask = np.any(np.abs(crop) > 0, axis=2)
        seq = iaa.Sequential([
            iaa.Sometimes(0.6, iaa.Multiply((0.9, 1.1), per_channel=0.5)),
            iaa.Sometimes(0.3, iaa.Add((-0.02, 0.02), per_channel=0.5)),
            iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=(0.0, 0.02), per_channel=0.5)),
            iaa.Sometimes(0.15, iaa.GaussianBlur(sigma=(0.0, 0.6))),
            iaa.Sometimes(0.08, iaa.CoarseDropout(p=(0.0, 0.003), size_px=(1, 2), per_channel=0.1)),
        ], random_order=True)

        aug = seq(images=crop.astype(np.float32))
        aug[~mask] = 0
        return aug

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        image = self.load_image(idx)    
        # image = image.astype(np.float32)    

        if self.training:
            # Obtain a crop of size Patch_size
            crops = self.get_crops(image,idx)

            # Perform sequential image transformations
            augs_1 = []
            augs_2 = []
            for crop in crops:
                augs_1.append(self.perform_augmentation(crop))
                augs_2.append(self.perform_augmentation(crop))
            del image, crops
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
            return torch.tensor(np.array(augs_1)), torch.tensor(np.array(augs_2))

        else:
            crops, patch_pos, diags, ecc, comp, pol, nucl, nucl_size, cell_size, mean_ch = self.get_all_crops(image,idx)
            # crops, patch_pos,diags = self.get_all_crops(image,idx)

            del image
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)
            return crops, patch_pos, self.image_list[idx], diags, ecc, comp, pol, nucl, nucl_size, cell_size, mean_ch

# visualizar augmentation
def _robust_norm(x, lo, hi):
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)

def _build_rgb_shared_scale(images_hwc, rgb_channels=(0, 1, 2), pmin=1, pmax=99):
    """
    images_hwc: lista de arrays HxWxC (original + augmentadas)
    Devuelve lista de pseudo-RGB con la misma escala por canal para comparar justo.
    """
    ch0 = np.concatenate([im[..., rgb_channels[0]].ravel() for im in images_hwc])
    ch1 = np.concatenate([im[..., rgb_channels[1]].ravel() for im in images_hwc])
    ch2 = np.concatenate([im[..., rgb_channels[2]].ravel() for im in images_hwc])

    lo0, hi0 = np.percentile(ch0, [pmin, pmax])
    lo1, hi1 = np.percentile(ch1, [pmin, pmax])
    lo2, hi2 = np.percentile(ch2, [pmin, pmax])

    rgbs = []
    for im in images_hwc:
        r = _robust_norm(im[..., rgb_channels[0]], lo0, hi0)
        g = _robust_norm(im[..., rgb_channels[1]], lo1, hi1)
        b = _robust_norm(im[..., rgb_channels[2]], lo2, hi2)
        rgbs.append(np.stack([r, g, b], axis=-1))
    return rgbs

def visualize_original_plus_augs(dataset, idx=0, crop_idx=0, rgb_channels=(0, 1, 2), n_augs=6, seed=None):
    """
    Muestra en una sola figura: original + n_augs augmentations del mismo crop.
    Requiere dataset con métodos load_image, get_crops y perform_augmentation.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1) Cargar crop original
    image = dataset.load_image(idx)
    crops = dataset.get_crops(image, idx)
    crop = crops[crop_idx].astype(np.float32, copy=True)

    # 2) Generar augmentations del mismo crop
    aug_list = [dataset.perform_augmentation(crop.copy()).astype(np.float32) for _ in range(n_augs)]

    # 3) Construir pseudo-RGB con escala compartida (comparación consistente)
    all_views = [crop] + aug_list
    rgb_views = _build_rgb_shared_scale(all_views, rgb_channels=rgb_channels, pmin=1, pmax=99)

    # 4) Plot
    cols = n_augs + 1
    fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.6))
    if cols == 1:
        axes = [axes]

    titles = ["Original"] + [f"Aug {i}" for i in range(1, n_augs + 1)]
    for ax, rgb, t in zip(axes, rgb_views, titles):
        ax.imshow(rgb)
        ax.set_title(t)
        ax.axis("off")

    fig.suptitle(f"Crop {crop_idx} | Image idx {idx} | RGB channels {rgb_channels}", y=1.02)
    plt.tight_layout()
    plt.show()

    return crop, aug_list

