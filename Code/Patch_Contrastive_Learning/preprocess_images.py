"""
This module normalizes the cohort based on the mean and standard deviation.
"""

import numpy as np
from tqdm import tqdm
import csv
import concurrent.futures
import math
import random
import os
from typing import Sequence

def Mean_std_experiment(obj, base_path, image_paths):
    '''
    Obtain mean and standard deviation from the cohort (memory efficient)
    '''
    n_images = len(image_paths)
    sum_channels = None
    sum_sq_channels = None
    n_pixels_total = 0

    for n_im in tqdm(range(n_images), ascii=True, desc='Calculate Mean/Std'):
        image = obj.load_image(n_im)  # shape: (H, W, C)
        image = image.astype(np.float64) 

        if sum_channels is None:
            sum_channels = np.zeros(image.shape[-1], dtype=np.float64)
            sum_sq_channels = np.zeros(image.shape[-1], dtype=np.float64)

        # Calcular sumas y sumas de cuadrados por canal
        sum_channels += np.sum(image, axis=(0, 1))
        sum_sq_channels += np.sum(np.square(image), axis=(0, 1))

        # Contar píxeles totales por canal (H*W)
        n_pixels_total += image.shape[0] * image.shape[1]

    # Calcular media y desviación típica por canal
    mean_per_channel = sum_channels / n_pixels_total
    var_per_channel = (sum_sq_channels / n_pixels_total) - np.square(mean_per_channel)
    std_per_channel = np.sqrt(var_per_channel)

    with open(base_path[:-7]+'Experiment_Stats.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(mean_per_channel)
        writer.writerow(std_per_channel)

    return mean_per_channel, std_per_channel


# def Mean_std_experiment_fast(obj, base_path, image_paths, max_workers: int = None, sample_fraction: float = 1.0):
#     '''
#     Faster, thread-parallel version to compute mean and std over a large cohort.

#     Args:
#         obj: object providing `load_image(index)`.
#         base_path: path used to write Experiment_Stats.csv (same behaviour as original).
#         image_paths: sequence of image identifiers (only length is used here).
#         max_workers: number of threads to use (default: min(32, os.cpu_count()*4)).
#         sample_fraction: fraction of images to sample (0 < sample_fraction <= 1.0). If < 1.0,
#                          a random subset of images is used to estimate stats much faster.

#     Returns:
#         (mean_per_channel, std_per_channel)
#     '''
#     if not (0 < sample_fraction <= 1.0):
#         raise ValueError('sample_fraction must be in (0, 1].')

#     n_images = len(image_paths)
#     if n_images == 0:
#         raise ValueError('image_paths is empty')

#     # Select indices to process (sampling to speed up)
#     if sample_fraction < 1.0:
#         k = max(1, math.floor(n_images * sample_fraction))
#         indices = random.sample(range(n_images), k)
#     else:
#         indices = list(range(n_images))

#     if max_workers is None:
#         max_workers = min(32, (os.cpu_count() or 4) * 4)

#     sum_channels = None
#     sum_sq_channels = None
#     n_pixels_total = 0

#     def _process_index(n_im):
#         image = obj.load_image(n_im)
#         image = image.astype(np.float64)
#         if image.ndim == 2:
#             image = np.expand_dims(image, axis=-1)

#         s = np.sum(image, axis=(0, 1), dtype=np.float64)
#         ss = np.sum(np.square(image), axis=(0, 1), dtype=np.float64)
#         n_pixels = image.shape[0] * image.shape[1]
#         return s, ss, n_pixels

#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exc:
#         futures = {exc.submit(_process_index, idx): idx for idx in indices}
#         for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), ascii=True, desc='Calculate Mean/Std (fast)'):
#             s, ss, n_pixels = fut.result()

#             if sum_channels is None:
#                 sum_channels = np.zeros_like(s, dtype=np.float64)
#                 sum_sq_channels = np.zeros_like(ss, dtype=np.float64)

#             sum_channels += s
#             sum_sq_channels += ss
#             n_pixels_total += n_pixels

#     mean_per_channel = sum_channels / n_pixels_total
#     var_per_channel = (sum_sq_channels / n_pixels_total) - np.square(mean_per_channel)
#     std_per_channel = np.sqrt(np.maximum(var_per_channel, 0.0))

#     with open(base_path[:-7]+'Experiment_Stats.csv', 'w', encoding='UTF8', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(mean_per_channel)
#         writer.writerow(std_per_channel)

#     return mean_per_channel, std_per_channel

# def Mean_std_experiment(obj, base_path, image_paths):    
#     ''' 
#     Obtain mean and standard deviation from the cohort
#     base_path: (string) that specifies the directory where the experiment is carried out.
#     images_paths: (list of strings) that specifies the names of the files executed.
#     '''
#     images_array = []
#     # Read slide by slide
#     for n_im in tqdm(range(len(image_paths)), ascii=True, desc='Calculate Mean and Standard deviation'): 
        
#         # Load Image
#         image = obj.load_image(n_im)
#         images_array.append(image)

#     images_array = np.array(images_array)

#     # BEGIN: modify_shape_handling
#     if images_array.ndim == 1:
#         images_array = np.stack(images_array)  # Ensure proper shape for mean/std calculation
#     elif images_array.ndim == 3:
#         images_array = np.expand_dims(images_array, axis=0)  # Add a new dimension to make it 4D
#     # END: modify_shape_handling

#     mean = np.mean(images_array, axis=(0, 1, 2))  # Calculate mean across the last dimension (channels)
#     std = np.std(images_array, axis=(0, 1, 2))

#     # mean = np.mean(images_array, axis=(0, 2, 3))  # Calculate mean across the first dimension (images)
#     # std = np.std(images_array, axis=(0, 2, 3))

#     with open(base_path[:-7]+'Experiment_Stats.csv', 'w', encoding='UTF8', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(mean)
#         writer.writerow(std)
    
#     return mean, std

# def Mean_std_experiment(obj, base_path,image_paths):    
#     ''' 
#     Obtain mean and standard deviation from the cohort
#     base_path: (string) that specifies the directory where the experiment is carried out.
#     images_paths: (list of strings) that specifies the names of the files executed.
#     Channels: (vector of int) channels that should be included in the experiment.    
#     '''
    
#     # Read slide by slide
#     for n_im in tqdm(range(len(image_paths)),ascii=True,desc='Calculate Mean and Standard deviation'): 
        
#         # Load Image
#         image = obj.load_image(n_im)

#         # Reduce image size if it is too large
#         max_num_pix = 5000000
#         max_num_pix_dim = int(max_num_pix**0.5)
#         im_s = image.shape
#         if im_s[0]>max_num_pix_dim and im_s[1]>max_num_pix_dim:
#             image = image[0:im_s[0]:int(im_s[0]/max_num_pix_dim),0:im_s[1]:int(im_s[1]/max_num_pix_dim),:]
        
#         # To concatenate image information we sum the histograms of several images.
#         if n_im==0:
#             minImage = image.min(tuple(range(len(image.shape)-1)))
#             minImage = [m*10 if m<0 else m/10 for m in minImage]
#             maxImage = image.max(tuple(range(len(image.shape)-1)))
#             maxImage = [m/10 if m<0 else m*10 for m in maxImage]
#             Global_hist = [list(np.histogram(np.concatenate((image[:,:,i].flatten(),np.arange(minImage[i],maxImage[i],(maxImage[i]-minImage[i])/10000))),range=(minImage[i],maxImage[i]),bins=10000)) for i in range(image.shape[-1])]                                    
#         else:
#             Local_hist = [list(np.histogram(np.concatenate((image[:,:,i].flatten(),np.arange(minImage[i],maxImage[i],(maxImage[i]-minImage[i])/10000))),range=(minImage[i],maxImage[i]),bins=10000)) for i in range(image.shape[-1])]                                    
#             for n_g_h, g_h in enumerate(Global_hist):
#                 g_h[0] += Local_hist[n_g_h][0]      

#         del image 

#     # Calculate Mean
#     mean = []    
#     for g_h in Global_hist:
#         hist_WA = []
#         den = 0
#         num = 0
#         for g_n, g_h_h in enumerate(g_h[0]):
#             den+=(g_h_h-(n_im+1)+1e-16)
#             num+=g_h[1][g_n]*(g_h_h-(n_im+1)+1e-16)
#         mean.append(num/den)
    
#     # Calculate Standard deviation
#     std = []
#     for hn, g_h in enumerate(Global_hist):
#         hist_WA = []
#         den = 0
#         num = 0
#         for g_n, g_h_h in enumerate(g_h[0]):
#             den+=(g_h_h-(n_im+1)+1e-16)
#             num+=((g_h[1][g_n]-mean[hn])**2)*(g_h_h-(n_im+1)+1e-16)
#         std.append((num/den)**0.5)

#     # Save mean and std to file
#     with open(base_path[:-7]+'Experiment_Stats.csv', 'w', encoding='UTF8') as f:
#         writer = csv.writer(f)
#         writer.writerow(mean)
#         writer.writerow(std)

#     return np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)