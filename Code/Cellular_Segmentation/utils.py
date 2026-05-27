import os
# from dotenv import load_dotenv
# load_dotenv()
import numpy as np
from csbdeep.utils import normalize
from PIL import Image
from skimage.segmentation import watershed
from skimage import color
import SimpleITK as sitk
import tifffile as tiff


def nuclear_segmentation(image, model):

    axis_norm = (0,1)
    image = normalize(image, 1, 99.8, axis=axis_norm)

    nuclear_mask,_= model.predict_instances(image)

    return nuclear_mask


def cyto_segmentation(dapi_image, cyto_image, app):

    CombinedCh = np.stack((dapi_image, cyto_image), axis=-1)
    CombinedCh = np.expand_dims(CombinedCh,0)

    cellular_mask = app.predict(CombinedCh)

    return cellular_mask[0,...,0]


def cellular_segmentation_correction(cellular_mask, nuclear_mask, input_image):
    # input_image = color.rgb2gray(input_image)

    # Apply anisotropic filter
    input_image = sitk.GetImageFromArray(input_image)
    gradient_filter = sitk.GradientAnisotropicDiffusionImageFilter()
    input_image = gradient_filter.Execute(input_image)
    input_image = sitk.GetArrayFromImage(input_image)    

    cyto_seg_supr = cellular_mask.copy()

    sheds = np.zeros(nuclear_mask.shape, dtype=np.uint16)
    mask = np.zeros(nuclear_mask.shape, dtype=bool)

    cellular_mask_correc= np.zeros(cellular_mask.shape, dtype=np.uint16)

    # Unique cytoplasm ids
    cytoplasm_ids = np.unique(cellular_mask)
    cytoplasm_ids = cytoplasm_ids[cytoplasm_ids > 0]

    all_possible_ids = np.arange(1, 65535)

    # Values not in cytoplasm_ids
    shed_ids = np.setdiff1d(all_possible_ids, cytoplasm_ids)
    shed_index = len(shed_ids)

    # Search single cytos
    for cyto_id in cytoplasm_ids:
        # Array equality 
        cytoplasm_pixels = cellular_mask == cyto_id 

        # Search pixel values in nuclear seg in the cyto
        unique_elements = np.unique(nuclear_mask[cytoplasm_pixels])

        if unique_elements.size > 0 and unique_elements[0] == 0:
            # Only background -> cell without nuclei
            if len(unique_elements) == 1:
                cyto_seg_supr[cytoplasm_pixels] = 0
            unique_elements = unique_elements[1:]
            
        # More than 1 nuclei unique_elements -> deepcell error
        if len(unique_elements) > 1:
            # Each shed 
            for j in unique_elements:
                nuclei_pixels = (nuclear_mask == j)
                sheds[nuclei_pixels] = shed_ids[shed_index-1]
                shed_index -= 1
                mask[nuclei_pixels] = True
                unique_cyto = np.unique(cellular_mask[nuclei_pixels])
                if unique_cyto[0] == 0:
                    unique_cyto = unique_cyto[1:]
                if len(unique_cyto) > 1:
                    for x in unique_cyto:
                        cyto_pixels = (cellular_mask == x)
                        mask[cyto_pixels] = True
                        cyto_seg_supr[cyto_pixels] = 0
            mask[cytoplasm_pixels] = True        

    cyto_seg_supr[mask] = 0
    markers = watershed(input_image, sheds, mask=mask)
    cellular_mask_correc = cyto_seg_supr + markers

    # tiff.imwrite('C:/Users/maria.sanguesa/OneDrive - UPNA/pcl_test_images/deepcell_masks/maks.tiff',mask.astype('uint16'))
    # tiff.imwrite('C:/Users/maria.sanguesa/OneDrive - UPNA/pcl_test_images/deepcell_masks/input_image.tiff',input_image.astype('uint16'))

    return cellular_mask_correc
