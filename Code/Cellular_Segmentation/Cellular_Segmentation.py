import os
import numpy as np
import tifffile as tiff
from Code.Cellular_Segmentation.utils import nuclear_segmentation, cyto_segmentation, cellular_segmentation_correction

def mask_generation(img_name, path, nuclear_seg_path, cellular_seg_raw_path, cellular_seg_path, nuclear_model, cyto_model, num_DAPI_Ch, cyto_channels):
    img_path = os.path.join(path+'Raw_Data/Images/', img_name)
    image = tiff.imread(img_path)
    image = image.transpose((1, -1, 0))

    nuclear_mask_file = os.path.join(nuclear_seg_path, img_name)
    cellular_mask_file_raw = os.path.join(cellular_seg_raw_path, img_name)
    cellular_mask_file = os.path.join(cellular_seg_path, img_name)

    # Nuclear segmentation
    if not os.path.exists(nuclear_mask_file):
        nuclear_mask = nuclear_segmentation(image[:, :, num_DAPI_Ch], nuclear_model)
        tiff.imwrite(nuclear_mask_file, nuclear_mask.astype('uint16'))

    # Cellular segmentation
    if not os.path.exists(cellular_mask_file_raw):
        cyto_image = np.zeros((image.shape[0], image.shape[1]))
        for ch_num in cyto_channels:
            cyto_image = cyto_image + image[:, :, ch_num]
        cellular_mask = cyto_segmentation(image[:, :, num_DAPI_Ch], cyto_image, cyto_model)
        tiff.imwrite(cellular_mask_file_raw, cellular_mask.astype('uint16'))

    # Cellular segmentation correction
    if not os.path.exists(cellular_mask_file):
        cyto_image = np.zeros((image.shape[0], image.shape[1]))
        for ch_num in cyto_channels:
            cyto_image = cyto_image + image[:, :, ch_num]
        nuclear_mask = tiff.imread(nuclear_mask_file)
        cellular_mask = tiff.imread(cellular_mask_file_raw)
        cellular_mask_correc = cellular_segmentation_correction(cellular_mask, nuclear_mask, cyto_image)
        tiff.imwrite(cellular_mask_file, cellular_mask_correc.astype('uint16'))


def cellular_segmentation(args):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    import tensorflow as tf
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    # Importa modelos después de configurar TF
    from stardist.models import StarDist2D
    from deepcell.applications import Mesmer
    # Load models
    #nuclear_model = StarDist2D(None, "C:/Users/maria.sanguesa/OneDrive - UPNA/UMITIC/Code/Cellular_Segmentation/trainedModel")
    nuclear_model = StarDist2D(None, "C:/Users/sanguesa.127969/OneDrive - UPNA/UMITIC/Code/Cellular_Segmentation/trainedModel")

    cyto_model = Mesmer()

    # Output folders
    nuclear_seg_path = args['path']+'Raw_Data/nuclear_masks/'
    cellular_seg_path = args['path']+'Raw_Data/cellular_masks/'
    cellular_seg_raw_path = args['path']+'Raw_Data/cellular_masks_raw/'
    os.makedirs(nuclear_seg_path, exist_ok=True)
    os.makedirs(cellular_seg_path, exist_ok=True)
    os.makedirs(cellular_seg_raw_path, exist_ok=True)

    # Create cellular/nuclear masks
    imagenes_tiff = [f for f in os.listdir(args['path']+'Raw_Data/Images/') if f.endswith('.tiff') or f.endswith('.tif')]
    for img_name in imagenes_tiff:
            mask_generation(img_name, args['path'], nuclear_seg_path,cellular_seg_raw_path, cellular_seg_path, nuclear_model, cyto_model, args['num_DAPI_Ch'], args['cyto_channels'])
    
    # Delete models for memory optimization
    del nuclear_model
    del cyto_model
