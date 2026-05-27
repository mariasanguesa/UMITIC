import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.measure import regionprops
import math

def get_cells_from_image(path, image_name, data, n_crops_per_image, crop_size, is_training):
    image_cell_segmented = cv2.imread(path[:-8] + '/cellular_masks/' + image_name, -1)
    image_nuc_segmented = cv2.imread(path[:-8] + '/nuclear_masks/' + image_name, -1)

    image_input = data

    images = []
    patch_pos = []

    array = np.unique(image_cell_segmented)
    array = array[array > 0]

    diag = []
    ecc = []
    comp = []
    pol = []
    nucl = []
    nucl_size = []
    cell_size = []
    mean_ch = []

    _, arr_nuc = cv2.threshold(image_nuc_segmented, 0, 255, cv2.THRESH_BINARY)
    arr_nuc[arr_nuc == 255] = 1

    if is_training:
        cyto_values = np.random.choice(array, n_crops_per_image)
    else:
        cyto_values = array

    for cytoplasm_id in cyto_values:
        mask = (image_cell_segmented == cytoplasm_id)

        y_indices, x_indices = np.where(mask)
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        center_x = (x_max + x_min) // 2
        center_y = (y_max + y_min) // 2

        cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
        cropped_image = image_input[y_min:y_max + 1, x_min:x_max + 1, :]
        cropped_image = cropped_image * cropped_mask[:, :, np.newaxis]

        cell_nucleus = arr_nuc * mask
        cropped_image_nucleus = cell_nucleus[y_min:y_max + 1, x_min:x_max + 1] * cropped_mask

        original_height, original_width = cropped_image.shape[:2]
        aspect_ratio = original_width / original_height

        if aspect_ratio > 1:
            new_width = crop_size
            new_height = max(1, int(crop_size / aspect_ratio))
        else:
            new_height = crop_size
            new_width = max(1, int(crop_size * aspect_ratio))

        padded_image = np.zeros((crop_size, crop_size, cropped_image.shape[2]), dtype=cropped_image.dtype)

        for j in range(cropped_image.shape[2]):
            original_channel = cropped_image[:, :, j]
            original_height, original_width = original_channel.shape
            aspect_ratio = original_width / original_height

            if aspect_ratio > 1:
                new_width = crop_size
                new_height = int(crop_size / aspect_ratio)
            else:
                new_height = crop_size
                new_width = int(crop_size * aspect_ratio)

            resized_channel = cv2.resize(original_channel, (new_width, new_height), interpolation=cv2.INTER_AREA)

            delta_w = crop_size - new_width
            delta_h = crop_size - new_height
            top = delta_h // 2
            left = delta_w // 2

            padded_image[top:top+new_height, left:left+new_width, j] = resized_channel
        images.append(padded_image)

        if not is_training:
            patch_pos.append([center_x, center_y])

            props = regionprops(cropped_mask.astype(np.uint8))
            if props:
                prop = props[0]
                diag_1 = prop.major_axis_length
                diag.append(diag_1)
                ecc.append(prop.eccentricity)

                perimeter = prop.perimeter
                cell_area = prop.area
                complexity = (2 * np.sqrt(np.pi * cell_area)) / perimeter if perimeter > 0 else 0
                comp.append(complexity)
                cell_size.append(cell_area)

                com_y, com_x = prop.centroid
            else:
                diag_1 = 0
                diag.append(0)
                ecc.append(0)
                comp.append(0)
                cell_size.append(0)
                com_y, com_x = 0, 0

            mean_channel = image_input[mask].mean(axis=0)
            mean_ch.append([mean_channel])

            props_nuc = regionprops(cropped_image_nucleus.astype(np.uint8))
            if props_nuc:
                prop_nuc = props_nuc[0]
                com_y_nuc, com_x_nuc = prop_nuc.centroid
                polarity = math.dist([com_x_nuc, com_y_nuc], [com_x, com_y])
                polarity = polarity / (diag_1 / 2) if diag_1 > 0 else 0
                pol.append(polarity)

                nucl_to_cell = prop_nuc.area / cell_area if cell_area > 0 else 0
                nucl.append(nucl_to_cell)
                nucl_size.append(prop_nuc.area)
            else:
                pol.append(0)
                nucl.append(0)
                nucl_size.append(0)

    if is_training:
        return images
    else:
        return images, patch_pos, diag, ecc, comp, pol, nucl, nucl_size, cell_size, mean_ch

# get_cells_from_image('C:/Users/maria.sanguesa/Desktop/01_Experiment_Panel33C/01_Panel33C/Raw_Data/Images/','Panel33C_Scan2_1_12.tiff',tiff.imread('C:/Users/maria.sanguesa/Desktop/01_Experiment_Panel33C/01_Panel33C/Raw_Data/Images/Panel33C_Scan2_1_12.tiff'),50,30,True)