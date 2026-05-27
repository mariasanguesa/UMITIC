import os
import tifffile as tiff
import numpy as np

def print_multichannel_tiff_sizes(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.tiff') or filename.lower().endswith('.tif'):
            file_path = os.path.join(folder_path, filename)
            try:
                img = tiff.imread(file_path)
                # Considera multicanal si tiene más de 2 dimensiones y la última dimensión > 1
                if img.ndim > 2 and img.shape[-1] > 1:
                    print(f"{filename}: shape={img.shape}, dtype={img.dtype}")
            except Exception as e:
                print(f"Error leyendo {filename}: {e}")

def print_and_fix_multichannel_tiff_sizes(folder_path, target_shape=(8, 2538, 2538)):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.tiff') or filename.lower().endswith('.tif'):
            file_path = os.path.join(folder_path, filename)
            try:
                img = tiff.imread(file_path)
                # Considera multicanal si tiene más de 2 dimensiones y la última dimensión > 1
                if img.ndim > 2 and img.shape[-1] > 1:
                    print(f"{filename}: shape={img.shape}, dtype={img.dtype}")
                    if img.shape != target_shape:
                        print(f"  -> Corrigiendo tamaño a {target_shape}")
                        # Ajustar el shape
                        fixed = np.zeros(target_shape, dtype=img.dtype)
                        # Calcular slices para copiar datos existentes
                        slices = tuple(slice(0, min(s, t)) for s, t in zip(img.shape, target_shape))
                        fixed[slices] = img[slices]
                        tiff.imwrite(file_path, fixed)
            except Exception as e:
                print(f"Error leyendo {filename}: {e}")

if __name__ == "__main__":
    folder = "C:/Users/maria.sanguesa/Desktop/Clusters_exp1/imgs_murino_panel2_linfoide" # Cambia esta ruta por la de tu carpeta
    print_and_fix_multichannel_tiff_sizes(folder)
    print("NUEVOS SHAPES")
    print_multichannel_tiff_sizes(folder)