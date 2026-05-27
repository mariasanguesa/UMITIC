import os

# Ruta de la carpeta con las imágenes
carpeta = "Z:/sanguesa.127969/imgs/CRC_public_dataset/bestFocus_tmaB_hyperstacks/"

# Recorrer todos los archivos de la carpeta
for nombre_archivo in os.listdir(carpeta):
    if nombre_archivo.lower().endswith(".tif"):
        ruta_original = os.path.join(carpeta, nombre_archivo)
        
        # Separar nombre y extensión
        nombre, extension = os.path.splitext(nombre_archivo)
        
        # Crear nuevo nombre
        nuevo_nombre = f"{nombre}_B{extension}"
        ruta_nueva = os.path.join(carpeta, nuevo_nombre)
        
        # Renombrar archivo
        os.rename(ruta_original, ruta_nueva)

print("Renombrado completado.")