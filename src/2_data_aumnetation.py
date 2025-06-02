import os
from PIL import Image
from shutil import copy2

# Carpeta de entrada y salida
input_dir = '../data_recolor'  
output_dir = '../data_aumt'

# Crear la carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Recorrer todas las subcarpetas (por letra)
for letra in os.listdir(input_dir):
    subcarpeta_in = os.path.join(input_dir, letra)
    subcarpeta_out = os.path.join(output_dir, letra)
    if not os.path.isdir(subcarpeta_in):
        continue
    os.makedirs(subcarpeta_out, exist_ok=True)

    # Recorrer todas las imágenes en la subcarpeta
    for nombre_img in os.listdir(subcarpeta_in):
        ruta_img = os.path.join(subcarpeta_in, nombre_img)
        if not os.path.isfile(ruta_img):
            continue

        # Copiar la imagen original
        copy2(ruta_img, os.path.join(subcarpeta_out, nombre_img))

        # Crear y guardar la imagen espejo
        try:
            img = Image.open(ruta_img)
            img_mirror = img.transpose(Image.FLIP_LEFT_RIGHT)
            nombre_base, ext = os.path.splitext(nombre_img)
            nombre_mirror = f"{nombre_base}_mirror{ext}"
            img_mirror.save(os.path.join(subcarpeta_out, nombre_mirror))
        except Exception as e:
            print(f"Error procesando {ruta_img}: {e}")