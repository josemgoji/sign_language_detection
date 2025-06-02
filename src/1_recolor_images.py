import os
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import shutil

# Inicializa el modelo una sola vez
img_colorization = pipeline(Tasks.image_colorization, model='damo/cv_ddcolor_image-colorization')

input_root = '../data'
output_root = '../data_recolor'

for subdir, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(subdir, file)
            # Construye la ruta de salida manteniendo la estructura
            relative_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Lee y procesa la imagen
            img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
            result = img_colorization(img)
            recolored = result[OutputKeys.OUTPUT_IMG]

            # Guarda la imagen recolorizada
            cv2.imwrite(output_path, recolored)
            print(f'Guardada: {output_path}')

# --- Mover las imagenes de letras faltantes a la carpeta de recolorización que ya estaban a color---

origen = '../data_extra/letras_faltantes'

destino = '../data_recolor'

os.makedirs(destino, exist_ok=True)

for nombre in os.listdir(origen):
    ruta_completa = os.path.join(origen, nombre)
    destino_carpeta = os.path.join(destino, nombre)
    if os.path.isdir(ruta_completa) and nombre != 'data':
        # Copiar la carpeta solo si no existe en destino
        if not os.path.exists(destino_carpeta):
            shutil.copytree(ruta_completa, destino_carpeta)
            print(f'Copiada: {nombre} -> {destino}')
        else:
            print(f'Ya existe: {destino_carpeta}')