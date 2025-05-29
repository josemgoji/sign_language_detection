import os
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

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
            cv2.imwrite(output_path, cv2.cvtColor(recolored, cv2.COLOR_RGB2BGR))
            print(f'Guardada: {output_path}')