import kagglehub
import shutil
import zipfile
import os
import random
from PIL import Image, ImageEnhance

#---Download the dataset from Kaggle ----
path = kagglehub.dataset_download("danielrey96/colombian-sign-language-lsc-alphabet")

DATA_DIR = '../data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

todas_path = os.path.join(path, "Todas")

for item in os.listdir(todas_path):
    origen = os.path.join(todas_path, item)
    destino_final = os.path.join(DATA_DIR, item)
    shutil.move(origen, destino_final)

os.rmdir(todas_path)

#--- extraer los datos de letras faltantes ----
zip_path = '../data_extra/letras_faltantes.zip'  

extract_dir = os.path.dirname(zip_path) or '.'


with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Descomprimido en: {extract_dir}")


main_dir = '../data_extra/letras_faltantes'

##--- Aumentar las imágenes para cada letra ----
def augment_image(img):
    # Cambios aleatorios de brillo y contraste
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.8, 1.2)  # Cambios suaves
        img = enhancer.enhance(factor)
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
    return img

for letra in map(str, range(10)):
    subcarpeta = os.path.join(main_dir, letra)
    if not os.path.isdir(subcarpeta):
        continue
    imagenes = [f for f in os.listdir(subcarpeta) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n_actual = len(imagenes)
    n_faltan = 250 - n_actual
    print(f"Letra {letra}: {n_actual} imágenes, generando {n_faltan} nuevas...")

    if n_faltan > 0:
        for i in range(n_faltan):
            img_name = random.choice(imagenes)
            img_path = os.path.join(subcarpeta, img_name)
            img = Image.open(img_path)
            img_aug = augment_image(img)
            new_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
            img_aug.save(os.path.join(subcarpeta, new_name))