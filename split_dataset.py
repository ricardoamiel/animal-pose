import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# Rutas (ajusta según tu máquina)
DATA_DIR = "data"       # Carpeta donde están TODAS las imágenes y los CSV
DATA2_DIR = "data2"     # Carpeta de salida para estructura organizada
TRAINING_CSV = os.path.join("training.csv")
EVAL_CSV = os.path.join("eval.csv")

# Parámetros
SEED = 42
SPLIT_RATIO = 0.8

# Crear subcarpetas
subdirs = [
    "train/images", "train/labels_original",
    "val/images", "val/labels_original",
    "test/images", "test/labels_original"
]
for subdir in subdirs:
    os.makedirs(os.path.join(DATA2_DIR, subdir), exist_ok=True)

# Cargar CSVs
training_df = pd.read_csv(TRAINING_CSV)
eval_df = pd.read_csv(EVAL_CSV)
eval_ids = set(eval_df["Id"].dropna().astype(int))

# Dividir data
in_eval_df = training_df[training_df["Id"].isin(eval_ids)]
not_in_eval_df = training_df[~training_df["Id"].isin(eval_ids)]
remaining_ids = not_in_eval_df["Id"].unique()
train_ids, val_ids = train_test_split(remaining_ids, train_size=SPLIT_RATIO, random_state=SEED)

# Funciones
def copy_image(image_id, dest_subdir):
    # Formato correcto del nombre de archivo con ceros al principio (12 dígitos)
    src_filename = f"{int(image_id):012d}.jpg"  # Formato: 000000000001.jpg
    src = os.path.join(DATA_DIR, src_filename)
    dst = os.path.join(DATA2_DIR, dest_subdir, "images", f"{int(image_id)}.jpg")
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copiando {src} a {dst}")
    else:
        print(f"No se encontró la imagen: {src}")

def save_label(image_id, content, dest_subdir):
    path = os.path.join(DATA2_DIR, dest_subdir, "labels", f"{image_id}.txt")
    with open(path, "w") as f:
        if content.strip():
            f.write(content.strip() + "\n")

# Test (eval.csv)
print("\nProcesando imágenes de test...")
for image_id in eval_ids:
    copy_image(image_id, "test")
    rows = in_eval_df[in_eval_df["Id"] == image_id]
    label = "\n".join(rows["Predicted"].values) if not rows.empty else ""
    save_label(image_id, label, "test")

# Train
print("\nProcesando imágenes de train...")
for image_id in train_ids:
    copy_image(image_id, "train")
    rows = not_in_eval_df[not_in_eval_df["Id"] == image_id]
    label = "\n".join(rows["Predicted"].values)
    save_label(image_id, label, "train")

# Val
print("\nProcesando imágenes de validación...")
for image_id in val_ids:
    copy_image(image_id, "val")
    rows = not_in_eval_df[not_in_eval_df["Id"] == image_id]
    label = "\n".join(rows["Predicted"].values)
    save_label(image_id, label, "val")

print("\nProceso completado!")
