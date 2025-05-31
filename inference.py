import os
import pandas as pd
from PIL import Image

YOLO_LABELS_DIR = "runs/predict_target/target_infer_final/labels"
IMAGES_DIR = "data2/test/images"
EVAL_PATH = "eval.csv"
OUTPUT_CSV = "submission.csv"

# class_id → cat_id
cat_ids = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11,
           12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
           22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
           32, 33, 34, 35, 36, 38, 39, 40, 41, 42,
           43, 44, 45, 46, 47, 48, 50, 51, 52, 53]
classid_to_catid = {i: cat_id for i, cat_id in enumerate(cat_ids)}

# Leer eval.csv como plantilla
df = pd.read_csv(EVAL_PATH)

results = []
for idx, row in df.iterrows():
    image_id = str(int(row["Id"]))  # sin ceros
    image_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
    label_path = os.path.join(YOLO_LABELS_DIR, f"{image_id}.txt")

    if not os.path.exists(label_path) or not os.path.exists(image_path):
        results.append("")
        continue

    # Leer tamaño de imagen
    img = Image.open(image_path)
    w_img, h_img = img.size

    preds = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5 + 17 * 3:
                continue  # ignora líneas incompletas

            try:
                class_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:5])
                x_min = round((x_c - w / 2) * w_img)
                y_min = round((y_c - h / 2) * h_img)
                w_abs = round(w * w_img)
                h_abs = round(h * h_img)

                # Convertir keypoints
                kpts_abs = []
                for i in range(0, 17 * 3, 3):
                    x_kpt = float(parts[5 + i]) * w_img
                    y_kpt = float(parts[5 + i + 1]) * h_img
                    v_kpt = float(parts[5 + i + 2])
                    kpts_abs.extend([round(x_kpt), round(y_kpt), v_kpt])

                cat_id = classid_to_catid.get(class_id, -1)
                if cat_id == -1:
                    continue

                full_line = [cat_id, x_min, y_min, w_abs, h_abs, 1.0] + kpts_abs
                preds.append(" ".join(map(str, full_line)))

            except Exception as e:
                print(f"Error procesando línea en {label_path}: {e}")
                continue


    results.append(";".join(preds))

# Guardar resultados
df["Predicted"] = results
df.to_csv(OUTPUT_CSV, index=False)
print(f"Guardado en {OUTPUT_CSV}")
