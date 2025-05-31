import os
import glob
from PIL import Image

YOLO_LABELS_DIR = "runs/predict_target/target_infer_final/labels"
IMAGES_DIR = "data2/test/images"
OUTPUT_CSV = "submission.csv"

# Mismo mapeo inverso
cat_ids = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11,
           12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
           22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
           32, 33, 34, 35, 36, 38, 39, 40, 41, 42,
           43, 44, 45, 46, 47, 48, 50, 51, 52, 53]

# class_id → cat_id
classid_to_catid = {i: cat_id for i, cat_id in enumerate(cat_ids)}

rows = []

for label_path in sorted(glob.glob(os.path.join(YOLO_LABELS_DIR, "*.txt"))):
    image_id = os.path.splitext(os.path.basename(label_path))[0]
    image_path = os.path.join(IMAGES_DIR, f"{int(image_id):012d}.jpg")
    img = Image.open(image_path)
    w_img, h_img = img.size

    preds = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 + 17 * 3:
                continue
            class_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])
            x_min = (x_c - w / 2) * w_img
            y_min = (y_c - h / 2) * h_img
            w_abs = w * w_img
            h_abs = h * h_img

            # Keypoints
            kpts = parts[5:]
            kpts_abs = []
            for i in range(0, len(kpts), 3):
                x_kpt = float(kpts[i]) * w_img
                y_kpt = float(kpts[i + 1]) * h_img
                v_kpt = float(kpts[i + 2])
                kpts_abs.extend([round(x_kpt), round(y_kpt), v_kpt])

            # Final line
            cat_id = classid_to_catid[class_id]
            full_line = [cat_id, round(x_min), round(y_min), round(w_abs), round(h_abs), 1.0] + kpts_abs
            preds.append(" ".join(map(str, full_line)))

    final_row = {
        "Id": int(image_id),
        "Predicted": ";".join(preds)
    }
    rows.append(final_row)

# Guardar CSV
import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Guardado en {OUTPUT_CSV}")
