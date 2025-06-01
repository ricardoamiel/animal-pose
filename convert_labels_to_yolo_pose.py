import os
import glob
from PIL import Image

# Mapeo original de cat_id → class_id
cat_ids = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
           22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40,
           41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53]
catid_to_classid = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

# Rutas
DATA2_DIR = "data2"
#SUBSETS = ["train", "val", "test"]
SUBSETS = ["train", "test"]

def convert_label_format(label_path, image_path):
    try:
        with open(label_path, "r") as f:
            lines_raw = f.readlines()
        if not lines_raw:
            return []

        img = Image.open(image_path)
        img_w, img_h = img.size

        lines = []
        for raw_line in lines_raw:
            for instance in raw_line.strip().split(";"):
                parts = instance.strip().split()
                if len(parts) != 6 + 17 * 3:
                    continue

                cat_id = int(parts[0])
                if cat_id not in catid_to_classid:
                    continue

                class_id = catid_to_classid[cat_id]
                x_min, y_min, width, height = map(float, parts[1:5])
                x_center = (x_min + width / 2) / img_w
                y_center = (y_min + height / 2) / img_h
                w_norm = width / img_w
                h_norm = height / img_h

                keypoints_raw = parts[6:]
                keypoints = []
                for i in range(0, len(keypoints_raw), 3):
                    x = float(keypoints_raw[i])
                    y = float(keypoints_raw[i + 1])
                    v = float(keypoints_raw[i + 2])
                    if v == 0.0:
                        keypoints.extend([0.0, 0.0, 0])
                    else:
                        keypoints.extend([
                            round(x / img_w, 6),
                            round(y / img_h, 6),
                            int(2)
                        ])

                line = [str(class_id), f"{x_center:.6f}", f"{y_center:.6f}", f"{w_norm:.6f}", f"{h_norm:.6f}"] + \
                       [str(kp) for kp in keypoints]
                lines.append(" ".join(line))
        return lines
    except Exception as e:
        print(f"Error en {label_path}: {e}")
        return []


# Procesar cada subset
for subset in SUBSETS:
    input_label_dir = os.path.join(DATA2_DIR, subset, "labels_original")
    output_label_dir = os.path.join(DATA2_DIR, subset, "labels") # labels en formato yolo
    os.makedirs(output_label_dir, exist_ok=True)

    label_files = glob.glob(os.path.join(input_label_dir, "*.txt"))
    print(f"Convirtiendo {len(label_files)} archivos en {subset}...")

    for label_file in label_files:
        image_id = os.path.splitext(os.path.basename(label_file))[0]
        image_path = os.path.join(DATA2_DIR, subset, "images", f"{image_id}.jpg")
        new_lines = convert_label_format(label_file, image_path)

        # Guardar nuevo archivo en labels_yolo/
        output_path = os.path.join(output_label_dir, f"{image_id}.txt")
        with open(output_path, "w") as f:
            for line in new_lines:
                f.write(line + "\n")
