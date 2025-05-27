import os
import cv2
from glob import glob
from tqdm import tqdm

# === Настройки ===
INPUT_DIR    = r"D:/IAMTRYING2/Pipeline/PassportCrops"  # ← ваша папка с исходными фото
OUTPUT_DIR   = r"D:/IAMTRYING2/Pipeline/ResizedPhotos"   # ← куда складывать растянутые фото
TARGET_WIDTH  = 1519
TARGET_HEIGHT = 1069
# =================

os.makedirs(OUTPUT_DIR, exist_ok=True)

paths = glob(os.path.join(INPUT_DIR, "*.*"))
for path in tqdm(paths, desc="Resizing images"):
    img = cv2.imread(path)
    if img is None:
        print(f"Не удалось прочитать {path}, пропускаю")
        continue

    # Растягиваем до нужного разрешения
    resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

    # Сохраняем результат с тем же именем файла
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
    cv2.imwrite(out_path, resized)

print("Готово! Растянутые фото в:", OUTPUT_DIR)
