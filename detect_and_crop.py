import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO

# === CONFIGURE THESE PATHS / PARAMS ===
IMG_DIR      = r"D:/IAMTRYING2/Pipeline/dataset/photos"
YOLO_WEIGHTS = r"D:/IAMTRYING2/Pipeline/yolo/best.pt"
OUTPUT_DIR   = r"D:/IAMTRYING2/Pipeline/PassportCrops"
DEBUG_DIR    = r"D:/IAMTRYING2/Pipeline/PassportDebug"  # ← сюда сохраняем наложения масок
MAX_IMAGES   = 20
MARGIN       = 0.10
# =================================================

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def deskew_and_trim(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is not None:
        angles = [(l[0][1]*180/np.pi - 90) for l in lines]
        angle = float(np.median(angles))
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w,h), borderValue=(255,255,255))
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,binmask = cv2.threshold(gray2, 250, 255, cv2.THRESH_BINARY_INV)
    cnts,_ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x,y,ww,hh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        img = img[y:y+hh, x:x+ww]
    return img

import cv2
import numpy as np

# ---------- в самом верху  ----------
def extract_by_mask(img, res, debug_path=None, pad_ratio=0):
    """
    img  – исходное BGR
    res  – результат model(img)[0] (ultralytics.engine.results.Results)
    """
    H, W = img.shape[:2]

    # 1. Берём контур прямо из YOLO (уже нормированный)
    poly = (res.masks.xyn[0] * np.array([W, H])).astype(np.float32)

    # 2. Аппроксимируем до 4-угольника, если нужно
    if len(poly) > 4:
        poly = cv2.approxPolyDP(poly, 0.01 * cv2.arcLength(poly, True), True)
    pts = order_points(poly.reshape(-1, 2))

    # 3. Warp c небольшим паддингом
    wA = np.linalg.norm(pts[1] - pts[0])
    wB = np.linalg.norm(pts[2] - pts[3])
    hA = np.linalg.norm(pts[3] - pts[0])
    hB = np.linalg.norm(pts[2] - pts[1])
    Wt = int(max(wA, wB))
    Ht = int(max(hA, hB))

    pad = int(pad_ratio * max(Wt, Ht))
    dst = np.array([[pad, pad],
                    [Wt-1+pad, pad],
                    [Wt-1+pad, Ht-1+pad],
                    [pad, Ht-1+pad]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (Wt + 2*pad, Ht + 2*pad),
                                 borderValue=(255, 255, 255))

    if debug_path:
        dbg = img.copy()
        cv2.polylines(dbg, [pts.astype(int)], True, (0, 255, 0), 2)
        cv2.imwrite(debug_path, dbg)

    return warped


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    model = YOLO(YOLO_WEIGHTS)

    paths = glob(os.path.join(IMG_DIR, "*.*"))
    for i, path in enumerate(tqdm(paths, total=min(len(paths), MAX_IMAGES))):
        if i>=MAX_IMAGES: break
        img = cv2.imread(path)
        if img is None: continue

        res = model(img)[0]
        crop = None

        # 1) сегментация + debug
        if res.masks is not None and len(res.masks.data):
            debug_fname = os.path.join(DEBUG_DIR, f"{i:03d}_" + os.path.basename(path))
            try:
                crop = extract_by_mask(img, res, debug_path=debug_fname)
            except Exception as e:
                print(f"[{i}] Segmentation warp failed → fallback to box: {e}")

        # 2) fallback на box
        if crop is None:
            if res.boxes.shape[0]==0:
                print(f"[{i}] Паспорт не найден ни маской, ни боксом в {os.path.basename(path)}")
                continue
            boxes = res.boxes.xyxy.cpu().numpy()
            areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
            x1,y1,x2,y2 = boxes[areas.argmax()].astype(int)
            w_box,h_box = x2-x1,y2-y1
            dx,dy = int(w_box*MARGIN), int(h_box*MARGIN)
            x1m,y1m = max(0,x1-dx), max(0,y1-dy)
            x2m,y2m = min(img.shape[1],x2+dx), min(img.shape[0],y2+dy)
            crop = img[y1m:y2m, x1m:x2m]

        

        # 4) сохранение кропа
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
        cv2.imwrite(out_path, crop)

    print("Готово! Кропы в:", OUTPUT_DIR)
    print("Debug-наложения в:", DEBUG_DIR)

if __name__=="__main__":
    main()
