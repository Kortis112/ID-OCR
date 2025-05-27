# -*- coding: utf-8 -*-
"""
Gradio-интерфейс для пайплайна "YOLO-кроп  ▶  ресайз  ▶  EasyOCR".

⮕ Пользователь загружает фотографию.
⮕ YOLO (маска/бокс) вырезает паспорт.
⮕ Изображение приводится к 1519×1069.
⮕ EasyOCR выделяет ключевые поля, рисует боксы, распознаёт текст.
⮕ Возвращаем картинку + JSON со строками и confidence.

***
Перед запуском убедитесь, что:
  • установлены ultralytics, easyocr, gradio, pillow, opencv-python-headless
  • путь к YOLO-весам (YOLO_WEIGHTS) корректен
  • для работы на CPU замените gpu=True → False
"""

import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import easyocr

# ─────────────────────────── глобальные константы
YOLO_WEIGHTS = r"D:/IAMTRYING2/Pipeline/yolo/best.pt"  # ← поправьте путь при необходимости
TARGET_WIDTH, TARGET_HEIGHT = 1519, 1069               # resize.py
RIGHT_STRIP_RATIO = 0.15                               # detect_without_crop_1.py

# ─────────────────────────── модель и OCR-ридеры загружаем один раз
print("[INIT] loading YOLO weights …")
yolo_model = YOLO(YOLO_WEIGHTS)
print("[INIT] loading EasyOCR readers …")
reader_ru = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
reader_num = easyocr.Reader(["en"], gpu=False, verbose=False, recog_network="latin_g2")
print("[INIT] ready.")

# ─────────────────────────── вспомогательные функции для YOLO-кропа

def order_points(pts: np.ndarray) -> np.ndarray:
    """Сортировка углов: TL, TR, BR, BL (по часовой)."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extract_by_mask(img: np.ndarray, res, pad_ratio: float = 0.0) -> np.ndarray:
    """Warp-кроп по маске или выброс исключения."""
    H, W = img.shape[:2]
    poly = (res.masks.xyn[0] * np.array([W, H])).astype(np.float32)  # N×2
    if len(poly) > 4:
        poly = cv2.approxPolyDP(poly, 0.01 * cv2.arcLength(poly, True), True)
    pts = order_points(poly.reshape(-1, 2))

    # целевые размеры
    wA = np.linalg.norm(pts[1] - pts[0])
    wB = np.linalg.norm(pts[2] - pts[3])
    hA = np.linalg.norm(pts[3] - pts[0])
    hB = np.linalg.norm(pts[2] - pts[1])
    Wt, Ht = int(max(wA, wB)), int(max(hA, hB))

    pad = int(pad_ratio * max(Wt, Ht))
    dst = np.array(
        [[pad, pad], [Wt - 1 + pad, pad], [Wt - 1 + pad, Ht - 1 + pad], [pad, Ht - 1 + pad]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (Wt + 2 * pad, Ht + 2 * pad), borderValue=(255, 255, 255))


# ─────────────────────────── основной обработчик одного изображения

def process(image: np.ndarray):
    """Основная функция Gradio: принимает RGB-numpy, возвращает (annot_img, json-dict)."""
    if image is None:
        return None, {}

    # 0) cv2 ожидает BGR
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 1) YOLO: находим паспорт (маска - приоритет, иначе бокс)
    res = yolo_model(img_bgr)[0]
    crop_bgr = None
    if res.masks is not None and len(res.masks.data):
        try:
            crop_bgr = extract_by_mask(img_bgr, res, pad_ratio=0.02)
        except Exception:
            crop_bgr = None
    if crop_bgr is None:  # fallback bbox
        if res.boxes.shape[0] == 0:
            return None, {"error": "Паспорт не найден"}
        # берём самый крупный бокс
        boxes = res.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        x1, y1, x2, y2 = boxes[areas.argmax()].astype(int)
        crop_bgr = img_bgr[y1:y2, x1:x2]

    # 2) ресайз до константного разрешения (как resize.py)
    crop_bgr = cv2.resize(crop_bgr, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

    # 3) EasyOCR детект + визуализация
    annotated_pil, records = run_easyocr_pipeline(crop_bgr)

    # 4) выводы
    annotated_rgb = np.array(annotated_pil)
    return annotated_rgb, records


# ─────────────────────────── EasyOCR-часть (адаптирована из detect_without_crop_1.py)

def run_easyocr_pipeline(img_bgr: np.ndarray):
    """Запускаем два ридера, возвращаем PIL-картинку с боксами + list(dict)."""
    h_img, w_img = img_bgr.shape[:2]

    boxes, texts, records = [], [], []

    # ——— 1. основной читатель (рус/англ)
    results = reader_ru.readtext(img_bgr, paragraph=False, detail=1)
    for box, text, conf in results:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        w_box, h_box = x1 - x0, y1 - y0
        if h_box * w_box < 0.003 * h_img * w_img:
            continue

        if h_box > w_box * 2:  # вертикальный номер
            crop = img_bgr[y0:y1, x0:x1]
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            num_res = reader_num.readtext(crop, detail=0, allowlist="0123456789")
            if not num_res:
                continue
            digits = "".join(ch for ch in num_res[0] if ch.isdigit())
            if len(digits) != 10:
                continue
            text = f"{digits[:2]} {digits[2:4]} {digits[4:]}"
            conf = 1.0
        else:
            if conf < 0.55:
                continue

        text = text.upper()
        boxes.append((x0, y0, x1, y1))
        texts.append(f"{text} ({conf:.2f})")
        records.append({"text": text, "conf": float(conf), "box": [int(x0), int(y0), int(x1), int(y1)]})

    # ——— 2. правая полоса
    w_strip = int(w_img * RIGHT_STRIP_RATIO)
    strip_bgr = img_bgr[:, w_img - w_strip :]
    rot = cv2.rotate(strip_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    strip_results = reader_num.readtext(rot, detail=1, paragraph=False, allowlist="0123456789")
    strip_results.sort(key=lambda r: min(p[0] for p in r[0]))
    digits_all = "".join(ch for _, t, _ in strip_results for ch in t if ch.isdigit())
    if len(digits_all) == 10:
        digits_fmt = f"{digits_all[:2]} {digits_all[2:4]} {digits_all[4:]}"
        xs = [p[0] for r in strip_results for p in r[0]]
        ys = [p[1] for r in strip_results for p in r[0]]
        pts_glob = [(w_img - w_strip + (w_strip - 1 - y), x) for x, y in zip(xs, ys)]
        x0, y0 = min(p[0] for p in pts_glob), min(p[1] for p in pts_glob)
        x1, y1 = max(p[0] for p in pts_glob), max(p[1] for p in pts_glob)
        boxes.append((x0, y0, x1, y1))
        texts.append(f"{digits_fmt} (1.00)")
        records.append({"text": digits_fmt, "conf": 1.0, "box": [int(x0), int(y0), int(x1), int(y1)]})

    # ——— 3. визуализация
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for (x0, y0, x1, y1), t in zip(boxes, texts):
        draw.rectangle([x0, y0, x1, y1], outline="lime", width=2)
        draw.text((x0, max(0, y0 - 20)), t, fill="red", font=font)

    return img_pil, records


# ─────────────────────────── запускаем Gradio
with gr.Blocks(title="Passport OCR demo") as demo:
    gr.Markdown("## Паспорт-OCR\nЗагрузите фото, нажмите **Process**, получите распознанный текст и изображение с подсветкой полей.")
    with gr.Row():
        inp = gr.Image(label="Фото паспорта", type="numpy", height=800)
    btn = gr.Button("Process")
    with gr.Row():
        out_img = gr.Image(label="Боксы EasyOCR", height=800)
    out_json = gr.JSON(label="Распознанные строки")

    btn.click(process, inputs=inp, outputs=[out_img, out_json])

if __name__ == "__main__":
    demo.launch()
