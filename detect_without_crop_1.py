import cv2
import easyocr
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ─────────── настройки
IMAGE_PATH         = Path(r"D:/IAMTRYING2/Pipeline/ResizedPhotos/02.jpg")
OUT_VIS            = IMAGE_PATH.with_name("easyocr_result2.png")
OUT_JSON           = IMAGE_PATH.with_name("easyocr_result2.json")        # ← NEW
OUT_STRIP_RAW      = IMAGE_PATH.with_name("easyocr_side_strip.png")
OUT_STRIP_ROT      = IMAGE_PATH.with_name("easyocr_side_strip_rot.png")
RIGHT_STRIP_RATIO  = 0.15
NUM_LEN_ACCEPTED   = (8, 12)

# ─────────── 1. инициализация
reader_ru  = easyocr.Reader(['ru', 'en'], gpu=True,  verbose=False)
reader_num = easyocr.Reader(['en'],       gpu=True,  verbose=False,
                            recog_network='latin_g2')

img_bgr = cv2.imread(str(IMAGE_PATH))
if img_bgr is None:
    raise FileNotFoundError(IMAGE_PATH)
h_img, w_img = img_bgr.shape[:2]

# ─────────── вспомогательные контейнеры
boxes, texts, confs = [], [], []
records = []                                            # ← NEW

# ─────────── 2а. основной вызов: RU-ридер
for box, text, conf in reader_ru.readtext(img_bgr, paragraph=False, detail=1):
    xs = [p[0] for p in box]; ys = [p[1] for p in box]
    x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    w_box, h_box   = x1 - x0, y1 - y0
    if h_box * w_box < 0.003 * h_img * w_img:
        continue

    if h_box > w_box * 2:                        # подозрение на вертикальный номер
        crop = img_bgr[y0:y1, x0:x1]
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        num_res = reader_num.readtext(crop, detail=0, allowlist='0123456789')
        if not num_res:
            continue
        digits = ''.join(ch for ch in num_res[0] if ch.isdigit())
        if len(digits) != 10:                    # игнорируем неполные
            continue
        text = f"{digits[:2]} {digits[2:4]} {digits[4:]}"
        conf = 1.0
    else:
        if conf < 0.55:
            continue

    text = text.upper()                          # ← NEW: капс
    boxes.append((x0, y0, x1, y1))
    texts.append(f"{text} ({conf:.2f})")
    confs.append(conf)
    records.append({
        "text": text,
        "conf": float(conf),                  # ← python-float
        "box": [int(x0), int(y0), int(x1), int(y1)]   # ← python-int
    })
    print(text, conf)

# ─────────── 2b. отдельный поиск бокового номера справа
w_strip = int(w_img * RIGHT_STRIP_RATIO)
strip_bgr = img_bgr[:, w_img - w_strip:]
cv2.imwrite(str(OUT_STRIP_RAW), strip_bgr)

rot = cv2.rotate(strip_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imwrite(str(OUT_STRIP_ROT), rot)

strip_results = reader_num.readtext(rot, detail=1, paragraph=False,
                                    allowlist='0123456789')
strip_results.sort(key=lambda r: min(p[0] for p in r[0]))  # лево-→право

digits_all = ''.join(ch for _, t, _ in strip_results for ch in t if ch.isdigit())

if len(digits_all) == 10:
    digits_fmt = f"{digits_all[:2]} {digits_all[2:4]} {digits_all[4:]}"
    xs = [p[0] for r in strip_results for p in r[0]]
    ys = [p[1] for r in strip_results for p in r[0]]

    # координаты в глобальную систему
    pts_glob = [(w_img - w_strip + (w_strip - 1 - y), x) for x, y in zip(xs, ys)]
    x0, y0 = min(p[0] for p in pts_glob), min(p[1] for p in pts_glob)
    x1, y1 = max(p[0] for p in pts_glob), max(p[1] for p in pts_glob)

    boxes.append((x0, y0, x1, y1))
    texts.append(f"{digits_fmt} (1.00)")
    confs.append(1.0)
    records.append({
        "text": text,
        "conf": float(conf),                  # ← python-float
        "box": [int(x0), int(y0), int(x1), int(y1)]   # ← python-int
    })
    print('side strip:', digits_fmt, 1.0)
else:
    print("В кропе не набралось 10 цифр →", digits_all)

# ─────────── 3. визуализация
if not boxes:
    print("Ничего не найдено — проверьте путь к файлу и язык GPU/CPU")
    quit()

img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img_pil)
try:
    font = ImageFont.truetype("arial.ttf", 18)
except OSError:
    font = ImageFont.load_default()

for (x0, y0, x1, y1), t in zip(boxes, texts):
    draw.rectangle([x0, y0, x1, y1], outline="lime", width=2)
    draw.text((x0, max(0, y0 - 20)), t, fill="red", font=font)

img_pil.save(OUT_VIS)
print("✔ Визуализация сохранена →", OUT_VIS)

# ─────────── 4. экспорт JSON
with OUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("✔ JSON-файл сохранён →", OUT_JSON)
