"""
Helpers for the marine debris app.

- detect_debris():   runs YOLO on the whole image AND on overlapping tiles,
                     then merges the results. Small items that vanish when a
                     big photo is shrunk to 640 px get found in the tiles.
- draw_detections(): draws boxes + labels on the image.
- make_ocean_background_b64(): paints a soft, lightly blurred sunset-over-sea
                     picture (no download needed) for the page background.

Only needs numpy and Pillow, both installed with Streamlit.
"""

import base64
import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# =========================================================
# DETECTION
# =========================================================

def _starts(length, tile, stride):
    """Tile start positions that cover `length`, last tile flush with the edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def nms(boxes, scores, iou_thr):
    """Plain numpy non-maximum suppression. Returns indices to keep."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if rest.size == 0:
            break

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]

    return keep


def _collect(result, dets, ox=0, oy=0):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
        dets.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy, float(s), int(c)])


def detect_debris(
    model,
    image,
    conf=0.10,
    deep_scan=True,
    merge_overlaps=True,
    progress_cb=None,
    class_map=None,
):
    """
    Returns a list of (x1, y1, x2, y2, score, class_id), best score first.

    deep_scan      : also scan overlapping tiles (slower, finds small items)
    merge_overlaps : keep only the best box when several boxes sit on the
                     same object, even if they disagree on the class
    progress_cb    : optional function(fraction_done, message)
    class_map      : optional {old_class_id: new_class_id}, used to fold several
                     classes into one group. Applied before boxes are merged,
                     so two boxes on one object that disagree only inside the
                     group (e.g. "plastic bag" vs "plastic waste") become one.
    """
    W, H = image.size
    dets = []

    def report(frac, msg):
        if progress_cb:
            progress_cb(min(max(frac, 0.0), 1.0), msg)

    # ---- pass 1: the whole image -------------------------------------
    if deep_scan:
        full_size = int(min(1280, max(960, math.ceil(max(W, H) / 32) * 32)))
    else:
        full_size = 640

    report(0.05, "Scanning the whole image...")
    res = model.predict(
        source=image, conf=conf, iou=0.6, imgsz=full_size, verbose=False
    )
    _collect(res[0], dets)

    # ---- pass 2: overlapping tiles -----------------------------------
    if deep_scan and max(W, H) > 800:
        tile = 640
        while True:
            stride = int(tile * 0.75)
            xs = _starts(W, tile, stride)
            ys = _starts(H, tile, stride)
            if len(xs) * len(ys) <= 25:
                break
            tile = int(tile * 1.2)  # too many tiles, use bigger ones

        tile_w, tile_h = min(tile, W), min(tile, H)
        tile_imgsz = int(math.ceil(max(tile_w, tile_h) / 32) * 32)

        origins = [(x, y) for y in ys for x in xs]
        total = len(origins)
        chunk = 4

        for start in range(0, total, chunk):
            batch = origins[start:start + chunk]
            crops = [
                image.crop((x, y, x + tile_w, y + tile_h)) for x, y in batch
            ]
            results = model.predict(
                source=crops, conf=conf, iou=0.6,
                imgsz=tile_imgsz, verbose=False
            )
            for (ox, oy), r in zip(batch, results):
                _collect(r, dets, ox, oy)

            report(
                0.15 + 0.80 * min(start + chunk, total) / total,
                f"Scanning close-up sections ({min(start + chunk, total)}/{total})...",
            )

    # ---- merge --------------------------------------------------------
    report(0.97, "Merging results...")
    if not dets:
        return []

    if class_map:
        for d in dets:
            d[5] = class_map.get(d[5], d[5])

    arr = np.array(dets, dtype=np.float64)
    boxes, scores, classes = arr[:, :4], arr[:, 4], arr[:, 5].astype(int)

    if merge_overlaps:
        keep = nms(boxes, scores, 0.5)
    else:
        keep = []
        for c in np.unique(classes):
            idx = np.where(classes == c)[0]
            kept = nms(boxes[idx], scores[idx], 0.5)
            keep.extend(int(idx[k]) for k in kept)

    out = [
        (float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2]),
         float(arr[i, 3]), float(arr[i, 4]), int(arr[i, 5]))
        for i in keep
    ]
    out.sort(key=lambda d: d[4], reverse=True)
    report(1.0, "Done")
    return out


# =========================================================
# DRAWING
# =========================================================

PALETTE = [
    (255, 112, 67), (255, 202, 40), (38, 214, 160), (66, 165, 245),
    (206, 120, 230), (255, 138, 180), (139, 220, 120), (77, 208, 225),
    (255, 171, 64), (160, 140, 255), (200, 230, 90), (255, 95, 95),
]


def _load_font(size):
    for name in ("segoeuib.ttf", "arialbd.ttf", "arial.ttf",
                 "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_detections(image, dets, names):
    """Return a copy of `image` with labelled boxes drawn on it."""
    img = image.convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)

    line = max(2, int(max(W, H) / 380))
    font = _load_font(max(12, int(max(W, H) / 70)))

    for x1, y1, x2, y2, score, cid in dets:
        color = PALETTE[cid % len(PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line)

        label = f"{names.get(cid, cid)} {score:.2f}"
        l, t, r, b = draw.textbbox((0, 0), label, font=font)
        tw, th = r - l, b - t
        pad = max(2, line)

        ly = y1 - th - 2 * pad
        if ly < 0:
            ly = y1
        lx = min(max(0, x1), max(0, W - tw - 2 * pad))

        draw.rectangle([lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad], fill=color)
        draw.text((lx + pad - l, ly + pad - t), label, fill=(15, 25, 45), font=font)

    return img


# =========================================================
# BACKGROUND: sunset over a clean blue sea, lightly blurred
# =========================================================

def _grad(t, stops):
    """Interpolate colour stops [(pos, (r,g,b)), ...] over array t (0..1)."""
    pos = [s[0] for s in stops]
    out = np.zeros(t.shape + (3,), np.float32)
    for ch in range(3):
        out[..., ch] = np.interp(t, pos, [s[1][ch] for s in stops])
    return out


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _noise(rng, shape_small, size_wh):
    """Smooth random field resized to (W, H), values 0..1."""
    small = rng.random(shape_small).astype(np.float32)
    im = Image.fromarray((small * 255).astype(np.uint8))
    im = im.resize(size_wh, Image.BICUBIC)
    return np.asarray(im, np.float32) / 255.0


def make_ocean_background(W=1600, H=900, seed=11):
    rng = np.random.default_rng(seed)
    hy = int(H * 0.47)                       # horizon line
    sx, sy = int(W * 0.63), hy - int(H * 0.018)   # sun position

    img = np.zeros((H, W, 3), np.float32)
    xs = np.arange(W, dtype=np.float32)[None, :]

    # ---------------- sky ----------------
    t = np.linspace(0, 1, hy, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    sky = _grad(t, [
        (0.00, (30, 52, 112)),
        (0.30, (84, 82, 150)),
        (0.55, (196, 110, 150)),
        (0.78, (248, 146, 112)),
        (0.92, (255, 190, 112)),
        (1.00, (255, 228, 168)),
    ])

    yy = np.arange(hy, dtype=np.float32)[:, None]
    d = np.sqrt((xs - sx) ** 2 + (yy - sy) ** 2)
    glow = np.exp(-(d / (0.20 * W)) ** 1.4)[..., None]
    sky += glow * np.array([120, 70, 30], np.float32) * 1.0
    core = _smooth(52, 40, d)[..., None]
    sky = sky * (1 - core) + np.array([255, 246, 220], np.float32) * core

    # soft clouds, lit from below by the sun
    fbm = (
        0.55 * _noise(rng, (5, 16), (W, hy))
        + 0.30 * _noise(rng, (10, 34), (W, hy))
        + 0.15 * _noise(rng, (20, 70), (W, hy))
    )
    band = _smooth(0.05, 0.30, t) * (1 - _smooth(0.70, 0.97, t))
    cloud = _smooth(0.50, 0.72, fbm) * band
    cloud_col = _grad(t, [
        (0.0, (110, 84, 150)), (0.5, (236, 140, 140)), (1.0, (255, 190, 130))
    ])
    sky = sky * (1 - 0.62 * cloud[..., None]) + cloud_col * 0.62 * cloud[..., None]
    img[:hy] = sky

    # ---------------- sea ----------------
    sh = H - hy
    s = np.linspace(0, 1, sh, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    sea = _grad(s, [
        (0.00, (252, 190, 130)),
        (0.05, (214, 140, 130)),
        (0.16, (96, 116, 168)),
        (0.35, (34, 104, 172)),
        (0.65, (12, 114, 176)),
        (1.00, (6, 86, 138)),
    ])

    # perspective ripples: tight near the horizon, wide up close
    z = 1.0 / (s + 0.035)
    drift = _noise(rng, (4, 10), (W, sh)) * 6.0
    ripple = np.sin(z * 2.1 + drift + xs * 0.004) * 0.5 \
        + np.sin(z * 4.3 + drift * 1.7 - xs * 0.002) * 0.3
    ripple *= (0.25 + 0.75 * s)
    streak = _noise(rng, (sh // 5 + 2, 18), (W, sh)) - 0.5
    sea += (ripple * 9 + streak * 26)[..., None]

    # sun path on the water
    width = 0.012 * W + 0.13 * W * s
    path = np.exp(-((xs - sx) / width) ** 2)
    sea += (path * (1 - 0.55 * s) * 0.55)[..., None] * np.array([150, 95, 40], np.float32)

    sparkle_src = rng.random((sh, W)).astype(np.float32)
    sp = Image.fromarray((sparkle_src * 255).astype(np.uint8)).resize(
        (W // 3, sh), Image.BILINEAR).resize((W, sh), Image.BICUBIC)
    sparkle = _smooth(0.60, 0.80, np.asarray(sp, np.float32) / 255.0)
    sea += (path * sparkle * (1 - 0.7 * s) * 1.0)[..., None] * np.array([255, 215, 150], np.float32)

    img[hy:] = sea

    # thin haze right at the horizon
    for k in range(-6, 7):
        y = hy + k
        if 0 <= y < H:
            a = (1 - abs(k) / 7.0) * 0.45
            img[y] = img[y] * (1 - a) + np.array([255, 214, 160], np.float32) * a

    # gentle vignette
    yy_all = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    xx_all = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    vig = 1 - 0.18 * np.clip((xx_all ** 2 + yy_all ** 2) / 2, 0, 1)
    img *= vig[..., None]

    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    return out.filter(ImageFilter.GaussianBlur(2.4))   # lightly blurred


def make_ocean_background_b64(W=1600, H=900):
    im = make_ocean_background(W, H)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")
