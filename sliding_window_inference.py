"""
Sliding window inference with vote-based fusion.

Paper: "Smart Integration of Sliding Window and Vote-Based Fusion:
        Advancing UAV-Based Instance Segmentation with YOLOv8"
DOI:   10.1016/j.rsase.2026.101994

Patch size:  283x283 px  (YOLO auto-resizes to 288x288 internally)
Stride:      85 px  (70% overlap: 283 * 0.3 ≈ 85)
Fusion:      vote count accumulation + argmax per pixel

Author: Ilias Jennaoui — G2E Lab, Sultan Moulay Slimane University, Morocco
"""

import os
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ---- paths ------------------------------------------------------------------

MODEL_PATH = 'weights/best.pt'
OUTPUT_DIR = 'outputs/predictions'
OUTPUT_DPI = 300

IMAGE_MASK_PAIRS = [
    ('data/images/test_01.JPG', 'data/masks/test_01.png'),
    ('data/images/test_02.JPG', 'data/masks/test_02.png'),
    ('data/images/test_03.JPG', 'data/masks/test_03.png'),
]

# ---- patch / stride settings ------------------------------------------------
# Training patches: 280x280 px → YOLO auto-resizes to 288x288
# Inference window: 283x283 px → YOLO auto-resizes to 288x288 (same scale)
# Stride: 85 px = 283 * 0.3  →  70% overlap

WINDOW_SIZE = 283
STRIDE      = 85

# ---- class mapping ----------------------------------------------------------
# YOLO classes (0-indexed):  0=Lentisque  1=Chene-vert  2=Thuya  3=Oxycedre
# GT mask encoding (1-indexed): 0=Background 1=Lentisque 2=Chene-vert 3=Thuya 4=Oxycedre

CLASS_REMAP = {0: 1, 1: 2, 2: 3, 3: 4}   # YOLO class -> GT class

CLASS_NAMES = ['Background', 'Lentisque', 'Chene-vert', 'Thuya', 'Oxycedre']

# Colors match published figures
COLORS = [
    [0,   0,   0  ],   # Background - black
    [255, 0,   0  ],   # Lentisque  - red
    [0,   255, 0  ],   # Chene-vert - green
    [0,   0,   255],   # Thuya      - blue
    [255, 255, 0  ],   # Oxycedre   - yellow
]

# Per-class confidence thresholds (tuned on validation set)
CONF_THRESHOLDS = {
    'Lentisque':  0.35,
    'Chene-vert': 0.60,
    'Thuya':      0.45,
    'Oxycedre':   0.70,
}

YOLO_TO_NAME = {0: 'Lentisque', 1: 'Chene-vert', 2: 'Thuya', 3: 'Oxycedre'}

BASE_CONF    = 0.25
MORPH_KERNEL = 7


# ---- functions --------------------------------------------------------------

def extract_patches(image, window, stride):
    """Sliding window extraction. Returns patches and (x, y) top-left coords."""
    H, W = image.shape[:2]
    patches, coords = [], []
    for y in range(0, H - window + 1, stride):
        for x in range(0, W - window + 1, stride):
            patches.append(image[y:y+window, x:x+window])
            coords.append((x, y))
    # right edge
    if (W - window) % stride != 0:
        x = W - window
        for y in range(0, H - window + 1, stride):
            patches.append(image[y:y+window, x:x+window])
            coords.append((x, y))
    # bottom edge
    if (H - window) % stride != 0:
        y = H - window
        for x in range(0, W - window + 1, stride):
            patches.append(image[y:y+window, x:x+window])
            coords.append((x, y))
    # bottom-right corner
    if (H - window) % stride != 0 and (W - window) % stride != 0:
        patches.append(image[H-window:H, W-window:W])
        coords.append((W - window, H - window))
    return patches, coords


def vote_fusion(results, coords, img_shape, window):
    """
    Vote-based fusion as described in the paper (Equations 3-5):
      V(x,y,c) = sum of binary mask votes for class c
      C(x,y)   = max confidence (for visualization only, not classification)
      final(x,y) = argmax_c V(x,y,c)
    """
    H, W = img_shape[:2]
    n    = len(CLASS_NAMES)
    votes   = np.zeros((H, W, n), dtype=np.int32)
    conf_map = np.zeros((H, W),   dtype=np.float32)

    for result, (x0, y0) in zip(results, coords):
        x1 = min(x0 + window, W)
        y1 = min(y0 + window, H)
        ph, pw = y1 - y0, x1 - x0

        if result.masks is None or result.boxes is None:
            continue

        masks   = result.masks.data.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs   = result.boxes.conf.cpu().numpy()

        for mask, cid, conf in zip(masks, cls_ids, confs):
            name = YOLO_TO_NAME.get(cid, '')
            if conf < CONF_THRESHOLDS.get(name, BASE_CONF):
                continue

            if mask.shape != (window, window):
                mask = cv2.resize(mask.astype(np.float32), (window, window),
                                  interpolation=cv2.INTER_LINEAR)

            binary   = (mask[:ph, :pw] > 0.3).astype(np.int32)
            gt_cls   = CLASS_REMAP.get(cid, 0)
            votes[y0:y1, x0:x1, gt_cls] += binary

            # confidence map — max across detections (visualization only)
            conf_map[y0:y1, x0:x1] = np.maximum(
                conf_map[y0:y1, x0:x1], conf * binary
            )

    final = np.argmax(votes, axis=2).astype(np.uint8)
    return final, conf_map


def postprocess(mask):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask


def compute_metrics(pred, gt):
    metrics = {}
    ious, f1s = [], []
    for c, name in enumerate(CLASS_NAMES):
        pc = pred == c
        gc = gt   == c
        tp = int(np.sum( pc &  gc))
        fp = int(np.sum( pc & ~gc))
        fn = int(np.sum(~pc &  gc))
        iou = tp / (tp + fp + fn + 1e-10)
        f1  = 2*tp / (2*tp + fp + fn + 1e-10)
        metrics[name] = {'iou': iou, 'f1': f1}
        ious.append(iou)
        f1s.append(f1)
    metrics['mIoU'] = float(np.mean(ious[1:]))
    metrics['mF1']  = float(np.mean(f1s[1:]))
    return metrics


def save_figure(image, gt, pred, miou, path):
    cmap = ListedColormap([np.array(c) / 255.0 for c in COLORS])
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image');      axes[0].axis('off')
    axes[1].imshow(gt,   cmap=cmap, vmin=0, vmax=len(CLASS_NAMES)-1)
    axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=len(CLASS_NAMES)-1)
    axes[2].set_title(f'Prediction  mIoU={miou:.4f}'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close()


# ---- main -------------------------------------------------------------------

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)
    print(f"Model: {MODEL_PATH}")
    print(f"Window: {WINDOW_SIZE}x{WINDOW_SIZE}  Stride: {STRIDE}  "
          f"Overlap: {100*(1 - STRIDE/WINDOW_SIZE):.0f}%")

    all_miou = []

    for img_path, mask_path in IMAGE_MASK_PAIRS:
        print(f"\n{img_path}")
        image = cv2.imread(img_path)
        gt    = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"  cannot read {img_path}, skipping")
            continue

        H, W = image.shape[:2]
        patches, coords = extract_patches(image, WINDOW_SIZE, STRIDE)
        print(f"  {W}x{H}  ->  {len(patches)} patches")

        results = [model(p, conf=BASE_CONF, iou=0.5, verbose=False)[0]
                   for p in patches]

        pred, _ = vote_fusion(results, coords, image.shape, WINDOW_SIZE)
        pred    = postprocess(pred)

        m = compute_metrics(pred, gt)
        all_miou.append(m['mIoU'])

        print(f"  mIoU={m['mIoU']:.4f}  mF1={m['mF1']:.4f}")
        for name in CLASS_NAMES[1:]:
            print(f"    {name:<15s}  IoU={m[name]['iou']:.4f}  F1={m[name]['f1']:.4f}")

        stem = os.path.splitext(os.path.basename(img_path))[0]
        save_figure(image, gt, pred, m['mIoU'],
                    os.path.join(OUTPUT_DIR, f"{stem}_pred.png"))

    if all_miou:
        print(f"\nOverall mIoU: {np.mean(all_miou):.4f}")


if __name__ == '__main__':
    run()
