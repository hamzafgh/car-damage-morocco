"""Train the CNN+LSTM captioner on auto-generated French captions.

The training set is built from the pipeline's own annotations:
  - run Stage 1 (parts) + Stage 2 (damage) over your damage image set
  - for each detected damage, crop the part region, compute (part, damage_type,
    area_ratio, position), then synthesize a caption with describe_damage.describe()
  - the (cropped image, caption) pair is the training example

This is distillation: the CNN+LSTM learns to map directly from image -> French
description, without needing any human-written captions. Once trained, the model
runs end-to-end on a fresh image without going through Stage 1/2.

Run this on Kaggle once Stage 1 (parts_seg/best.pt) is trained. Expected:
~30 min on T4 for 30 epochs over a few thousand cropped patches.

Usage:
  python train_caption_model.py \
      --images-dir /kaggle/input/damage-images \
      --parts-weights /kaggle/input/stage1/parts_seg_best.pt \
      --damage-weights /kaggle/input/stage2/damage_seg_best.pt \
      --out-dir /kaggle/working/captioner
"""
from __future__ import annotations
import argparse, json, re, os
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

from describe_damage import (
    Damage, describe, position_from_centroid, severity_from_area,
    PART_FR, DAMAGE_FR,
)
from caption_model import (
    IMG_SIZE, build_captioner, masked_sparse_ce, greedy_decode,
)

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"


# ----------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------
def tokenize_fr(s: str) -> list[str]:
    # lowercase, keep accents, split on whitespace + punctuation as separate tokens
    s = s.lower()
    return re.findall(r"[a-zà-ÿ]+|\d+%?|[.,]", s)


def build_vocab(captions: list[str], min_freq: int = 1) -> list[str]:
    counter: Counter[str] = Counter()
    for c in captions:
        counter.update(tokenize_fr(c))
    vocab = [PAD, START, END, UNK]
    vocab += [w for w, n in counter.most_common() if n >= min_freq]
    return vocab


def encode(caption: str, stoi: dict[str, int], max_len: int) -> np.ndarray:
    ids = [stoi[START]]
    for w in tokenize_fr(caption):
        ids.append(stoi.get(w, stoi[UNK]))
    ids.append(stoi[END])
    ids = ids[:max_len]
    ids += [stoi[PAD]] * (max_len - len(ids))
    return np.asarray(ids, dtype=np.int32)


# ----------------------------------------------------------------------
# Build training pairs from the pipeline's own outputs
# ----------------------------------------------------------------------
def build_training_pairs(
    images_dir: Path,
    parts_weights: Path,
    damage_weights: Path,
    out_dir: Path,
    iomin_thr: float = 0.35,
    pad_px: int = 20,
) -> tuple[list[Path], list[str]]:
    """Run Stage 1 + Stage 2 on every image in images_dir. For each damage,
    crop the *part* region (with padding) and synthesize a French caption.
    Saves cropped images to out_dir/crops/ and returns (image_paths, captions)."""
    from ultralytics import YOLO
    import cv2

    parts = YOLO(str(parts_weights))
    damage = YOLO(str(damage_weights))
    parts_names = parts.model.names if hasattr(parts.model, "names") else parts.names
    damage_names = damage.model.names if hasattr(damage.model, "names") else damage.names

    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    captions:    list[str] = []
    crop_idx = 0

    for img_path in sorted(images_dir.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        p_res = parts.predict(img, verbose=False)[0]
        d_res = damage.predict(img, verbose=False)[0]
        if p_res.masks is None or d_res.masks is None:
            continue

        p_masks = p_res.masks.data.cpu().numpy().astype(bool)   # (Np, h, w)
        p_cls   = p_res.boxes.cls.cpu().numpy().astype(int)
        d_masks = d_res.masks.data.cpu().numpy().astype(bool)   # (Nd, h, w)
        d_cls   = d_res.boxes.cls.cpu().numpy().astype(int)

        # Resize masks to image size if needed
        def _resize(m: np.ndarray) -> np.ndarray:
            if m.shape[1:] != (H, W):
                return np.stack([cv2.resize(mi.astype(np.uint8), (W, H)).astype(bool) for mi in m])
            return m
        p_masks = _resize(p_masks)
        d_masks = _resize(d_masks)

        # Fuse with IoMin: assign each damage to the part with highest IoMin
        for di in range(len(d_masks)):
            dm = d_masks[di]
            d_area = dm.sum()
            if d_area == 0:
                continue
            best_pi, best_iomin = -1, 0.0
            for pi in range(len(p_masks)):
                inter = np.logical_and(dm, p_masks[pi]).sum()
                if inter == 0:
                    continue
                p_area = p_masks[pi].sum()
                iomin = inter / max(1, min(d_area, p_area))
                if iomin > best_iomin:
                    best_iomin, best_pi = iomin, pi
            if best_pi < 0 or best_iomin < iomin_thr:
                continue

            part_name = parts_names[int(p_cls[best_pi])]
            dmg_name  = damage_names[int(d_cls[di])]
            if part_name not in PART_FR or dmg_name not in DAMAGE_FR:
                continue

            pm = p_masks[best_pi]
            ys, xs = np.where(pm)
            x1, x2 = max(0, xs.min() - pad_px), min(W, xs.max() + pad_px)
            y1, y2 = max(0, ys.min() - pad_px), min(H, ys.max() + pad_px)
            crop = img[y1:y2, x1:x2]

            # Centroid of damage inside the part bbox
            dys, dxs = np.where(dm)
            cy, cx = float(dys.mean()), float(dxs.mean())
            position = position_from_centroid(cx, cy, x1, y1, x2, y2)
            area_ratio = float(np.logical_and(dm, pm).sum() / max(1, pm.sum()))

            d_obj = Damage(part=part_name, damage_type=dmg_name,
                           area_ratio=area_ratio, position=position)
            caption = describe(d_obj)

            crop_path = crops_dir / f"crop_{crop_idx:06d}_{part_name}_{dmg_name}.jpg"
            cv2.imwrite(str(crop_path), crop)
            image_paths.append(crop_path)
            captions.append(caption)
            crop_idx += 1

    print(f"Built {len(image_paths)} (image, caption) pairs.")
    return image_paths, captions


# ----------------------------------------------------------------------
# tf.data dataset
# ----------------------------------------------------------------------
def make_dataset(
    image_paths: list[Path],
    captions: list[str],
    stoi: dict[str, int],
    max_len: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    paths = tf.constant([str(p) for p in image_paths])
    encoded = np.stack([encode(c, stoi, max_len) for c in captions])  # (N, max_len)
    tokens_in  = encoded[:, :-1]
    tokens_out = encoded[:, 1:]

    def _load(path, tin, tout):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.uint8)
        return (img, tin), tout

    ds = tf.data.Dataset.from_tensor_slices((paths, tokens_in, tokens_out))
    if shuffle:
        ds = ds.shuffle(min(8192, len(image_paths)), reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir",      type=Path, required=True)
    ap.add_argument("--parts-weights",   type=Path, required=True)
    ap.add_argument("--damage-weights",  type=Path, required=True)
    ap.add_argument("--out-dir",         type=Path, default=Path("captioner"))
    ap.add_argument("--epochs",          type=int, default=30)
    ap.add_argument("--batch-size",      type=int, default=32)
    ap.add_argument("--max-len",         type=int, default=40)
    ap.add_argument("--val-split",       type=float, default=0.1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths, caps = build_training_pairs(
        args.images_dir, args.parts_weights, args.damage_weights, args.out_dir
    )
    if not paths:
        raise RuntimeError("No training pairs were generated — check inputs.")

    vocab = build_vocab(caps, min_freq=1)
    stoi = {w: i for i, w in enumerate(vocab)}
    print("Vocab size:", len(vocab))
    (args.out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    # Train/val split
    n = len(paths)
    idx = np.arange(n); np.random.seed(42); np.random.shuffle(idx)
    n_val = max(1, int(n * args.val_split))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    tr_paths, tr_caps = [paths[i] for i in tr_idx], [caps[i] for i in tr_idx]
    va_paths, va_caps = [paths[i] for i in val_idx], [caps[i] for i in val_idx]

    train_ds = make_dataset(tr_paths, tr_caps, stoi, args.max_len, args.batch_size, shuffle=True)
    val_ds   = make_dataset(va_paths, va_caps, stoi, args.max_len, args.batch_size, shuffle=False)

    model = build_captioner(vocab_size=len(vocab))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=masked_sparse_ce(pad_id=stoi[PAD]),
    )
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        str(args.out_dir / "best.keras"),
        monitor="val_loss", save_best_only=True, mode="min", verbose=1,
    )
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[ckpt_cb])

    # Quick sanity decode on the first validation example
    sample_img = tf.io.decode_image(tf.io.read_file(str(va_paths[0])), channels=3, expand_animations=False)
    pred = greedy_decode(model, sample_img, vocab, stoi[START], stoi[END], max_len=args.max_len)
    print("\nGold :", va_caps[0])
    print("Pred :", pred)


if __name__ == "__main__":
    main()
