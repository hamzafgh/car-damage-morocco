"""Stage 0 — car-model classifier wrapper.

Wraps the EfficientNetB0 model trained in notebooks/stage0_car_classifier.ipynb.
Preprocessing matches the notebook exactly:
  - resize to 224x224
  - uint8 in [0,255]  (EfficientNetB0 has built-in preprocessing in tf.keras)
  - softmax over 20 alphabetically-sorted class names from stage0_classes.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

INPUT_SIZE = (224, 224)


class CarClassifier:
    def __init__(self, weights_path: str | Path, classes_json: str | Path):
        import tensorflow as tf
        self.tf = tf
        self.model = tf.keras.models.load_model(str(weights_path))
        info = json.loads(Path(classes_json).read_text(encoding="utf-8"))
        self.class_names: list[str] = info["class_names"]
        self.display_names: dict[str, str] = info.get("display_names", {})
        assert len(self.class_names) == self.model.output_shape[-1], (
            f"Class count mismatch: {len(self.class_names)} names vs "
            f"{self.model.output_shape[-1]} model outputs"
        )

    def preprocess(self, image_bgr_or_rgb: np.ndarray) -> np.ndarray:
        """Convert an HxWx3 image (any common range) to a (1, 224, 224, 3) uint8 tensor."""
        import cv2
        img = image_bgr_or_rgb
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape {img.shape}")
        img = cv2.resize(img, INPUT_SIZE[::-1])  # (W, H) for cv2
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return np.expand_dims(img, 0)

    def predict(self, image: np.ndarray, top_k: int = 3) -> dict:
        """Returns {'label', 'display_label', 'confidence', 'topk': [(label, conf), ...]}."""
        x = self.preprocess(image)
        probs = self.model.predict(x, verbose=0)[0]                     # (20,)
        order = np.argsort(probs)[::-1]
        label = self.class_names[int(order[0])]
        topk = [(self.class_names[int(i)], float(probs[i])) for i in order[:top_k]]
        return {
            "label":         label,
            "display_label": self.display_names.get(label, label.replace("_", " ")),
            "confidence":    float(probs[order[0]]),
            "topk":          topk,
        }
