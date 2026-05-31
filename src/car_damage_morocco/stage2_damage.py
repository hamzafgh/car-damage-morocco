"""Stage 2 — damage segmentation wrapper (YOLOv8s-seg, 4 classes)."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DamageDetection:
    class_id:   int
    class_name: str            # 'dent' | 'scratch' | 'glass' | 'broken_part'
    confidence: float
    bbox:       tuple[int, int, int, int]
    mask:       np.ndarray     # bool array, shape (H, W)


class DamageSegmenter:
    def __init__(self, weights_path: str | Path, classes_json: str | Path):
        from ultralytics import YOLO
        self.model = YOLO(str(weights_path))
        info = json.loads(Path(classes_json).read_text(encoding="utf-8"))
        self.class_names: list[str] = info["class_names"]

    def predict(
        self, image_bgr: np.ndarray, conf: float = 0.25, imgsz: int = 640
    ) -> list[DamageDetection]:
        import cv2
        H, W = image_bgr.shape[:2]
        res = self.model.predict(image_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
        if res.masks is None or len(res.masks) == 0:
            return []

        masks = res.masks.data.cpu().numpy().astype(bool)
        cls   = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy().astype(float)
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)

        out: list[DamageDetection] = []
        for i in range(len(masks)):
            m = masks[i]
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            cid = int(cls[i])
            out.append(DamageDetection(
                class_id=cid,
                class_name=self.class_names[cid] if 0 <= cid < len(self.class_names) else f"class_{cid}",
                confidence=float(confs[i]),
                bbox=tuple(int(v) for v in boxes[i]),  # type: ignore[arg-type]
                mask=m,
            ))
        return out
