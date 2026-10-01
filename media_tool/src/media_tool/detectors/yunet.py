from pathlib import Path

import cv2
import numpy as np

from ..faces import DetectedFace


class YuNetDetector:
    def __init__(self, path: Path, settings):
        self.max_side = settings.detection_max_side
        self.detector = cv2.FaceDetectorYN.create(
            str(path),
            "",
            (320, 320),
            settings.detection_score_threshold,
            0.3,
            5000,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        height, width = frame.shape[:2]
        ratio = min(1.0, self.max_side / max(height, width))
        size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        scaled = cv2.resize(frame, size) if ratio < 1 else frame
        self.detector.setInputSize(size)
        _, detections = self.detector.detect(scaled)
        if detections is None:
            return []
        result = []
        scale = np.array([size[0] / width, size[1] / height], np.float32)
        for detection in detections:
            x, y, w, h = detection[:4]
            bbox = np.array([x, y, x + w, y + h], np.float32) / np.tile(scale, 2)
            # YuNet 的解剖学右眼对应图像左侧；顺序和 ArcFace 模板一致。
            kps = detection[4:14].reshape(5, 2).astype(np.float32) / scale
            result.append(DetectedFace(bbox, kps, float(detection[14])))
        return result
