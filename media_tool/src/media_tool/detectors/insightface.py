"""SCRFD decoder adapted from InsightFace v0.7 (MIT); see THIRD_PARTY_NOTICES.md."""

from pathlib import Path

import cv2
import numpy as np

from ..backends import OnnxModel
from ..errors import MediaToolError
from ..faces import DetectedFace


def non_maximum_suppression(
    boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.3
) -> list[int]:
    order = scores.argsort()[::-1]
    kept = []
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    while order.size:
        index = int(order[0])
        kept.append(index)
        rest = order[1:]
        left = np.maximum(boxes[index, :2], boxes[rest, :2])
        right = np.minimum(boxes[index, 2:], boxes[rest, 2:])
        intersection = np.maximum(right - left + 1, 0).prod(axis=1)
        union = areas[index] + areas[rest] - intersection
        iou = intersection / np.maximum(union, 1e-8)
        order = rest[iou <= threshold]
    return kept


class InsightFaceDetector:
    def __init__(self, path: Path, settings):
        self.model = OnnxModel(path, settings.onnx_provider)
        inputs, outputs = self.model.session.get_inputs(), self.model.session.get_outputs()
        if len(inputs) != 1 or len(outputs) not in {9, 15}:
            raise MediaToolError("SCRFD 模型必须包含五点关键点输出（9 或 15 个输出）")
        self.input_name = inputs[0].name
        self.batched = len(outputs[0].shape) == 3
        self.strides = [8, 16, 32] if len(outputs) == 9 else [8, 16, 32, 64, 128]
        self.anchors = 2 if len(outputs) == 9 else 1
        self.threshold = settings.detection_score_threshold
        shape = inputs[0].shape
        self.size = (
            (shape[3], shape[2])
            if all(isinstance(d, int) for d in shape[2:])
            else (settings.detection_max_side, settings.detection_max_side)
        )
        self.centers: dict = {}

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        height, width = frame.shape[:2]
        input_width, input_height = self.size
        ratio = min(input_width / width, input_height / height)
        resized_width, resized_height = max(1, int(width * ratio)), max(1, int(height * ratio))
        scaled = cv2.resize(frame, (resized_width, resized_height))
        padded = np.zeros((input_height, input_width, 3), np.uint8)
        padded[:resized_height, :resized_width] = scaled
        blob = cv2.dnn.blobFromImage(padded, 1 / 128, self.size, (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.model.run({self.input_name: blob})
        scores_list, boxes_list, points_list = [], [], []
        levels = len(self.strides)
        for level, stride in enumerate(self.strides):
            scores, distances, points = (
                outputs[level],
                outputs[level + levels],
                outputs[level + levels * 2],
            )
            if self.batched:
                scores, distances, points = scores[0], distances[0], points[0]
            scores = scores.reshape(-1)
            key = (input_height // stride, input_width // stride, stride)
            if key not in self.centers:
                centers = np.stack(np.mgrid[: key[0], : key[1]][::-1], axis=-1).astype(np.float32)
                centers = (centers * stride).reshape(-1, 2)
                self.centers[key] = np.repeat(centers, self.anchors, axis=0)
            centers = self.centers[key]
            chosen = np.flatnonzero(scores >= self.threshold)
            distances = distances.reshape(-1, 4)[chosen] * stride
            anchor = centers[chosen]
            boxes = np.column_stack((anchor - distances[:, :2], anchor + distances[:, 2:]))
            points = points.reshape(-1, 5, 2)[chosen] * stride + anchor[:, None, :]
            boxes_list.append(boxes)
            points_list.append(points)
            scores_list.append(scores[chosen])
        scores = np.concatenate(scores_list)
        if not len(scores):
            return []
        boxes, points = np.concatenate(boxes_list), np.concatenate(points_list)
        # 使用实际缩放比，处理整数尺寸取整造成的横纵微小差异。
        xy_scale = np.array([resized_width / width, resized_height / height], np.float32)
        boxes /= np.tile(xy_scale, 2)
        points /= xy_scale
        selected = non_maximum_suppression(boxes, scores)
        return [
            DetectedFace(
                boxes[i].astype(np.float32), points[i].astype(np.float32), float(scores[i])
            )
            for i in selected
        ]
