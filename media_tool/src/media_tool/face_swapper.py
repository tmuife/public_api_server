"""INSwapper inference and paste-back adapted from InsightFace v0.7 (MIT)."""

from pathlib import Path

import cv2
import numpy as np
import onnx

from .alignment import aligned_crop
from .backends import OnnxModel
from .errors import MediaToolError
from .faces import normalize_embedding


class FaceSwapper:
    def __init__(self, path: Path, mode: str):
        self.model = OnnxModel(path, mode)
        inputs = self.model.session.get_inputs()
        if len(inputs) != 2 or inputs[0].shape != [1, 3, 128, 128]:
            raise MediaToolError("换脸模型必须为 inswapper_128 的两输入格式")
        self.image_name, self.latent_name = inputs[0].name, inputs[1].name
        graph = onnx.load(str(path)).graph
        self.emap = onnx.numpy_helper.to_array(graph.initializer[-1]).copy()
        if self.emap.shape != (512, 512) or not np.isfinite(self.emap).all():
            raise MediaToolError("INSwapper 身份映射矩阵无效")

    def map_identity(self, embedding: np.ndarray) -> np.ndarray:
        mapped = normalize_embedding(embedding) @ self.emap
        return normalize_embedding(mapped)[None, :]

    def check(self):
        result = self.model.run(
            {
                self.image_name: np.zeros((1, 3, 128, 128), np.float32),
                self.latent_name: np.zeros((1, 512), np.float32),
            }
        )[0]
        if result.shape != (1, 3, 128, 128) or not np.isfinite(result).all():
            raise MediaToolError("INSwapper 试推理输出无效")

    def swap(self, frame: np.ndarray, face, latent: np.ndarray) -> np.ndarray:
        crop, matrix = aligned_crop(frame, face.kps, 128)
        blob = cv2.dnn.blobFromImage(crop, 1 / 255, (128, 128), swapRB=True)
        prediction = self.model.run({self.image_name: blob, self.latent_name: latent})[0]
        if not np.isfinite(prediction).all():
            raise MediaToolError("换脸输出包含非有限数值")
        restored = np.clip(prediction[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        restored = restored[:, :, ::-1]
        inverse = cv2.invertAffineTransform(matrix)
        height, width = frame.shape[:2]
        warped = cv2.warpAffine(restored, inverse, (width, height))
        mask = cv2.warpAffine(np.full((128, 128), 255, np.float32), inverse, (width, height))
        mask[mask > 20] = 255
        ys, xs = np.where(mask == 255)
        if not len(xs):
            raise MediaToolError("换脸区域未落在有效图像内")
        mask_size = int(np.sqrt((ys.max() - ys.min()) * (xs.max() - xs.min())))
        erosion = max(mask_size // 10, 10)
        mask = cv2.erode(mask, np.ones((erosion, erosion), np.uint8))
        blur = max(mask_size // 20, 5) * 2 + 1
        alpha = cv2.GaussianBlur(mask, (blur, blur), 0)[:, :, None] / 255
        return (alpha * warped + (1 - alpha) * frame.astype(np.float32)).astype(np.uint8)
