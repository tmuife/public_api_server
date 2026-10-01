from pathlib import Path

import cv2
import numpy as np
import onnx

from .alignment import aligned_crop
from .backends import OnnxModel
from .detectors import create_detector
from .errors import MediaToolError
from .faces import normalize_embedding


class FaceAnalyzer:
    def __init__(self, settings, paths: dict[str, Path]):
        self.detector = create_detector(settings, paths["detector"])
        self.recognition = OnnxModel(paths["recognition"], settings.onnx_provider)
        inputs = self.recognition.session.get_inputs()
        outputs = self.recognition.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape[1:] != [3, 112, 112] or len(outputs) != 1:
            raise MediaToolError("识别模型输入必须为 N×3×112×112")
        self.input_name = inputs[0].name
        graph = onnx.load(str(paths["recognition"])).graph
        embedded_normalization = any(
            n.name.startswith(("Sub", "_minus")) for n in graph.node[:8]
        ) and any(n.name.startswith(("Mul", "_mul")) for n in graph.node[:8])
        self.mean, self.std = (0, 1) if embedded_normalization else (127.5, 127.5)

    def get_faces(self, frame: np.ndarray):
        faces = self.detector.detect(frame)
        for face in faces:
            crop, _ = aligned_crop(frame, face.kps, 112)
            blob = cv2.dnn.blobFromImage(
                crop, 1 / self.std, (112, 112), (self.mean,) * 3, swapRB=True
            )
            face.embedding = normalize_embedding(self.recognition.run({self.input_name: blob})[0])
        return faces

    def check(self):
        self.detector.detect(np.zeros((640, 640, 3), np.uint8))
        output = self.recognition.run({self.input_name: np.zeros((1, 3, 112, 112), np.float32)})[0]
        normalize_embedding(output)
