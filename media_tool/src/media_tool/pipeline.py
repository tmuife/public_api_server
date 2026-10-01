from dataclasses import dataclass

import numpy as np

from .face_analyzer import FaceAnalyzer
from .face_matcher import register_faces
from .face_swapper import FaceSwapper


@dataclass
class FrameResult:
    frame: np.ndarray
    detected: int = 0
    matched: int = 0
    swapped: int = 0
    enhanced: int = 0
    matches: list | None = None


class FacePipeline:
    def __init__(self, analyzer, matcher, swapper, latent, enhancer=None):
        self.analyzer, self.matcher = analyzer, matcher
        self.swapper, self.latent, self.enhancer = swapper, latent, enhancer

    def process(self, frame: np.ndarray) -> FrameResult:
        faces = self.analyzer.get_faces(frame)
        matches = self.matcher.match(faces)
        if not matches:
            return FrameResult(frame, detected=len(faces), matches=[])
        working = frame.copy()
        for match in matches:
            working = self.swapper.swap(working, match.face, self.latent)
        if self.enhancer is not None:
            for match in matches:
                working = self.enhancer.enhance(working, match.face)
        details = [{"reference": match.reference, "distance": match.distance} for match in matches]
        return FrameResult(
            working,
            len(faces),
            len(matches),
            len(matches),
            len(matches) if self.enhancer is not None else 0,
            details,
        )

    def backend_info(self) -> dict:
        detector_model = getattr(self.analyzer.detector, "model", None)
        return {
            "detector": detector_model.describe()
            if detector_model
            else {"providers": ["OpenCV CPU"]},
            "recognition": self.analyzer.recognition.describe(),
            "swap": self.swapper.model.describe(),
            "enhance": str(self.enhancer.device) if self.enhancer else "disabled",
        }


def build_pipeline(settings) -> FacePipeline:
    paths = settings.required_models()
    analyzer = FaceAnalyzer(settings, paths)
    analyzer.check()
    matcher, replacement = register_faces(settings, analyzer)
    swapper = FaceSwapper(paths["swap"], settings.onnx_provider)
    swapper.check()
    enhancer = None
    if settings.enhance_enabled:
        # 无增强时不导入 PyTorch，不加载 GFPGAN。
        from .face_enhancer import FaceEnhancer

        enhancer = FaceEnhancer(paths["enhance"], settings.enhance_device, settings.enhance_blend)
        enhancer.check()
    return FacePipeline(
        analyzer, matcher, swapper, swapper.map_identity(replacement.embedding), enhancer
    )
