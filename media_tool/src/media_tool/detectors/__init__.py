from .insightface import InsightFaceDetector
from .yunet import YuNetDetector


def create_detector(settings, path):
    if settings.detect_method == "insightface":
        return InsightFaceDetector(path, settings)
    if settings.detect_method == "yunet":
        return YuNetDetector(path, settings)
    raise ValueError(f"未知检测器：{settings.detect_method}")
