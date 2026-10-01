from dataclasses import dataclass

import numpy as np

from .errors import MediaToolError


@dataclass
class DetectedFace:
    bbox: np.ndarray
    kps: np.ndarray
    score: float
    embedding: np.ndarray | None = None


def normalize_embedding(value: np.ndarray) -> np.ndarray:
    feature = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feature)
    if feature.shape != (512,) or not np.isfinite(feature).all() or norm <= 1e-8:
        raise MediaToolError("人脸特征必须是有限的非零 512 维向量")
    return feature / norm
