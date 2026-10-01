"""与 InsightFace 的 SimilarityTransform 等价的五点最小二乘对齐。"""

import cv2
import numpy as np

from .errors import MediaToolError

ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
GFPGAN_TEMPLATE = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float64,
)


def similarity_matrix(points: np.ndarray, template: np.ndarray) -> np.ndarray:
    # 保留关键点/模板的原始精度，与上游 skimage Umeyama 的运算顺序一致。
    source = np.asarray(points)
    if source.shape != (5, 2) or not np.isfinite(source).all():
        raise MediaToolError("人脸关键点必须为有限的 5×2 坐标")
    source_mean, dest_mean = source.mean(axis=0), template.mean(axis=0)
    source_centered, dest_centered = source - source_mean, template - dest_mean
    variance = source_centered.var(axis=0).sum()
    if variance < 1e-8:
        raise MediaToolError("人脸关键点退化，无法对齐")
    covariance = dest_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(2)
    if np.linalg.det(covariance) < 0:
        sign[-1] = -1
    rank = np.linalg.matrix_rank(covariance)
    if rank == 0:
        raise MediaToolError("人脸关键点退化，无法对齐")
    if rank == 1 and np.linalg.det(u) * np.linalg.det(vt) > 0:
        rotation = u @ vt
    elif rank == 1:
        saved = sign[-1]
        sign[-1] = -1
        rotation = u @ np.diag(sign) @ vt
        sign[-1] = saved
    else:
        rotation = u @ np.diag(sign) @ vt
    scale = 1.0 / variance * (singular @ sign)
    matrix = np.empty((2, 3), dtype=np.float64)
    matrix[:, 2] = dest_mean - scale * (rotation @ source_mean.T)
    matrix[:, :2] = rotation
    matrix[:, :2] *= scale
    return matrix


def aligned_crop(
    frame: np.ndarray, kps: np.ndarray, size: int, *, enhance: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    if enhance:
        template = GFPGAN_TEMPLATE * (size / 512)
    elif size % 112 == 0:
        template = ARCFACE_TEMPLATE * (size / 112)
    else:
        template = ARCFACE_TEMPLATE * (size / 128)
        template[:, 0] += 8 * (size / 128)
    matrix = similarity_matrix(kps, template)
    return cv2.warpAffine(frame, matrix, (size, size)), matrix


def paste_enhanced(
    frame: np.ndarray, restored: np.ndarray, matrix: np.ndarray, blend: float
) -> np.ndarray:
    height, width = frame.shape[:2]
    mask = np.zeros((512, 512), np.float32)
    cv2.ellipse(mask, (256, 285), (190, 215), 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    inverse = cv2.invertAffineTransform(matrix)
    warped = cv2.warpAffine(restored, inverse, (width, height))
    alpha = cv2.warpAffine(mask, inverse, (width, height))[:, :, None] * blend
    return np.clip(alpha * warped + (1 - alpha) * frame, 0, 255).astype(np.uint8)
