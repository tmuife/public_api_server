from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

from .errors import MediaToolError
from .faces import normalize_embedding
from .image_io import IMAGE_EXTENSIONS, read_image
from .materials import scan_files

logger = logging.getLogger(__name__)


@dataclass
class Match:
    face: object
    reference: str
    distance: float


class FaceMatcher:
    def __init__(self, references: list[tuple[str, np.ndarray]], threshold: float):
        if not references:
            raise MediaToolError("参考人脸集合为空")
        self.names = [name for name, _ in references]
        self.features = np.stack([normalize_embedding(feature) for _, feature in references])
        self.threshold = threshold

    def match(self, faces) -> list[Match]:
        result = []
        for face in faces:
            embedding = normalize_embedding(face.embedding)
            distances = np.linalg.norm(self.features - embedding, axis=1)
            index = int(np.argmin(distances))
            distance = float(distances[index])
            logger.debug(
                "最近参考=%s 距离=%.6f 阈值=%.4f", self.names[index], distance, self.threshold
            )
            if distance < self.threshold:
                result.append(Match(face, self.names[index], distance))
        return result


def register_faces(settings, analyzer) -> tuple[FaceMatcher, object]:
    reference_paths = scan_files(settings.source_face, IMAGE_EXTENSIONS)
    if not reference_paths:
        raise MediaToolError("参考图片目录为空")
    replacement_paths = scan_files(settings.target_face, IMAGE_EXTENSIONS)
    if len(replacement_paths) != 1:
        raise MediaToolError("target_face 必须恰好包含一张有效单脸图片")

    def one_face(path: Path):
        faces = analyzer.get_faces(read_image(path).frame)
        if len(faces) != 1:
            raise MediaToolError(f"参考/替换图必须恰好有一张脸：{path}（检测到 {len(faces)} 张）")
        return faces[0]

    references = [
        (str(path.relative_to(settings.source_face)), one_face(path).embedding)
        for path in reference_paths
    ]
    return FaceMatcher(references, settings.thresholds), one_face(replacement_paths[0])
