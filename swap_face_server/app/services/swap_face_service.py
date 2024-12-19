import time
from typing import Any, List
import cv2
import numpy
import insightface
from PIL import Image, ImageDraw
from torch.nn.functional import embedding
from typing import TypedDict, Union, Literal, Generic, TypeVar
import gfpgan
import numpy as np
#import modules.globals
from insightface.model_zoo import model_zoo
from io import BytesIO
import base64
from insightface.app.common import Face
from insightface.app import FaceAnalysis
Frame = numpy.ndarray[Any, Any]
#from modules.utilities import (
#    conditional_download,
#    is_image,
#    is_video,
#)
#from modules.cluster_analysis import find_closest_centroid
import os


class swap:
    def __init__(self):
        abs_dir = os.path.abspath(__file__)
        self._models_dir = os.path.join(os.path.dirname(abs_dir), 'models')
        self._detect_model_path = os.path.join(self._models_dir, 'face_detection_yunet_2023mar.onnx')
        self._detect_model = cv2.FaceDetectorYN.create(
            model=self._detect_model_path,
            config='',
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        self._recognition_model_path = os.path.join(self._models_dir, 'w600k_r50.onnx')
        self._recognition_model = model_zoo.get_model(self._recognition_model_path)
        print(self._models_dir)

    @staticmethod
    def base64_image(base64_string, ptype='RGB'):
        image_bytes = base64.b64decode(base64_string)
        image_buffer = BytesIO(image_bytes)
        image = Image.open(image_buffer)
        if 'RGB' == ptype:
            return image.convert('RGB')
        else:
            return image

    @staticmethod
    def pil_2_cv2(img):
        pil_image = img.convert('RGB')
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image

    @staticmethod
    def euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def get_face(self, img_path: Union[str, Frame]):
        image: Union[str, Frame]
        if isinstance(img_path, str):
            image = cv2.imread(img_path)
        else:
            image = img_path
        (frame_h, frame_w) = image.shape[:2]
        yunet.setInputSize([frame_w, frame_h])
        _, faces = yunet.detect(image)  # # faces: None, or nx15 np.array
        return_faces = []
        if faces is not None:
            for idx, face in enumerate(faces):
                coords = face[:-1].astype(np.int32)
                box = [
                    (coords[0], coords[1]),
                    (coords[0] + coords[2], coords[1] + coords[3])
                ]
                kps = [np.array([coords[4], coords[5]]).astype(np.float32),
                       np.array([coords[6], coords[7]]).astype(np.float32),
                       np.array([coords[8], coords[9]]).astype(np.float32),
                       np.array([coords[10], coords[11]]).astype(np.float32),
                       np.array([coords[12], coords[13]]).astype(np.float32)]
                det_score = face[-1]
                bbox = np.array(box).astype(np.float32)
                _face = Face(bbox=bbox, kps=np.array(kps), det_score=det_score)
                rec_model.get(image, _face)
                return_faces.append(_face)
        return return_faces

    def swap_face(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        model_path = os.path.join(models_dir, 'inswapper_128.onnx')
        face_swapper = insightface.model_zoo.get_model(
            model_path, providers=modules.globals.execution_providers
        )
        swapped_frame = face_swapper.get(
            temp_frame, target_face, source_face, paste_back=True
        )
        return swapped_frame

    def enhance_face(self, temp_frame: Frame) -> Frame:
        model_path = os.path.join(models_dir, 'GFPGANv1.4.pth')
        face_enhancer = gfpgan.GFPGANer(model_path=model_path, upscale=1)  # type: ignore[attr-defined]
        _, _, temp_frame = face_enhancer.enhance(temp_frame, paste_back=True)
        return temp_frame

    def convert(self, id_or_path):
        cap = cv2.VideoCapture(id_or_path)
        if cap.isOpened():
            w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                                   cv2.CAP_PROP_FRAME_HEIGHT,
                                                   cv2.CAP_PROP_FPS))
            _face_source = get_face("/Users/walter/Downloads/unknown.jpg")

            _target_face = get_face("/Users/walter/Downloads/walter.jpg")
            model_path = os.path.join(models_dir, 'inswapper_128.onnx')
            face_swapper = insightface.model_zoo.get_model(
                model_path, providers=modules.globals.execution_providers
            )

            while True:
                success, frame = cap.read()
                out_frame = face_swapper.get(
                    frame, _target_face, _face_source, paste_back=True
                )
                cv2.imshow('Live', out_frame)
                time.sleep(10)
                if cv2.waitKey(1) == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()

    def swap_example(self):
        # the face you want
        _face_source = get_face("/Users/walter/Downloads/walter.jpg")
        # _face_source = get_face(pil_2_cv2(base64ToImage(img)))
        # the face will be replaced
        _target_face = get_face("/Users/walter/Downloads/Messi.jpg")
        input_frame = cv2.imread("/Users/walter/Downloads/Messi.jpg")
        out_frame = swap_face(source_face=_face_source[0], target_face=_target_face[0], temp_frame=input_frame)
        cv2.imwrite("/Users/walter/Downloads/Messi_swapped.jpg", out_frame)

    def swap_example2(self):
        # the face you want
        _source_face = None
        _faces = get_face("/Users/walter/Downloads/Messi.jpg")
        # _faces = get_face(pil_2_cv2(base64ToImage(img)))
        if len(_faces) > 0:
            _source_face = _faces[0]
        # _face_source = get_face(pil_2_cv2(base64ToImage(img)))
        # the face will be replaced
        _target_face = None
        _target_embedding = None
        _faces = get_face("/Users/walter/Downloads/walter.jpg")
        if len(_faces) > 0:
            _target_face = _faces[0]
            _target_embedding = _target_face.normed_embedding

        input_frame = cv2.imread("/Users/walter/Downloads/me_and_other.jpg")
        _faces = get_face(input_frame)
        for _face in _faces:
            _embedding = _face.normed_embedding
            distance = euclidean_distance(_embedding, _target_embedding)
            print(distance)
            if distance < 0.6:
                out_frame = swap_face(source_face=_source_face, target_face=_face, temp_frame=input_frame)
                while True:
                    cv2.imshow('Live', out_frame)
                    time.sleep(1)
                    if cv2.waitKey(1) == 27:
                        break
                cv2.destroyAllWindows()

    def enhance_example(self):
        source_frame = "/Users/walter/Downloads/Messi_swapped.jpg"
        image_enhanced = enhance_face(cv2.imread(source_frame))
        cv2.imwrite("/Users/walter/Downloads/Messi_swapped_enhanced.jpg", image_enhanced)


_s = swap()