import time
from typing import Any, List
import cv2
import numpy
import insightface
from PIL import Image, ImageDraw
#from joblib.externals.cloudpickle import instance
#from torch.nn.functional import embedding
from typing import TypedDict, Union, Literal, Generic, TypeVar
#import gfpgan
import numpy as np
#import modules.globals
from insightface.model_zoo import model_zoo
from io import BytesIO
import base64
from insightface.app.common import Face
from insightface.app import FaceAnalysis
from decouple import config
Frame = numpy.ndarray[Any, Any]
#from modules.utilities import (
#    conditional_download,
#    is_image,
#    is_video,
#)
#from modules.cluster_analysis import find_closest_centroid
import os
#CUDAExecutionProvider
execution_providers: List[str] = [config("execution_providers")]
detect_method=config("detect_method")

class Swap:
    def __init__(self):
        abs_dir = os.path.abspath(__file__)
        self._models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), 'models')
        print(self._models_dir)
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
        self._swap_model_path = os.path.join(self._models_dir, 'inswapper_128.onnx')
        self._face_swapper = insightface.model_zoo.get_model(
            self._swap_model_path,
            providers=execution_providers
        )
        #self._enhance_model_path = os.path.join(self._models_dir, 'GFPGANv1.4.pth')
        #self._face_enhancer = gfpgan.GFPGANer(model_path=self._enhance_model_path, upscale=1)  # type: ignore[attr-defined]
        self._face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=execution_providers)
        self._face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        self.source_face = None
        self.target_face = None


    @staticmethod
    def frame_2_base64(image: Frame):
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return img_base64

    @staticmethod
    def base64_2_frame(base_str):
        img_data = base64.b64decode(base_str)
        # Step 3: Convert bytes into a NumPy array
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        # Step 4: Decode the NumPy array into an OpenCV image
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return image

    @staticmethod
    def base64_2_pil(base64_string, ptype='RGB'):
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

    def set_source_face(self, image_or_path: Union[str, Frame], multi=False):
        if not multi:
            image: Union[str, Frame]
            if isinstance(image_or_path, str):
                image = cv2.imread(image_or_path)
            else:
                image = image_or_path
            faces = self.get_face(image)
            if len(faces)>0:
                self.source_face = faces[0]

    def set_target_face(self, image_or_path: Union[str, Frame], multi=False):
        if not multi:
            image: Union[str, Frame]
            if isinstance(image_or_path, str):
                image = cv2.imread(image_or_path)
            else:
                image = image_or_path
            faces = self.get_face(image)
            if len(faces)>0:
                self.target_face = faces[0]

    def get_oneface(self, frame: Frame) -> Any:
        face = self._face_analyser.get(frame)
        try:
            return min(face, key=lambda x: x.bbox[0])
        except ValueError:
            return None
    def get_face(self, img_path: Union[str, Frame], method=detect_method):
        image: Union[str, Frame]
        if isinstance(img_path, str):
            image = cv2.imread(img_path)
        else:
            image = img_path
        if method=="insight":
            try:
                return self._face_analyser.get(image)
            except IndexError:
                return None
        if method=="yu":
            (frame_h, frame_w) = image.shape[:2]
            self._detect_model.setInputSize([frame_w, frame_h])
            _, faces = self._detect_model.detect(image)  # # faces: None, or nx15 np.array
            detected_faces = []
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
                    self._recognition_model.get(image, _face)
                    detected_faces.append(_face)
            return detected_faces

    def swap_face_deprecated(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        if self.source_face is None or self.target_face is None:
            print("Please set source and target face first!")
            exit(-1)
        swapped_frame = self._face_swapper.get(
            temp_frame, target_face, source_face, paste_back=True
        )
        return swapped_frame

    # just swap single face
    def swap_face(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        if self.source_face is None or self.target_face is None:
            print("Please set source and target face first!")
            exit(-1)
        _faces = self.get_face(temp_frame)
        for _face in _faces:
            _embedding = _face.normed_embedding
            distance = self.euclidean_distance(_embedding, self.target_face.normed_embedding)
            print(distance)
            if distance < 0.9:
                swapped_frame = self._face_swapper.get(
                    temp_frame, _face, source_face, paste_back=True
                )
                return swapped_frame
        return temp_frame

#    def enhance_face(self, temp_frame: Frame) -> Frame:
#        _, _, temp_frame = self._face_enhancer.enhance(temp_frame, paste_back=True)
#        return temp_frame


#_s = Swap()
#_s.set_source_face("/Users/walter/Downloads/walter.jpg")
### the face will be replaced
#_s.set_target_face("/Users/walter/Downloads/embedding.jpg")
#cap = cv2.VideoCapture("/Users/walter/Downloads/embedding.mp4")
#while cap.isOpened():
#    success, input_frame = cap.read()
#    if success:
#        #input_frame = cv2.imread("/Users/walter/Downloads/me_and_other.jpg")
#        _faces = _s.get_face(input_frame)
#        for _face in _faces:
#            _embedding = _face.normed_embedding
#            distance = _s.euclidean_distance(_embedding, _s.target_face.normed_embedding)
#            print(distance)
#            if distance < 0.6:
#                out_frame = _s.swap_face(source_face=_s.source_face, target_face=_face, temp_frame=input_frame)
#                #out_frame = _s.enhance_face(out_frame)
#                cv2.imshow('Live', out_frame)
#                time.sleep(0.01)
#    if cv2.waitKey(1) == 27:
#        break
#    cv2.destroyAllWindows()