import os, base64, numpy as np
from io import BytesIO
from PIL import Image
from insightface.app.common import Face
from insightface.model_zoo import model_zoo
import cv2


class Face_Recognition:

    def __init__(self):
        self.base_model_dir = os.path.join(os.path.dirname(__file__), "../../models")
        self.rec_model_path = os.path.join(self.base_model_dir, 'buffalo_l/w600k_r50.onnx')
        self.rec_model = model_zoo.get_model(self.rec_model_path)
        self.yunet_detect_model = os.path.join(self.base_model_dir, "yunet/face_detection_yunet_2023mar.onnx")
        self.yunet = cv2.FaceDetectorYN.create(
            model=self.yunet_detect_model,
            config='',
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

    def base64ToImage(self, base64_string):
        image_bytes = base64.b64decode(base64_string)
        image_buffer = BytesIO(image_bytes)
        image = Image.open(image_buffer)
        if 'RGB' == type:
            return image.convert('RGB')
        else:
            return image

    def pil_2_cv2(self, img):
        pil_image = img.convert('RGB')
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image

    def get_face_features(self, image_content):
        image_cv = self.pil_2_cv2(self.base64ToImage(image_content))
        (frame_h, frame_w) = image_cv.shape[:2]
        self.yunet.setInputSize([frame_w, frame_h])
        _, faces = self.yunet.detect(image_cv)  # # faces: None, or nx15 np.array
        features = []
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
                face_new = Face(bbox=bbox, kps=np.array(kps), det_score=det_score)
                self.rec_model.get(image_cv, face_new)
                embedding = face_new.normed_embedding
                features.append(embedding.tolist())
        return features
