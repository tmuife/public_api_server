from PIL import Image
#from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import base64
from io import BytesIO


class PPOCR:
    def __init__(self, use_angle_cls=True, lang='ch'):
        pass
        #self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')

    def recognize(self, base_img):
        image_bytes = base64.b64decode(base_img)
        image_buffer = BytesIO(image_bytes)
        image = Image.open(image_buffer).convert('RGB')
        open_cv_image = np.array(image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        result = self.ocr.ocr(open_cv_image, cls=True)
        return result



