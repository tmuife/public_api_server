import requests,base64
from io import BytesIO
from PIL import Image
from decouple import config
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class OWLVIT:
    processor = None
    model = None
    device = "cpu"

    # Here, you can replace model_id to model path, then you can use the cached files, like (/models/owlvit)
    # /models/owlvit/
    # ├── config.json
    # ├── preprocessor_config.json
    # ├── pytorch_model.bin
    # ├── special_tokens_map.json(可能有)
    # ├── tokenizer_config.json(可能有)
    def load_model(self, model_id):
        self.processor = OwlViTProcessor.from_pretrained(model_id)
        self.model = OwlViTForObjectDetection.from_pretrained(model_id)

    @staticmethod
    def base64_2_pil(base64_string, ptype='RGB'):
        image_bytes = base64.b64decode(base64_string)
        image_buffer = BytesIO(image_bytes)
        image = Image.open(image_buffer)
        if 'RGB' == ptype:
            return image.convert('RGB')
        else:
            return image

    def __init__(self):
        model_id = config("model_id", cast=str)
        device = config("device", cast=str)
        if self.processor is None or self.model is None:
            self.load_model(model_id)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def get_inference(self, img_base:str, texts: str):
        front_result = []
        image = self.base64_2_pil(img_base)
        texts = [texts.split("|")]
        #texts = [["a photo of a cat", "a photo of a dog"]]
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        # Print detected objects and rescaled box coordinates
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            front_result.append({"label":text[label],"confidence":round(score.item(), 3),"location": box})
        return front_result




