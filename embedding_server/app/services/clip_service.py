import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

class CLIPSearcher:
    def __init__(self, model_id = "openai/clip-vit-base-patch32", device = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: CLIPModel = CLIPModel.from_pretrained(model_id).to(self.device)
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_id)

    def _to_numpy_features(self, output):
        if isinstance(output, torch.Tensor):
            return output.cpu().detach().numpy()

        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output.cpu().detach().numpy()

        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            # Fallback to the first token if pooling output is unavailable.
            return output.last_hidden_state[:, 0, :].cpu().detach().numpy()

        if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            return output[0].cpu().detach().numpy()

        raise TypeError(f"Unsupported CLIP output type: {type(output).__name__}")

    def get_text_features(self, text):
        inputs = self.tokenizer(text, return_tensors = "pt").to(self.device)
        with torch.no_grad():
            output = self.model.get_text_features(**inputs)
        return self._to_numpy_features(output)

    def get_image_features(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.get_image_features(**inputs)
        return self._to_numpy_features(output)

