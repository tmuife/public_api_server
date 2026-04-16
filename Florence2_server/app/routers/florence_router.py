import requests
import os,json,base64,platform
from io import BytesIO
import logging
from click import prompt
from fastapi import APIRouter, Depends, Form
from pydantic import BaseModel
from fastapi.params import Security
from app.middleware.auth import get_api_key
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

class Item(BaseModel):
    content: str
    prompt: str

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_kwargs = {
    "trust_remote_code": True,
    "attn_implementation": "eager",
}
try:
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        dtype=torch_dtype,
        **model_kwargs,
    ).to(device)
except TypeError:
    # Backward compatibility for older transformers that only accept torch_dtype.
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        torch_dtype=torch_dtype,
        **model_kwargs,
    ).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
logger = logging.getLogger(__name__)


def jsonMsg(status, data, error):
    result = {}
    result["status"] = status
    if "success" == status:
        result["data"] = data
    else:
        result["error"] = error
    return json.dumps(result)


def base64ToImage(base64_string,type = 'RGB'):
    image_bytes = base64.b64decode(base64_string)
    image_buffer = BytesIO(image_bytes)
    image = Image.open(image_buffer)
    if 'RGB' == type:
        return image.convert('RGB')
    else:
        return image


def _generate_with_fallback(input_ids, pixel_values):
    generate_kwargs = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "max_new_tokens": 1024,
        "do_sample": False,
    }
    try:
        return model.generate(
            **generate_kwargs,
            num_beams=3,
        )
    except AttributeError as exc:
        # Florence-2 remote code can fail in beam-search cache preparation with some
        # transformers/torch combinations. Fallback to greedy + no KV cache.
        if "NoneType" in str(exc) and "shape" in str(exc):
            logger.warning(
                "Florence beam-search cache issue detected, fallback to greedy decode: %s",
                exc,
            )
            return model.generate(
                **generate_kwargs,
                num_beams=1,
                use_cache=False,
            )
        raise

router = APIRouter(
    prefix="/florence",
    tags=["SECURE"],
    responses={404: {"message": "Not found"}},
    dependencies=[Security(get_api_key)]
)
@router.post("/image_process")
def image_process(item:Item):
    #prompt = "<OD>"
    prompt = item.prompt
    #url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    #image = Image.open(requests.get(url, stream=True).raw)
    image = base64ToImage(item.content)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch_dtype)

    generated_ids = _generate_with_fallback(
        input_ids=input_ids,
        pixel_values=pixel_values,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task="<OD>",
                                                      image_size=(image.width, image.height))

    print(parsed_answer)
    return jsonMsg("success", parsed_answer, None)
