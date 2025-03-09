import requests
import os,json,base64,platform
from io import BytesIO
from click import prompt
from fastapi import APIRouter, Depends, Form
from pydantic import BaseModel
from fastapi.params import Security
from app.middleware.auth import get_api_key
from app.services.minicpm_service import CPMService
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


cpm_service = CPMService()
class Item(BaseModel):
    img_content: str
    question: str


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

router = APIRouter(
    prefix="/minicpm",
    tags=["SECURE"],
    responses={404: {"message": "Not found"}},
    dependencies=[Security(get_api_key)]
)
@router.post("/image_chat")
def image_process(item:Item):
    img_content = item.img_content
    question = item.question
    image = base64ToImage(img_content)
    #question = "Provide a description of the image in English."
    msgs = [{'role': 'user', 'content': [image, question]}]

    res = cpm_service.chat_omni(msgs=msgs, generate_audio=False, output_audio_path="/tmp")
    print(res)
    print(str(res).replace("\n", "").replace("<|endoftext|>", ""))
    return jsonMsg("success", res, None)
