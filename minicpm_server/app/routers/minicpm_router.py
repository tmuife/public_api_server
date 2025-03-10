import base64
import io
import json

import librosa
from PIL import Image
from fastapi import APIRouter
from fastapi.params import Security
from pydantic import BaseModel

from app.middleware.auth import get_api_key
from app.services.minicpm_service import CPMService

cpm_service = CPMService()
class Item(BaseModel):
    content: str
    question: str


def jsonMsg(status, data, error):
    result = {}
    result["status"] = status
    if "success" == status:
        result["data"] = data
    else:
        result["error"] = error
    return json.dumps(result)

def load_audio_from_base64(base64_string: str):
    # 解码 base64 字符串成字节流
    audio_bytes = base64.b64decode(base64_string)
    audio_buffer = io.BytesIO(audio_bytes)
    # 使用 librosa 加载音频
    ref_audio, _ = librosa.load(audio_buffer, sr=16000, mono=True)
    return ref_audio

def base64ToImage(base64_string,type = 'RGB'):
    image_bytes = base64.b64decode(base64_string)
    image_buffer = io.BytesIO(image_bytes)
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
@router.post("/image_query")
async def image_query(item:Item):
    img_content = item.content
    question = item.question
    image = base64ToImage(img_content)
    #question = "Provide a description of the image in English."
    msgs = [{'role': 'user', 'content': [image, question]}]
    res = await cpm_service.chat_omni(msgs=msgs, generate_audio=False, output_audio_path="/tmp")
    print(res)
    #print(str(res).replace("\n", "").replace("<|endoftext|>", ""))
    return jsonMsg("success", res, None)

@router.post("/audio_query")
async def audio_query(item:Item):
    audio_content = item.content
    question = item.question
    audio = load_audio_from_base64(audio_content)
    #question = "Provide a description of the image in English."
    #msgs = [{'role': 'user', 'content': [image, question]}]
    msgs = [{'role': 'user', 'content': [audio,question]}]
    res = await cpm_service.chat_omni(msgs=msgs, generate_audio=False, output_audio_path="/tmp")
    print(type(res))
    print(res)
    #print(str(res).replace("\n", "").replace("<|endoftext|>", ""))
    return jsonMsg("success", res, None)