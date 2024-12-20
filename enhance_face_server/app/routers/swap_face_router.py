import os,json,base64,platform
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from fastapi import APIRouter, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKey
from fastapi.params import Security
from app.middleware.auth import get_api_key
from app.services.swap_face_service import Swap

# Initialize the model and request processor
swap = Swap()


class Item(BaseModel):
    content: str


def jsonMsg(status, data, error):
    result = {}
    result["status"] = status
    if "success" == status:
        result["data"] = data
    else:
        result["error"] = error
    return json.dumps(result)


router = APIRouter(
    prefix="/face",
    tags=["SECURE"],
    responses={404: {"message": "Not found"}},
    dependencies=[Security(get_api_key)]
)

@router.post("/set_source_face")
def set_source_face(item:Item):
    content = item.content
    try:
        swap.set_source_face(swap.base64_2_frame(content))
        return jsonMsg("success", "set source success", None)
    except Exception as e:
        return jsonMsg("fail", None, e)

@router.post("/set_target_face")
def set_target_face(item:Item):
    content = item.content
    try:
        swap.set_target_face(swap.base64_2_frame(content))
        return jsonMsg("success", "set source success", None)
    except Exception as e:
        return jsonMsg("fail", None, e)

@router.post("/swap_face")
def swap_face(item:Item):
    content = item.content
    try:
        frame = swap.swap_face(swap.source_face, swap.target_face, swap.base64_2_frame(content))
        return jsonMsg("success", swap.frame_2_base64(frame), None)
    except Exception as e:
        return jsonMsg("fail", None, e)

@router.post("/swap_face_and_enhance")
def swap_face_and_enhance(item:Item):
    content = item.content
    try:
        frame = swap.enhance_face(swap.swap_face(swap.source_face, swap.target_face, swap.base64_2_frame(content)))
        return jsonMsg("success", swap.frame_2_base64(frame), None)
    except Exception as e:
        return jsonMsg("fail", None, e)

@router.post("/enhance_face")
def enhance_face(item:Item):
    content = item.content
    try:
        frame = swap.enhance_face(swap.base64_2_frame(content))
        return jsonMsg("success", swap.frame_2_base64(frame), None)
    except Exception as e:
        return jsonMsg("fail", None, e)