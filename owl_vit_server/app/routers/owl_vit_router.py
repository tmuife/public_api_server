import os,json,base64,platform
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from fastapi import APIRouter, Depends, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKey
from fastapi.params import Security
from app.middleware.auth import get_api_key
from app.services.owl_vit_service import OWLVIT
import numpy as np
from pathlib import Path
import subprocess


# Initialize the model and request processor
owl = OWLVIT()


class Item(BaseModel):
    image_base: str
    texts: str


def jsonMsg(status, data, error):
    result = {}
    result["status"] = status
    if "success" == status:
        result["data"] = data
    else:
        result["error"] = error
    return json.dumps(result)


router = APIRouter(
    prefix="/detect",
    tags=["SECURE"],
    responses={404: {"message": "Not found"}},
    dependencies=[Security(get_api_key)]
)


@router.post("/detect_with_prompt")
async def detect_with_prompt(item:Item):
    image_base = item.image_base
    texts = item.texts
    try:
        result = owl.get_inference(image_base, texts)
        return jsonMsg("success", result, None)
    except Exception as e:
        return jsonMsg("fail", None, e)

