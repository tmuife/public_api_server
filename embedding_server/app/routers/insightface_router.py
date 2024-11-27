import os,json,base64,platform
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from fastapi import APIRouter, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKey
from fastapi.params import Security
from app.middleware.auth import get_api_key
from app.services.insightface_service import Face_Recognition

# Initialize the model and request processor
f_reg = Face_Recognition()


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


@router.post("/get_face_features")
def get_image_features(item:Item):
    content = item.content
    try:
        embedding_converted_list = f_reg.get_face_features(content)
        if len(embedding_converted_list)>0:
            return jsonMsg("success", embedding_converted_list, None)
        else:
            return jsonMsg("fail", [], "No Face")
    except Exception as e:
        return jsonMsg("fail", None, e)