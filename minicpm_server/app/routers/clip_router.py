#import os,json,base64,platform
#from io import BytesIO
#from PIL import Image
#from pydantic import BaseModel
#from fastapi import APIRouter, Depends, Form
#from fastapi.responses import JSONResponse
#from fastapi.security.api_key import APIKey
#from fastapi.params import Security
#from app.middleware.auth import get_api_key
#from app.services.clip_service import CLIPSearcher
#
## Initialize the model and request processor
#clip_searcher = CLIPSearcher()
#
#
#class Item(BaseModel):
#    content: str
#
#
#def jsonMsg(status, data, error):
#    result = {}
#    result["status"] = status
#    if "success" == status:
#        result["data"] = data
#    else:
#        result["error"] = error
#    return json.dumps(result)
#
#
#def base64ToImage(base64_string,type = 'RGB'):
#    image_bytes = base64.b64decode(base64_string)
#    image_buffer = BytesIO(image_bytes)
#    image = Image.open(image_buffer)
#    if 'RGB' == type:
#        return image.convert('RGB')
#    else:
#        return image
#
#
#router = APIRouter(
#    prefix="/clip",
#    tags=["SECURE"],
#    responses={404: {"message": "Not found"}},
#    dependencies=[Security(get_api_key)]
#)
#
#
#@router.post("/get_text_features")
#def get_text_features(item:Item):
#    content = item.content
#    try:
#        embedding = clip_searcher.get_text_features(text=content)
#        return jsonMsg("success",str(embedding[0].tolist()),None)
#    except Exception as e:
#        return jsonMsg("fail",None,e)
#
#
#@router.post("/get_image_features")
#def get_image_features(item:Item):
#    content = item.content
#    try:
#        embedding = clip_searcher.get_image_features(image=base64ToImage(content))
#        return jsonMsg("success", embedding[0].tolist(), None)
#    except Exception as e:
#        return jsonMsg("fail", None, e)