#import os,json,base64,platform
#from io import BytesIO
#from PIL import Image
#from pydantic import BaseModel
#from fastapi import APIRouter, Depends, Form
#from fastapi.responses import JSONResponse
#from fastapi.security.api_key import APIKey
#from fastapi.params import Security
#from app.middleware.auth import get_api_key
#from app.services.paddleocr_service import PPOCR
#
## Initialize the model and request processor
#ppocr = PPOCR()
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
#router = APIRouter(
#    prefix="/ocr",
#    tags=["SECURE"],
#    responses={404: {"message": "Not found"}},
#    dependencies=[Security(get_api_key)]
#)
#
#
#@router.post("/ocr_recognize")
#def ocr_recognize(item:Item):
#    content = item.content
#    try:
#        result = ppocr.recognize(base_img=content)
#        return jsonMsg("success",result,None)
#    except Exception as e:
#        return jsonMsg("fail",None,e)
#
