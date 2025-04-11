import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, status
from fastapi.middleware.cors import CORSMiddleware
from app.routers import (template, secure,
                         swap_face_router
                         )
from app.tag import SubTags, Tags
from decouple import config
import cv2, os
import numpy as np
from app.services.swap_face_service import Swap
swap = Swap()

app = FastAPI(
    title="FastAPI",
    description="Web API helps you do awesome stuff. 🚀",
    version="0.0.1",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "Walter",
        "url": "http://www.demo.com",
        "email": "jinshuhaicc@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    openapi_url="/api/v1/openapi.json",
    docs_url="/docs",
    openapi_tags=Tags(),
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(template.router)
app.include_router(secure.router)
#app.include_router(m3_router.router)
#app.include_router(clip_router.router)
#app.include_router(paddleocr_router.router)
#app.include_router(insightface_router.router)
app.include_router(swap_face_router.router)
#
#
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, access_token: str = Query(...)):
    if access_token != config("API_KEY"):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    #base_dir = config("base_dir")
    #source_face = "walter.jpg"
    #target_face = "Messi.jpg"
    #swap.set_source_face(cv2.imread(os.path.join(base_dir,source_face)))
    #swap.set_target_face(cv2.imread(os.path.join(base_dir,target_face)))
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive()  # 👈 通用接收
            # 判断消息类型
            if data["type"] == "websocket.receive":
                if "text" in data:
                    #print("收到文本：", data["text"])
                    jobj = json.loads(data["text"])
                    if jobj["face_type"] == "source":
                        swap.set_source_face(image_or_path=swap.base64_2_frame(jobj["face_data"]))
                        await websocket.send_text("set source successful!")
                    elif jobj["face_type"] == "target":
                        swap.set_target_face(image_or_path=swap.base64_2_frame(jobj["face_data"]))
                        await websocket.send_text("set target successful!")
                    else:
                        pass
                elif "bytes" in data:
                    #print("收到二进制数据：", data["bytes"])
                    binary_data=data["bytes"]
                    #binary_data = await websocket.receive_bytes()
                    frame = await swap.swap_face(swap.source_face, swap.target_face, cv2.imdecode(np.frombuffer(binary_data, np.uint8), cv2.IMREAD_COLOR))
                    _, encoded_img = cv2.imencode(".jpg", frame)
                    return_binary_data = encoded_img.tobytes()
                    #print("send_bytes is:", websocket.send_bytes)
                    #print("type is:", type(websocket.send_bytes))
                    await websocket.send_bytes(return_binary_data)
                    #data = await websocket.receive_text()
                    #print(f"Received: {data}")
                    #await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        print("Client disconnected gracefully.")
    except Exception as e:
        print(f"Unexpected error: {e}")

subapi = FastAPI(openapi_tags=SubTags(), swagger_ui_parameters={"defaultModelsExpandDepth": -1})

subapi.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

subapi.include_router(template.router)
#
#
#

app.mount("/subapi", subapi)
