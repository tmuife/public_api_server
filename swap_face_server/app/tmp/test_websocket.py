import asyncio
import os.path

import websockets
from decouple import config
import json, io
import requests
import base64
from typing import Any, List
import numpy,cv2
Frame = numpy.ndarray[Any, Any]


@staticmethod
def frame_2_base64(image: Frame):
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
    return img_base64

@staticmethod
def base64_2_frame(base_str):
    img_data = base64.b64decode(base_str)
    # Step 3: Convert bytes into a NumPy array
    img_array = numpy.frombuffer(img_data, dtype=numpy.uint8)
    # Step 4: Decode the NumPy array into an OpenCV image
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image

def frame_2_binary(frame: Frame) -> bytes:
    _, encoded_img = cv2.imencode(".jpg", frame)
    return_binary_data = encoded_img.tobytes()
    return return_binary_data

def binary_2_frame(binary_data: bytes) -> Frame:
    frame = cv2.imdecode(numpy.frombuffer(binary_data, numpy.uint8), cv2.IMREAD_COLOR)
    return frame

def remote_call(data, url, api_key) -> json:
    # 设置请求头
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        "access_token": config("API_KEY")
    }
    # 发送 GET 请求
    response = requests.post(url, headers=headers, json=data)
    # 输出返回的结果
    if response.status_code == 200:
        #print("成功调用接口:", response.text)
        jobj = json.loads(response.text)
        if type(jobj) == str:
            jobj = json.loads(jobj)
        return jobj
    else:
        print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
        return None

async def call_websocket():
    token = config("API_KEY")
    base_dir = config("base_dir")
    source_face = "walter.jpg"
    target_face = "Messi.jpg"
    test_image = "Messi.jpg"

    #remote_call({"content": frame_2_base64(cv2.imread(os.path.join(base_dir,source_face)))},
    #                 config("set_source_face_url"), token)
    #remote_call({"content": frame_2_base64(cv2.imread(os.path.join(base_dir,target_face)))},
    #                 config("set_target_face_url"), token)

    uri = f"ws://140.238.3.222:7860/ws?access_token={token}"
    frame = None
    async with websockets.connect(uri) as websocket:
        #for i in range(5):
        #    msg = f"Message {i}"
        #    await websocket.send(msg)
        #    print(f"Sent: {msg}")
        #    response = await websocket.recv()
        #    print(f"Received from server: {response}")
        #    await asyncio.sleep(0.1)  # 等待 1 秒再发下一条

        source_face = {"face_type": "source", "face_data": frame_2_base64(cv2.imread(os.path.join(base_dir, source_face)))}
        target_face = {"face_type": "target", "face_data": frame_2_base64(cv2.imread(os.path.join(base_dir, target_face)))}
        await websocket.send(json.dumps(source_face))
        response = await websocket.recv()
        print(response)

        await websocket.send(json.dumps(target_face))
        response = await websocket.recv()
        print(response)
        for i in range(0,100):
            binary_data = frame_2_binary(cv2.imread(os.path.join(base_dir,test_image)))
            await websocket.send(binary_data)
            response = await websocket.recv()
            print(i)

        # 2. Show image in a window
        cv2.imshow("My Image", binary_2_frame(response))
        # 3. Wait for a key press (0 = wait forever)
        cv2.waitKey(0)
        # 4. Close all OpenCV windows
        cv2.destroyAllWindows()

        print("Done sending messages. Closing connection.")



if __name__ == "__main__":
    asyncio.run(call_websocket())
