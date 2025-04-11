import os,json,base64,platform
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from fastapi import APIRouter, Depends, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKey
from fastapi.params import Security
from app.middleware.auth import get_api_key
from app.services.swap_face_service import Swap
import cv2
import numpy as np
from pathlib import Path
import subprocess


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
async def swap_face(item:Item):
    content = item.content
    try:
        frame = await swap.swap_face(swap.source_face, swap.target_face, swap.base64_2_frame(content))
        return jsonMsg("success", swap.frame_2_base64(frame), None)
    except Exception as e:
        return jsonMsg("fail", None, e)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            #data = await websocket.receive_text()
            binary_data = websocket.receive_bytes()
            frame = await swap.swap_face(swap.source_face, swap.target_face, cv2.imdecode(np.frombuffer(binary_data, np.uint8), cv2.IMREAD_COLOR))
            _, encoded_img = cv2.imencode(".jpg", frame)
            return_binary_data = encoded_img.tobytes()
            await websocket.send_bytes(return_binary_data)
            #await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        print("Client disconnected gracefully.")
    except Exception as e:
        print(f"Unexpected error: {e}")

#curl -X 'POST' \
#  'http://127.0.0.1:7860/uploadfile/' \
#  -H 'accept: application/json' \
#  -H 'Content-Type: multipart/form-data' \
#  -H 'access_token: valid_token' \
#  -F 'file=@/Users/walter/Downloads/ai.txt' \
#  -F 'description=This is a test file'
from fastapi import FastAPI, File, UploadFile
UPLOAD_DIR = "./uploads"
@router.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...), action: str = Form(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    file_suffix = Path(file.filename).suffix.lstrip(".")
    print(action)
    stream = BytesIO()
    with open(file_location, "wb") as f:
        stream.write(await file.read())
    stream.seek(0)
    ## 转换为 NumPy 数组
    #nparr = np.frombuffer(stream.getvalue(), np.uint8)
    ## 解码为 OpenCV 图像
    #frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if action in ["source_face","target_face"]:
        img_base64 = base64.b64encode(stream.getvalue()).decode("utf-8")
        _item = Item(content=img_base64)
        if action == "source_face":
            set_source_face(_item)
        elif action == "target_face":
            set_target_face(_item)
    elif action in ["swap_face"]:
        read_video_meta_ffprobe(stream)
        read_video_frame_ffmpeg(stream)
    return JSONResponse(content={"filename": file.filename, "file_location": file_location})
#@router.post("/swap_face_and_enhance")
#def swap_face_and_enhance(item:Item):
#    content = item.content
#    try:
#        frame = swap.enhance_face(swap.swap_face(swap.source_face, swap.target_face, swap.base64_2_frame(content)))
#        return jsonMsg("success", swap.frame_2_base64(frame), None)
#    except Exception as e:
#        return jsonMsg("fail", None, e)
#
#@router.post("/enhance_face")
#def enhance_face(item:Item):
#    content = item.content
#    try:
#        frame = swap.enhance_face(swap.base64_2_frame(content))
#        return jsonMsg("success", swap.frame_2_base64(frame), None)
#    except Exception as e:
#        return jsonMsg("fail", None, e)
def read_video_meta_ffprobe(stream: BytesIO):
    stream.seek(0)
    # 使用 ffprobe 获取视频的元数据信息，包括分辨率和帧率
    command = [
        'ffprobe',
        '-print_format', 'json',  # 输出为 JSON 格式
        '-show_streams',  # 获取视频流的详细信息
        '-analyzeduration', '10000000',  # 增加分析时长，单位微秒（1秒=1000000微秒）
        '-probesize', '5000000',  # 增加探测数据大小
        'pipe:0'  # 从标准输入读取视频数据
    ]
    # 启动 ffprobe 进程
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 将视频二进制数据写入 ffprobe 的标准输入
    stdout, stderr = process.communicate(input=stream.getvalue())
    # 解析 ffprobe 输出的 JSON 数据
    metadata = json.loads(stdout.decode('utf-8'))
    # 获取视频的宽度、高度和帧率
    width = int(metadata['streams'][1]['width'])
    height = int(metadata['streams'][1]['height'])
    # 获取视频的帧率，注意它是以分数形式表示的，如 "30/1"
    frame_rate_str = metadata['streams'][1]['r_frame_rate']
    numerator, denominator = map(int, frame_rate_str.split('/'))
    frame_rate = numerator / denominator
    print(f"Video dimensions: {width}x{height}")
    print(f"Frame rate: {frame_rate} fps")
    # 等待 ffmpeg 子进程退出
    process.wait()
def read_video_frame_nparray(stream: BytesIO):
    stream.seek(0)
    # 使用 imageio 从内存中读取视频数据
    file_bytes = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def read_video_frame_ffmpeg(stream: BytesIO):
    stream.seek(0)
    # 使用 ffmpeg 启动一个子进程，将视频解码为原始的像素数据流
    command = [
        'ffmpeg',
        '-i', 'pipe:0',  # 从标准输入（管道）读取视频
        '-f', 'rawvideo',  # 输出为原始像素数据
        '-pix_fmt', 'rgb24',  # 输出像素格式为 rgb24 (每个像素 3 字节)
        '-analyzeduration', '10000000',  # 增加分析时长（10秒）
        '-probesize', '5000000',  # 增加探测数据大小
        'pipe:1'  # 输出到标准输出（管道）
    ]
    # 启动 ffmpeg 进程
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 启动一个线程来实时读取stderr
    import threading
    def read_stderr():
        for line in process.stderr:
            print("FFmpeg stderr:", line)

    stderr_thread = threading.Thread(target=read_stderr)
    stderr_thread.start()
    # 将二进制视频数据写入 ffmpeg 的标准输入
    process.stdin.write(stream.getvalue())
    process.stdin.flush()  # 强制刷新标准输入
    process.stdin.close()
    # 读取解码后的视频帧数据
    stdout, stderr = process.communicate()  # 等待进程完成
    # 从 stderr 获取视频的宽度和高度
    #stderr = process.stderr.read().decode('utf-8')
    width = int([line for line in stderr.split('\n') if 'Video' in line][0].split(' ')[-2].split('x')[0])
    height = int([line for line in stderr.split('\n') if 'Video' in line][0].split(' ')[-2].split('x')[1])

    print(f"Video dimensions: {width}x{height}")

    # 每帧的字节数：宽度 * 高度 * 每个像素 3 字节 (RGB)
    frame_size = width * height * 3

    # 逐帧读取并处理
    frame_count = 0
    while True:
        raw_frame = process.stdout.read(frame_size)  # 读取一帧数据

        if not raw_frame:
            break  # 如果没有数据，结束

        # 将读取的帧数据转换为 NumPy 数组
        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))  # 根据实际宽高进行调整

        # 此时 `frame` 是一帧的图像数据，可以进一步处理
        print(f"Extracted Frame {frame_count}")

        # 增加帧计数器
        frame_count += 1

    # 等待 ffmpeg 子进程退出
    process.wait()