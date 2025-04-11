import time, base64
import datetime
import math
from tqdm.asyncio import tqdm as tqdm_async
import asyncio
from typing import Any, List
import requests
import cv2
import numpy
import os
from decouple import config
from pathlib import Path
import json
import subprocess
from dbutil import mysql as mydb
from app.services.swap_face_service import Swap
from decouple import config

# Initialize the model and request processor
#swap = Swap()

class CVSWAP:
    def __init__(self, video_path):
        self.swap_service = Swap()
        self.fps = 18
        self.video_path = video_path
        video_parent_path = Path(self.video_path).parent
        self.frame_dir = os.path.join(video_parent_path, "frames")
        self.parallel = 10
        #self.db = mydb()

    def init_swap_source_and_target(self):
        if hasattr(self, 'db'):
            df = self.db.query("select raw from stop where id=7")
            source_face_base = df.to_dict('records')[0]["raw"]
            self.swap_service.set_source_face(self.base_2_frame(source_face_base))


    def analysis_video(self):
        # 使用 ffprobe 分析文件
        command = [
            'ffprobe',
            '-i',self.video_path,
            '-print_format', 'json',  # 输出为 JSON 格式
            '-show_streams',  # 获取视频流的详细信息
            '-analyzeduration', '10000000',  # 增加分析时长，单位微秒（1秒=1000000微秒）
            '-probesize', '5000000',  # 增加探测数据大小
        ]
        # 启动 ffprobe 进程
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 将视频二进制数据写入 ffprobe 的标准输入
        #stdout, stderr = process.communicate(input=stream.read())
        stdout, stderr = process.communicate()

        # 解析 ffprobe 输出的 JSON 数据
        metadata = json.loads(stdout.decode('utf-8'))
        # 获取视频的宽度、高度和帧率
        width = int(metadata['streams'][0]['width'])
        height = int(metadata['streams'][0]['height'])
        # 获取视频的帧率，注意它是以分数形式表示的，如 "30/1"
        frame_rate_str = metadata['streams'][0]['r_frame_rate']
        numerator, denominator = map(int, frame_rate_str.split('/'))
        frame_rate = numerator / denominator
        print(f"Video dimensions: {width}x{height}")
        print(f"Frame rate: {frame_rate} fps")
        self.fps = int(frame_rate)
        process.wait()

    def frame_2_base(self, frame):
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return img_base64

    def base_2_frame(self, base_str):
        img_data = base64.b64decode(base_str)
        # Step 3: Convert bytes into a NumPy array
        img_array = numpy.frombuffer(img_data, dtype=numpy.uint8)
        # Step 4: Decode the NumPy array into an OpenCV image
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return image

    def extract_frames(self):
        if not os.path.exists(self.frame_dir):
            os.makedirs(self.frame_dir)
        command = [
            'ffmpeg',
            '-i',self.video_path,
            '-vf', 'fps=%s' % (str(self.fps), ),
            os.path.join(self.frame_dir,'origin_frame_%05d.jpg')
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 将视频二进制数据写入 ffprobe 的标准输入
        # stdout, stderr = process.communicate(input=stream.read())
        stdout, stderr = process.communicate()
        process.wait()

    async def split_frames_for_tasks(self):
        result = []
        for f in os.listdir(self.frame_dir):
            result.append(os.path.join(self.frame_dir,f))
        print("total_frames:",len(result))
        parallel = int(config("parallel"))
        _max_batch_size = int(math.ceil(len(result)/parallel))
        batches = [
            result[i: i + _max_batch_size]
            for i in range(0, len(result), _max_batch_size)
        ]
        async def local_call(frame_path_list):
            for path in frame_path_list:
                #await asyncio.sleep(0.1)
                swapped_frame = await self.swap_service.swap_face(source_face=None, target_face=None, temp_frame=cv2.imread(path))
                file_name = Path(path).name
                cv2.imwrite(os.path.join(self.frame_dir,file_name.replace("origin_", "swapped_")), swapped_frame)
            pbar.update(1)
            return 1
        async def remote_call(frame_path_list):
            # 设置请求头
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json',
                "access_token": config("api_key")
            }
            for path in frame_path_list:
                #await asyncio.sleep(0.1)
                origin_frame = cv2.imread(path)
                swapped_frame = origin_frame
                data = {"content": self.frame_2_base(origin_frame)}
                response = requests.post(config("url"), headers=headers, json=data)
                # 输出返回的结果
                if response.status_code == 200:
                    j_obj = json.loads(response.text)
                    if type(j_obj) == str:
                        j_obj = json.loads(j_obj)
                        swapped_frame = self.base_2_frame(j_obj["data"])
                else:
                    print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
                file_name = Path(path).name
                cv2.imwrite(os.path.join(self.frame_dir,file_name.replace("origin_", "swapped_")), swapped_frame)
            pbar.update(1)
            return 1
        start_time = time.time()
        swap_tasks = [local_call(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(swap_tasks), desc="Generating swapping", unit="batch"
        )
        embeddings_list = await asyncio.gather(*swap_tasks)
        print(embeddings_list)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time} seconds")


#c = CVSWAP(video_path="/Users/walter/Downloads/bilibili/test/West-vs-non-West.mp4")
#c.analysis_video()
## c.extract_frames()
#asyncio.run(c.split_frames_for_tasks())
import pytest
@pytest.mark.asyncio
async def test_face_detect():
    c = CVSWAP(video_path="/Users/walter/Downloads/bilibili/test/West-vs-non-West.mp4")
    c.analysis_video()
    # c.extract_frames()
    await c.split_frames_for_tasks()