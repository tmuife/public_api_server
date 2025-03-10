import time

from decouple import config
import os
import requests
import json
import asyncio
from PIL import Image
import base64
import io

async def call_image_query(baseimg):
    api_key = config("API_KEY")
    url = config("minicpm_url")

    # 设置请求头
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        "access_token": api_key
    }

    data = {"content": baseimg, "question": "Provide a description of the image in English."}
    # 发送 GET 请求
    response = requests.post(url, headers=headers, json=data)
    # 输出返回的结果
    if response.status_code == 200:
        jobj = json.loads(response.json())
        if type(jobj) == str:
            jobj = json.loads(jobj)
        # print(jobj)
        print(jobj["data"]["text"])
    else:
        print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
def image_to_base64(image_path: str) -> str:
    # 打开图片
    with Image.open(image_path) as img:
        # 创建一个字节流
        buffered = io.BytesIO()
        # 将图片保存到字节流中 (以 PNG 格式保存，确保无损)
        img.save(buffered, format="PNG")
        # 获取字节流的二进制内容
        img_bytes = buffered.getvalue()
        # 编码为 base64 字符串
        base64_string = base64.b64encode(img_bytes).decode("utf-8")
        return base64_string
image_content = image_to_base64(image_path="/Users/walter/Downloads/猿人/4.jpg")
asyncio.run(call_image_query(baseimg=image_content))