import time
from datetime import datetime
from decouple import config
import os
import requests
import json
import asyncio
from PIL import Image
import base64
import io
import threading

def call_image_query(batch, name, baseimg):
    print(f"Run batch {batch},thread name {name}")
    api_key = config("API_KEY")
    url = config("minicpm_url")+"image_query"

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
    _now = datetime.now()
    print("当前时间是：", _now)

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


if __name__ == "__main__":

    image_content = image_to_base64(image_path=config("test_image_path"))
    # 获取当前时间
    now = datetime.now()
    print("当前时间是：", now)
    for run_batch in range(0,config("run_times" ,cast=int)):
        with open("concurrency.txt", "r", encoding="utf-8") as file:
            concurrency = file.read()
        print(concurrency)
        thread_list: [threading.Thread] = []
        for i in range(0,int(concurrency)):
            thread_list.append(threading.Thread(target=call_image_query, args=(str(run_batch), "Thread-"+str(i), image_content)))
        for thread in thread_list:
            # 启动线程
            thread.start()
            # 等待所有线程完成
            #thread.join()
        print("所有线程执行完毕。")
