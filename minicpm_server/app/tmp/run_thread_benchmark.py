import threading
import time
import concurrent.futures
import random
from decouple import config
import requests
import json
from PIL import Image
import base64
import io
from datetime import datetime
import time

CONFIG_FILE = "concurrency.txt"

threads = []
stop_event = threading.Event()
executor = concurrent.futures.ThreadPoolExecutor()  # 全局线程池
data_lock = threading.Lock()  # 用于保护全局数据的锁
global_dict = {}  # 全局共享的字典变量
stop_signals = {}  # 用于控制单个线程停止的信号

question = '''Task: Analyze the input image and perform the following tasks:
Text Description: 
Generate a detailed description of the image, focusing on main objects and their activities. Follow the instructions strictly:
 
Your main focus is on main objects in the scene, including people, animals, cars etc.
Carefully count the main objects.
Describe the appearance and physical characteristics of the objects, including their colors, attire, and other visual attributes
Describe their actions/movements reactions (if any)
Output a concise description 15 words max in English language.
Object Detection: Detect and localize only the following objects in the image:
Person
Face (ensure high confidence for facial detection)
Vehicle
Package
Animal (if any animal is detected, return "Animal" as the result, regardless of the specific type)
Output Json
Format:
{"text": "<Generated text description based on the requirements in %s language>","object": [{"class": "","confidence": <confidence score between 0 and 1>,}]}
This is an example you should follow the format but not the result:
{"text": "A person holding a baby in an indoor setting with furniture and plants.", "object": [{"class": "Person", "confidence": 0.9}, {"class": "Vehicle", "confidence": 0.8}, {"class": "Face", "confidence": 0.7}]}'''


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

def call_query(api_url="image_query") -> int:
    #global image_content
    cost = 0
    try:
        api_key = config("API_KEY")
        url = config("minicpm_url")+api_url
        # 设置请求头
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            "access_token": api_key
        }
        data = {"content": image_content, "question": question}
        # 发送 GET 请求
        time_start = datetime.now()
        response = requests.post(url, headers=headers, json=data)
        # 输出返回的结果
        if response.status_code == 200:
            jobj = json.loads(response.json())
            if type(jobj) == str:
                jobj = json.loads(jobj)
            # print(jobj)
            if len(str(jobj["data"]["text"])) > 0:
                time_end = datetime.now()
                time_difference = time_end - time_start
                cost = int(time_difference.total_seconds())
        else:
            print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
    except Exception as e:
        pass
    finally:
        return cost

# 任务函数：不停地运行并修改全局字典
def worker(task_id, stop_signal):
    try:
        while not stop_event.is_set() and not stop_signal.is_set():
            cost = call_query(api_url="image_query")
            if cost > 0:
                with data_lock:
                    global_dict[str(len(threads))+"-runtimes"] = global_dict.get(str(len(threads))+"-runtimes", 0) + 1  # 增加计数
                    global_dict[str(len(threads))+"-costtime"] = global_dict.get(str(len(threads))+"-costtime", 0) + cost  # 增加计数
            else:
                print("No cost time founded!!!!!!!!")
            #global_dict[task_id] = global_dict.get(task_id, 0) + 1  # 增加计数
            #print(f"Thread {task_id} is running... Count: {global_dict[task_id]}")
            #print(f"Thread {task_id} is running...")
            print("Current Concurrent is -> ",len(threads))
            #time.sleep(1)
    except Exception as e:
        print(f"Thread {task_id} 出现错误：{e}")
    finally:
        #with data_lock:
        #    global_dict[task_id] = f"Thread {task_id} 已退出"  # 标记为已退出
        print(f"Thread {task_id} 正在退出...")


# 检查配置文件中的线程数量
def read_thread_count_from_file():
    try:
        with open(CONFIG_FILE, "r") as f:
            count = int(f.read().strip())
            return count
    except Exception as e:
        print(f"读取配置文件出错：{e}")
        return 5  # 如果出错，默认返回 5


# 启动新的线程任务
def start_new_threads(new_count, current_count):
    global threads, stop_signals
    for i in range(current_count, new_count):
        stop_signal = threading.Event()  # 为每个线程创建一个停止信号
        stop_signals[i] = stop_signal
        future = executor.submit(worker, i, stop_signal)
        threads.append(future)


# 停止多余的线程
def stop_threads(current_count, target_count):
    global stop_signals
    for i in range(target_count, current_count):
        if i in stop_signals:
            print(f"正在停止线程 {i}...")
            stop_signals[i].set()  # 设置对应线程的停止信号
            del stop_signals[i]  # 移除已停止的信号

def run_schedule():
    #schedule = list(range(2, 100, 2))
    schedule = [1 * (2 ** i) for i in range(config("max_concurrent", cast=int))]
    global threads
    current_thread_count = 0
    try:
        for target_thread_count in schedule:
            if target_thread_count > current_thread_count:
                print(f"增加 {target_thread_count - current_thread_count} 个线程...")
                start_new_threads(target_thread_count, current_thread_count)
                current_thread_count = target_thread_count
            elif target_thread_count < current_thread_count:
                print(f"减少 {current_thread_count - target_thread_count} 个线程...")
                stop_threads(current_thread_count, target_thread_count)
                current_thread_count = target_thread_count
            time.sleep(config("continue_time", cast=int))
    except KeyboardInterrupt:
        print("程序终止中...")
        stop_event.set()  # 停止所有线程的全局信号
        executor.shutdown(wait=True)
        print("所有线程已安全退出。")
        with data_lock:
            print("最终的 global_dict 内容：", global_dict)
    finally:
        stop_event.set()  # 停止所有线程
        executor.shutdown(wait=True)  # 等待所有线程安全退出
        print("所有线程已安全退出。")
        # 打印最终的全局字典内容
        with data_lock:
            print("最终的 global_dict 内容：", global_dict)
            for _concurrent in schedule:
                avg_time = round(float(global_dict[str(_concurrent) + "-costtime"])/float(global_dict[str(_concurrent) + "-runtimes"]), 2)
                print(f"When concurrent is {_concurrent} the avg cost time is {avg_time}")

# 主控制函数
def main():
    global threads
    current_thread_count = 0
    try:
        while True:
            target_thread_count = read_thread_count_from_file()
            if target_thread_count > current_thread_count:
                print(f"增加 {target_thread_count - current_thread_count} 个线程...")
                start_new_threads(target_thread_count, current_thread_count)
                current_thread_count = target_thread_count
            elif target_thread_count < current_thread_count:
                print(f"减少 {current_thread_count - target_thread_count} 个线程...")
                stop_threads(current_thread_count, target_thread_count)
                current_thread_count = target_thread_count
            time.sleep(3)

    except KeyboardInterrupt:
        print("程序终止中...")
        stop_event.set()  # 停止所有线程的全局信号
        executor.shutdown(wait=True)
        print("所有线程已安全退出。")
        with data_lock:
            print("最终的 global_dict 内容：", global_dict)
    finally:
        stop_event.set()  # 停止所有线程
        executor.shutdown(wait=True)  # 等待所有线程安全退出
        print("所有线程已安全退出。")
        # 打印最终的全局字典内容
        with data_lock:
            print("最终的 global_dict 内容：", global_dict)

if __name__ == "__main__":
    #main()
    global image_content
    image_content = image_to_base64(config("test_image_path"))
    run_schedule()
