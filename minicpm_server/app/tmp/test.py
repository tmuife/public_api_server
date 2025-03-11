import threading
import time
import concurrent.futures

# 假定文件的路径
CONFIG_FILE = "concurrency.txt"

# 用于存放所有线程的引用
threads = []
stop_event = threading.Event()
executor = concurrent.futures.ThreadPoolExecutor()  # 全局线程池
data_lock = threading.Lock()  # 用于保护全局数据的锁
global_dict = {}  # 全局共享的字典变量

# 任务函数：不停地运行
def worker(task_id):
    try:
        while not stop_event.is_set():
            with data_lock:  # 确保修改全局字典时是线程安全的
                global_dict[task_id] = global_dict.get(task_id, 0) + 1
            #print(f"Thread {task_id} is running...")
            print("Current Concurrent is -> ",len(threads))
            time.sleep(1)  # 模拟任务工作
    except Exception as e:
        print(f"Thread {task_id} 出现错误：{e}")
    finally:
        #with data_lock:
        #    if task_id in global_dict:
        #        del global_dict[task_id]
        print(f"Thread {task_id} 正在退出...")  # 确认退出


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
    global threads
    for i in range(current_count, new_count):
        future = executor.submit(worker, i)
        threads.append(future)


# 主控制函数
def main():
    global threads
    current_thread_count = 0  # 当前运行的线程数量

    try:
        while True:
            # 读取当前配置文件中的线程数量
            target_thread_count = read_thread_count_from_file()

            if target_thread_count > current_thread_count:
                # 如果目标数量增加，启动新的线程
                print(f"增加 {target_thread_count - current_thread_count} 个线程...")
                start_new_threads(target_thread_count, current_thread_count)
                current_thread_count = target_thread_count

            elif target_thread_count < current_thread_count:
                # 如果目标数量减少
                print(f"检测到减少线程请求。当前线程数量：{current_thread_count}，目标线程数量：{target_thread_count}")
                print("减少线程功能暂未实现！")

            # 每隔 3 秒检查一次配置文件
            with data_lock:
                print("global_dict 内容：", global_dict)
            time.sleep(3)

    except KeyboardInterrupt:
        print("程序终止中...")
        stop_event.set()  # 停止所有线程
        executor.shutdown(wait=True)  # 等待所有线程安全退出
        print("所有线程已安全退出。")
    finally:
        stop_event.set()  # 停止所有线程
        executor.shutdown(wait=True)  # 等待所有线程安全退出
        print("所有线程已安全退出。")
        # 打印最终的全局字典内容
        with data_lock:
            print("最终的 global_dict 内容：", global_dict)


if __name__ == "__main__":
    main()
