import numpy as np
import cv2
from PIL import Image
import time
import os
from datetime import datetime
from decouple import config
from dbutil import mysql as mydb
from app.services.swap_face_service import Swap
import asyncio
import logging
logger = logging.getLogger("DangerRecognizer")
# 简单测试代码

async def swap_function():
    logging.basicConfig(level=logging.INFO)
    base_dir = config("base_dir")
    swap = Swap()
    db = mydb()
    df = db.query("select raw from stop where id=7")
    source_face_base = df.to_dict('records')[0]["raw"]
    now = datetime.now()
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')  # 文件扩展名.mp4
    swap.set_source_face(swap.base64_2_frame(source_face_base))
    swap.set_target_face(cv2.imread(os.path.join(base_dir,'target_face.jpg')))
    cap = cv2.VideoCapture(os.path.join(base_dir,"raw.mp4"))
    _out = cv2.VideoWriter(os.path.join(base_dir,'convert_{}.mp4'.format(str(now).replace(":", ''))), fourcc, cap.get(cv2.CAP_PROP_FPS),
                           (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    video_path = os.path.join(base_dir,'test.mp4')
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()
    # Loop through the video frame by frame
    while True:
        ret, frame = cap.read()  # Read a frame
        if not ret:
            print("End of video or can't read the frame.")
            break
        try:
            out_frame = await swap.swap_face(None, None, frame)
        except Exception as e:
            #cv2.imwrite(os.path.join(base_dir, 'null_face.jpg'), input_frame)
            print(e)
            out_frame = frame
        # Show the frame in a window
        cv2.imshow('Video Frame', out_frame)
        # Press 'q' to quit early
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()
    # Walter test end

if __name__ == "__main__":
    asyncio.run(swap_function())