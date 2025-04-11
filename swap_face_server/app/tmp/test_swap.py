import time
import datetime
from typing import Any, List
import cv2
import numpy
import os
from decouple import config
from dbutil import mysql as mydb
from app.services.swap_face_service import Swap

# Initialize the model and request processor
swap = Swap()
#-ss 01:34:40 -to 01:35:40

def test_mac_platform_video():
    base_dir = config("base_dir")
    db = mydb()
    df = db.query("select raw from stop where id=7")
    source_face_base = df.to_dict('records')[0]["raw"]
    now = datetime.datetime.now()
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')  # 文件扩展名.mp4
    swap.set_source_face(swap.base64_2_frame(source_face_base))
    swap.set_target_face(cv2.imread(os.path.join(base_dir,'target_face.jpg')))
    cap = cv2.VideoCapture(os.path.join(base_dir,"raw.mp4"))
    _out = cv2.VideoWriter(os.path.join(base_dir,'convert_{}.mp4'.format(str(now).replace(":", ''))), fourcc, cap.get(cv2.CAP_PROP_FPS),
                           (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while cap.isOpened():
        success, input_frame = cap.read()
        if not success:
            break
        if success:
            # input_frame = cv2.imread("/Users/walter/Downloads/me_and_other.jpg")
            start_time = time.time()
            try:
                out_frame = swap.swap_face(swap.source_face,swap.target_face,input_frame)
            except Exception as e:
                cv2.imwrite(os.path.join(base_dir, 'null_face.jpg'), input_frame)
                print(e)
                out_frame = input_frame
            cv2.imwrite(os.path.join(base_dir,'swap_face.jpg'), out_frame)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time} seconds")
            #cv2.namedWindow('Live', cv2.WINDOW_NORMAL)
            # Resize the window (width, height)
            #cv2.resizeWindow('Live', 800, 640)
            #cv2.imshow('Live', tool.base64_2_frame(out["data"]))
            _out.write(out_frame)

        if cv2.waitKey(1) == 27:
            break
    _out.release()
    cap.release()
    cv2.destroyAllWindows()

test_mac_platform_video()