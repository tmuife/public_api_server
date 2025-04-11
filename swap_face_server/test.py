from io import BytesIO
import cv2
import numpy as np
from pathlib import Path
import subprocess
import json
import av, av.datasets




def test_get_meta():
    stream = BytesIO()
    # 将二进制数据保存为临时文件
    with open("/Users/walter/Downloads/08123200_END_OF_EMAIL_OVERLOAD_mcclendons.mp4", "rb") as f:
        bys = f.read()

    stream.write(bys)
    stream.seek(0)
    #stream.close()
    # 使用 ffprobe 分析文件
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
    stdout, stderr = process.communicate(input=stream.read())

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
    process.wait()

def test_get_frame():
    # 使用 ffmpeg 处理文件
    command = [
        'ffmpeg',
        '-i', '/Users/walter/Downloads/embedding.mp4',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        'pipe:1'
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    #if stderr:
    #    print("FFmpeg error:", stderr.decode('utf-8'))

    print(stdout[:100])

def test_pyav():
    byteio = BytesIO()
    # 将二进制数据保存为临时文件
    with open("/Users/walter/Downloads/embedding.mp4", "rb") as f:
        bys = f.read()

    byteio.write(bys)
    byteio.seek(0)
    container = av.open(byteio, mode="r")
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = 'NONKEY'
    for frame in container.decode(stream):
        frame.to_image().save('night-sky.{:04d}.jpg'.format(frame.pts),
                              quality=80)


from moviepy.editor import AudioFileClip, VideoFileClip, CompositeVideoClip
def test_moviepy(video_file, audio_file, output_file):
    # 读取视频和音频文件
    video_clip = VideoFileClip(video_file)
    audio_clip = AudioFileClip(audio_file)
    # 获取视频和音频的时长
    video_duration = video_clip.duration
    audio_duration = audio_clip.duration
    # 裁剪音频和视频以匹配最短的时长
    if audio_duration < video_duration:
        # 如果音频更短，裁剪视频
        video_clip = video_clip.subclip(0, audio_duration)
    elif audio_duration > video_duration:
        # 如果视频更短，裁剪音频
        audio_clip = audio_clip.subclip(0, video_duration)
        # 合并音频和视频
    final_clip = video_clip.set_audio(audio_clip)
    # 写入输出文件
    final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')
    # 关闭clip的reader，释放资源
    video_clip.reader.close()
    audio_clip.reader.close()
    final_clip.reader.close_all()

def split_audio_from_video():
    # Define the input video file and output audio file
    mp4_file = "Video.mp4"
    mp3_file = "audio.mp3"
    # Load the video clip
    video_clip = VideoFileClip(mp4_file)
    # Extract the audio from the video clip
    audio_clip = video_clip.audio
    # Write the audio to a separate file
    audio_clip.write_audiofile(mp3_file)
    # Close the video and audio clips
    audio_clip.close()
    video_clip.close()
    print("Audio extraction successful!")

#test_get_frame()
test_pyav()