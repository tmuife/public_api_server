import os
from moviepy.editor import AudioFileClip, VideoFileClip, CompositeVideoClip

def merge_audio_video(video_file, audio_file, output_file):
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


def test_clip(file_name='test.mp4'):
    temp_dir = "/Users/walter/temp"
    # 读取原视频
    video = VideoFileClip(os.path.join(temp_dir,file_name))
    # 剪切视频 (从第 5 秒 到 第 15 秒)
    subclip = video.subclip(5, 15)
    # 保存剪切后的视频 (保留音频)
    subclip.write_videofile(os.path.join(temp_dir,"out.mp4"), codec="libx264", audio_codec="aac")

def test_get_frames(file_name='test.mp4'):
    temp_dir = "/Users/walter/temp"
    # 读取原视频
    video = VideoFileClip(os.path.join(temp_dir, file_name))
    total_frames = int(video.fps * video.duration)
    print(f"Total frames: {total_frames}")

    # 创建保存帧的文件夹
    output_dir = "frames_output"
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(video.iter_frames(fps=video.fps)):
        frame_image_path = os.path.join(output_dir, f"frame_{i}.png")

        # 使用 Pillow 保存图片
        from PIL import Image
        image = Image.fromarray(frame)
        image.save(frame_image_path)

    print(f"Total frames saved: {i + 1}")

