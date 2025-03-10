import base64


def wav_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    # 将音频文件内容编码为 base64 字符串
    base64_string = base64.b64encode(audio_bytes).decode("utf-8")
    return base64_string


# 示例：读取一个 WAV 文件
base64_audio_string = wav_to_base64('/Users/walter/Downloads/speech_orig.wav')
print(base64_audio_string)  # 打印前100个字符做预览