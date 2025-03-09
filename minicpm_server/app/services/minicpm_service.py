import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import math
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
import tempfile
import librosa
import soundfile as sf

class CPMService:
    def __init__(self, model_id = "openbmb/MiniCPM-o-2_6", device = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # load omni model default, the default init_vision/init_audio/init_tts is True
        # if load vision-only model, please set init_audio=False and init_tts=False
        # if load audio-only model, please set init_vision=False
        self.model: AutoModel = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation='sdpa', # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=True
        )
        self.model = self.model.eval().cuda()
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model.init_tts()

    async def chat_omni(self, msgs:list[dict],generate_audio:bool, output_audio_path:bool):
        res = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.5,
            max_new_tokens=4096,
            omni_input=True,  # please set omni_input=True when omni inference
            use_tts_template=True,
            generate_audio=generate_audio,
            output_audio_path=output_audio_path,
            max_slice_nums=1,
            use_image_id=False,
            return_dict=True
        )
        return res

    async def get_video_chunk_content(self, video_path, flatten=True):
        video = VideoFileClip(video_path)
        print('video_duration:', video.duration)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
            temp_audio_file_path = temp_audio_file.name
            video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
            audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
        num_units = math.ceil(video.duration)
        # 1 frame + 1s audio chunk
        contents= []
        for i in range(num_units):
            frame = video.get_frame(i+1)
            image = Image.fromarray((frame).astype(np.uint8))
            audio = audio_np[sr*i:sr*(i+1)]
            if flatten:
                contents.extend(["<unit>", image, audio])
            else:
                contents.append(["<unit>", image, audio])
        return contents

