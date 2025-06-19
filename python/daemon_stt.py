import os
import time
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_PATH = "models/phowhisper_multistage"
AUDIO_DIR = "data/audio_chunks"
TXT_DIR = "data/transcriptions"

# Load model và processor
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
processor = WhisperProcessor.from_pretrained(MODEL_PATH)

def transcribe(wav_path):
    # Load audio
    wav, sr = torchaudio.load(wav_path)
    # Chuẩn bị input cho model
    inputs = processor(wav.squeeze().numpy(), sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features
    # Dùng prompt mặc định của Whisper: ngôn ngữ tiếng Việt, task transcribe
    forced_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
    # Generate kết quả
    ids = model.generate(input_features, forced_decoder_ids=forced_ids)
    # Decode text
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

if __name__ == "__main__":
    os.makedirs(TXT_DIR, exist_ok=True)
    while True:
        for f in os.listdir(AUDIO_DIR):
            if not f.endswith(".wav"):
                continue
            src = os.path.join(AUDIO_DIR, f)
            text = transcribe(src)
            dst = os.path.join(TXT_DIR, f.replace(".wav", ".txt"))
            with open(dst, "w") as out:
                out.write(text)
            os.remove(src)
            print(f"✔️  Transcribed → {dst}")
        time.sleep(1)
