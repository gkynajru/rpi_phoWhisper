import pyaudio
import numpy as np
import pvporcupine
import time
import os
import queue
import threading
import torch
import sounddevice as sd
import librosa
import soundfile as sf  # Thay thế sd.write_wav bằng sf.write

# Configuration
DEVICE_INDEX = 0
SAMPLE_RATE = 48000  # Thiết bị thu âm ở 48 kHz
PORCUPINE_SAMPLE_RATE = 16000  # Porcupine yêu cầu 16 kHz
FRAME_LENGTH = 512   # Porcupine frame length ở 16 kHz
BUFFER_SECONDS = 5
PRE_BUFFER_SECONDS = 1
VAD_THRESHOLD = 0.02
SILENCE_DURATION = 1.0
ACCESS_KEY = '1yGPnMC26F0ODv6AjGLjy7T0VRBYt4AtR1tyiyvoRl0Uc9dSdBKhMw=='  # Thay thế bằng Access Key của bạn
WAKEWORD = "computer"
WAKEWORD_MODEL_PATH = "models/hellopi_wakeword.ppn"
VAD_MODEL_PATH = "models/silero_vad.jit"
AUDIO_CHUNKS_DIR = "data/audio_chunks"
os.makedirs(AUDIO_CHUNKS_DIR, exist_ok=True)

# Shared queue for audio data
audio_queue = queue.Queue()

# Suppress ALSA warnings
os.environ["ALSA_LOG_LEVEL"] = "none"

# Initialize PyAudio
pa = pyaudio.PyAudio()

def list_audio_devices():
    print("Available audio devices:")
    for i in range(pa.get_device_count()):
        device_info = pa.get_device_info_by_index(i)
        print(f"Index {i}: {device_info['name']}, Input Channels: {device_info['maxInputChannels']}")

list_audio_devices()

# Initialize Porcupine
porcupine = None
try:
    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keywords=[WAKEWORD],
        sensitivities=[0.8]
    )
    print("Porcupine initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Porcupine: {e}")
    print("Please check your access key and model path.")
    if not os.path.exists(WAKEWORD_MODEL_PATH):
        print(f"Model path {WAKEWORD_MODEL_PATH} does not exist.")
    exit(1)

# Open audio stream at 48 kHz
try:
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=int(FRAME_LENGTH * (SAMPLE_RATE / PORCUPINE_SAMPLE_RATE)),
        input_device_index=DEVICE_INDEX
    )
except Exception as e:
    print(f"Failed to open audio stream: {e}")
    porcupine.delete()
    pa.terminate()
    exit(1)

# Initialize SileroVAD
try:
    print("🔁 Loading Silero VAD model...")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
    (get_speech_timestamps, _, read_audio, _, _) = utils
except Exception as e:
    print(f"Failed to initialize SileroVAD: {e}")
    porcupine.delete()
    stream.close()
    pa.terminate()
    exit(1)

# Buffer - moved to global scope
buffer = np.zeros(int(SAMPLE_RATE * BUFFER_SECONDS), dtype=np.int16)
pre_buffer = np.zeros(int(SAMPLE_RATE * PRE_BUFFER_SECONDS), dtype=np.int16)

# Function to resample audio data
def resample_audio(audio_data, original_sr, target_sr):
    return librosa.resample(audio_data.astype(np.float32), orig_sr=original_sr, target_sr=target_sr)

def audio_processing():
    global buffer, pre_buffer  # Added global keyword to avoid re-initialization
    recording = False
    silence_start = None
    full_audio = []
    
    while True:
        try:
            # Đọc dữ liệu từ luồng ở 48 kHz
            data_48k = np.frombuffer(stream.read(int(FRAME_LENGTH * (SAMPLE_RATE / PORCUPINE_SAMPLE_RATE)), exception_on_overflow=False), dtype=np.int16)
            
            # Resample dữ liệu từ 48 kHz xuống 16 kHz cho Porcupine
            data_16k = resample_audio(data_48k, SAMPLE_RATE, PORCUPINE_SAMPLE_RATE)
            data_16k = (data_16k * 32768).astype(np.int16)  # Chuyển đổi về int16
            
            if not recording:
                # Update buffers (ở 48 kHz)
                buffer = np.roll(buffer, -len(data_48k))
                buffer[-len(data_48k):] = data_48k
                pre_buffer = np.roll(pre_buffer, -len(data_48k))
                pre_buffer[-len(data_48k):] = data_48k
                
                # Detect wake word với dữ liệu 16 kHz
                keyword_index = porcupine.process(data_16k)
                if keyword_index >= 0:
                    print(f"Detected wake word: {WAKEWORD}")
                    recording = True
                    full_audio = list(pre_buffer) + list(buffer)
                    # Save debug WAV (ở 48 kHz)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    sf.write(f"{AUDIO_CHUNKS_DIR}/audio_{timestamp}.wav", np.array(full_audio), SAMPLE_RATE)
                    print(f"Saved {AUDIO_CHUNKS_DIR}/audio_{timestamp}.wav")
            else:
                # Continue recording (ở 48 kHz)
                full_audio.extend(data_48k)
                
                # Resample full_audio từ 48 kHz xuống 16 kHz cho SileroVAD
                full_audio_16k = resample_audio(np.array(full_audio), SAMPLE_RATE, 16000)
                
                # Use SileroVAD
                wav = read_audio(full_audio_16k, sampling_rate=16000)
                speech = get_speech_timestamps(wav, model, sampling_rate=16000)
                
                if speech:
                    total_speech = sum([s["end"] - s["start"] for s in speech]) / 16000
                    print(f"🗣️ Speech detected: {total_speech:.2f} seconds")
                    silence_start = None
                else:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        audio_queue.put(np.array(full_audio, dtype=np.int16))
                        print("Audio segment sent to queue")
                        recording = False
                        full_audio = []
                        silence_start = None
                
        except Exception as e:
            print(f"Audio processing error: {e}")

def main():
    audio_thread = threading.Thread(target=audio_processing)
    audio_thread.daemon = True
    audio_thread.start()
    
    print(f"Audio daemon started, listening for '{WAKEWORD}'...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping audio daemon...")
        porcupine.delete()
        stream.stop_stream()
        stream.close()
        pa.terminate()

if __name__ == "__main__":
    main()
