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
import soundfile as sf
from scipy import signal

# Configuration
DEVICE_INDEX = 1  # FIX 1: Đổi từ 0 sang 1 (pulse device)
SAMPLE_RATE = 48000  # Thiết bị thu âm ở 48 kHz
PORCUPINE_SAMPLE_RATE = 16000  # Porcupine yêu cầu 16 kHz
FRAME_LENGTH = 512   # Porcupine frame length ở 16 kHz
BUFFER_SECONDS = 5
PRE_BUFFER_SECONDS = 1
VAD_THRESHOLD = 0.02
SILENCE_DURATION = 1.0
ACCESS_KEY = '1yGPnMC26F0ODv6AjGLjy7T0VRBYt4AtR1tyiyvoRl0Uc9dSdBKhMw=='
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

# FIX 2: Thêm function cleaning audio data
def clean_audio_data(audio_data):
    """Clean audio data - remove NaN, inf, normalize"""
    # Remove NaN and inf values
    audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip to valid int16 range
    audio_data = np.clip(audio_data, -32768, 32767)
    
    return audio_data

# FIX 3: Thêm simple downsampling method
def simple_downsample(audio_data, factor=3):
    """Simple downsampling by taking every nth sample"""
    try:
        # Clean data first
        audio_data = clean_audio_data(audio_data)
        return audio_data[::factor]
    except Exception as e:
        print(f"Simple downsample error: {e}")
        return None

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
        sensitivities=[0.95]  # FIX 4: Tăng từ 0.8 lên 0.95
    )
    print("Porcupine initialized successfully.")
    print(f"Frame length: {porcupine.frame_length}, Sample rate: {porcupine.sample_rate}")
except Exception as e:
    print(f"Failed to initialize Porcupine: {e}")
    print("Please check your access key and model path.")
    if not os.path.exists(WAKEWORD_MODEL_PATH):
        print(f"Model path {WAKEWORD_MODEL_PATH} does not exist.")
    exit(1)

# Open audio stream at 48 kHz
try:
    # FIX 5: Sử dụng buffer size chính xác
    buffer_size = porcupine.frame_length * 3  # 512 * 3 = 1536
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=buffer_size,  # Sử dụng buffer size chính xác
        input_device_index=DEVICE_INDEX
    )
    print(f"Audio stream opened with buffer size: {buffer_size}")
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
    print("✅ SileroVAD loaded successfully")
except Exception as e:
    print(f"Failed to initialize SileroVAD: {e}")
    porcupine.delete()
    stream.close()
    pa.terminate()
    exit(1)

# Buffer - moved to global scope
buffer = np.zeros(int(SAMPLE_RATE * BUFFER_SECONDS), dtype=np.int16)
pre_buffer = np.zeros(int(SAMPLE_RATE * PRE_BUFFER_SECONDS), dtype=np.int16)

# FIX 6: Function resample được cải thiện với fallback
def resample_audio(audio_data, original_sr, target_sr):
    """Resample với fallback methods"""
    try:
        # Method 1: Simple downsampling (fast and stable)
        if original_sr == 48000 and target_sr == 16000:
            return simple_downsample(audio_data, factor=3)
        
        # Method 2: Librosa fallback
        clean_data = clean_audio_data(audio_data.astype(np.float32))
        resampled = librosa.resample(clean_data, orig_sr=original_sr, target_sr=target_sr)
        return clean_audio_data(resampled)
    except Exception as e:
        print(f"Resample error: {e}")
        # Method 3: Last resort - simple decimation
        factor = int(original_sr / target_sr)
        return clean_audio_data(audio_data[::factor])

def audio_processing():
    global buffer, pre_buffer
    recording = False
    silence_start = None
    full_audio = []
    frame_count = 0
    detection_count = 0
    
    # FIX 7: Sử dụng buffer size chính xác
    chunk_size = porcupine.frame_length * 3  # 1536 cho 48kHz
    
    while True:
        try:
            # Đọc dữ liệu từ luồng ở 48 kHz
            data_48k = np.frombuffer(
                stream.read(chunk_size, exception_on_overflow=False), 
                dtype=np.int16
            )
            
            # FIX 8: Clean audio data trước khi processing
            data_48k = clean_audio_data(data_48k)
            
            # Debug volume mỗi 100 frames
            if frame_count % 100 == 0:
                volume = np.sqrt(np.mean(data_48k.astype(np.float32)**2))
                max_val = np.max(np.abs(data_48k))
                print(f"Audio Frame {frame_count}: Volume={volume:.2f}, Max={max_val}")
            
            # FIX 9: Sử dụng simple downsampling cho wake word detection
            data_16k = simple_downsample(data_48k, factor=3)
            
            if data_16k is not None and len(data_16k) >= porcupine.frame_length:
                # Lấy đúng frame length cho Porcupine
                frame_16k = data_16k[:porcupine.frame_length].astype(np.int16)
            else:
                # Fallback: resample bằng librosa
                data_16k = resample_audio(data_48k, SAMPLE_RATE, PORCUPINE_SAMPLE_RATE)
                if len(data_16k) >= porcupine.frame_length:
                    frame_16k = data_16k[:porcupine.frame_length].astype(np.int16)
                else:
                    frame_count += 1
                    continue
            
            if not recording:
                # Update buffers (ở 48 kHz)
                buffer = np.roll(buffer, -len(data_48k))
                buffer[-len(data_48k):] = data_48k
                pre_buffer = np.roll(pre_buffer, -len(data_48k))
                pre_buffer[-len(data_48k):] = data_48k
                
                # FIX 10: Detect wake word với frame đã cleaned
                try:
                    keyword_index = porcupine.process(frame_16k)
                    if keyword_index >= 0:
                        detection_count += 1
                        print(f"\n🎉 WAKE WORD DETECTED! #{detection_count}")
                        print(f"Detected wake word: {WAKEWORD}")
                        
                        recording = True
                        audio_processing.recording_start_time = time.time()  # Track recording start
                        full_audio = list(pre_buffer) + list(buffer)
                        
                        # Save debug WAV (ở 48 kHz)
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        sf.write(f"{AUDIO_CHUNKS_DIR}/audio_{timestamp}.wav", np.array(full_audio), SAMPLE_RATE)
                        sf.write(f"{AUDIO_CHUNKS_DIR}/wake_frame_{timestamp}.wav", frame_16k, PORCUPINE_SAMPLE_RATE)
                        print(f"Saved {AUDIO_CHUNKS_DIR}/audio_{timestamp}.wav")
                        
                        # FIX 11: Thêm delay để tránh multiple detections
                        time.sleep(0.5)
                        
                except Exception as e:
                    print(f"Porcupine process error: {e}")
                    
            else:
                # Continue recording (ở 48 kHz)
                full_audio.extend(data_48k)
                
                # VAD processing - FIX: Cải thiện silence detection
                try:
                    # Chỉ check VAD mỗi 0.5 giây để tránh spam
                    current_time = time.time()
                    if not hasattr(audio_processing, 'last_vad_check'):
                        audio_processing.last_vad_check = current_time
                    
                    if current_time - audio_processing.last_vad_check >= 0.5:  # Check mỗi 0.5s
                        audio_processing.last_vad_check = current_time
                        
                        full_audio_16k = resample_audio(np.array(full_audio), SAMPLE_RATE, 16000)
                        
                        if len(full_audio_16k) > 16000:  # Ít nhất 1 giây audio
                            # Normalize audio to [-1, 1] range for VAD
                            audio_normalized = full_audio_16k.astype(np.float32) / 32768.0
                            
                            # Create torch tensor
                            wav = torch.from_numpy(audio_normalized)
                            
                            # Get speech timestamps với thresholds
                            speech = get_speech_timestamps(
                                wav, 
                                model, 
                                sampling_rate=16000,
                                threshold=0.5,  # Tăng threshold để giảm false positive
                                min_speech_duration_ms=500,  # Tối thiểu 0.5s speech
                                min_silence_duration_ms=300   # Tối thiểu 0.3s silence
                            )
                            
                            # Check speech trong 2 giây gần nhất thay vì toàn bộ audio
                            recent_audio_seconds = 2.0
                            recent_samples = int(recent_audio_seconds * 16000)
                            if len(audio_normalized) > recent_samples:
                                recent_wav = torch.from_numpy(audio_normalized[-recent_samples:])
                                recent_speech = get_speech_timestamps(
                                    recent_wav, 
                                    model, 
                                    sampling_rate=16000,
                                    threshold=0.5,
                                    min_speech_duration_ms=500,
                                    min_silence_duration_ms=300
                                )
                            else:
                                recent_speech = speech
                            
                            # Tính speech ratio trong recent audio
                            if recent_speech:
                                recent_speech_duration = sum([s["end"] - s["start"] for s in recent_speech]) / 16000
                                speech_ratio = recent_speech_duration / recent_audio_seconds
                                
                                print(f"🗣️ Recent speech: {recent_speech_duration:.2f}s/{recent_audio_seconds:.1f}s (ratio: {speech_ratio:.2f})")
                                
                                # Nếu speech ratio < 0.3 thì coi như silence
                                if speech_ratio < 0.3:
                                    if silence_start is None:
                                        silence_start = time.time()
                                        print("🔇 Silence started...")
                                    elif time.time() - silence_start > SILENCE_DURATION:
                                        audio_queue.put(np.array(full_audio, dtype=np.int16))
                                        print(f"✅ Audio segment sent to queue ({len(full_audio)/SAMPLE_RATE:.2f}s)")
                                        recording = False
                                        full_audio = []
                                        silence_start = None
                                else:
                                    silence_start = None  # Reset silence khi có speech
                            else:
                                # Không có speech detected
                                if silence_start is None:
                                    silence_start = time.time()
                                    print("🔇 Complete silence started...")
                                elif time.time() - silence_start > SILENCE_DURATION:
                                    audio_queue.put(np.array(full_audio, dtype=np.int16))
                                    print(f"✅ Audio segment sent to queue ({len(full_audio)/SAMPLE_RATE:.2f}s)")
                                    recording = False
                                    full_audio = []
                                    silence_start = None
                    
                    # Force timeout sau 15 giây recording
                    if hasattr(audio_processing, 'recording_start_time'):
                        if time.time() - audio_processing.recording_start_time > 15.0:
                            audio_queue.put(np.array(full_audio, dtype=np.int16))
                            print(f"⏰ Timeout: Audio segment sent to queue ({len(full_audio)/SAMPLE_RATE:.2f}s)")
                            recording = False
                            full_audio = []
                            silence_start = None
                    
                except Exception as e:
                    print(f"VAD processing error: {e}")
                
            frame_count += 1
                
        except Exception as e:
            print(f"Audio processing error: {e}")
            time.sleep(0.1)  # Ngắt ngỉ khi có lỗi

def main():
    audio_thread = threading.Thread(target=audio_processing)
    audio_thread.daemon = True
    audio_thread.start()
    
    print(f"🎙️ Audio daemon started, listening for '{WAKEWORD}'...")
    print(f"🔧 Using device {DEVICE_INDEX}, sensitivity 0.95")
    print("💡 Speak clearly and close to microphone")
    
    try:
        while True:
            # FIX 12: Thêm thông tin về queue status
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                print(f"📥 Retrieved audio segment: {len(audio_data)} samples, {len(audio_data)/SAMPLE_RATE:.2f}s")
                
                # Process audio segment here (STT, etc.)
                # Bạn có thể thêm code xử lý audio ở đây
                
            time.sleep(0.1)  # Giảm CPU usage
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping audio daemon...")
        porcupine.delete()
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("✅ Cleanup completed")

if __name__ == "__main__":
    main()