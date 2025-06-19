# FIX: Clean audio recording logic

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
DEVICE_INDEX = 1  # Sử dụng pulse (device 1) 
SAMPLE_RATE = 48000
PORCUPINE_SAMPLE_RATE = 16000
ACCESS_KEY = '1yGPnMC26F0ODv6AjGLjy7T0VRBYt4AtR1tyiyvoRl0Uc9dSdBKhMw=='
WAKEWORD = "computer"
AUDIO_CHUNKS_DIR = "data/audio_chunks"
os.makedirs(AUDIO_CHUNKS_DIR, exist_ok=True)

# Buffer settings
BUFFER_SECONDS = 5
PRE_BUFFER_SECONDS = 1
VAD_THRESHOLD = 0.02
SILENCE_DURATION = 1.0

# Shared queue for audio data
audio_queue = queue.Queue()

# Suppress ALSA warnings
os.environ["ALSA_LOG_LEVEL"] = "none"

# Initialize PyAudio
pa = pyaudio.PyAudio()

def clean_audio_data(audio_data):
    """Clean audio data - remove NaN, inf, normalize"""
    audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
    audio_data = np.clip(audio_data, -32768, 32767)
    return audio_data

def simple_downsample(audio_data, factor=3):
    """Simple downsampling by taking every nth sample"""
    try:
        audio_data = clean_audio_data(audio_data)
        return audio_data[::factor]
    except Exception as e:
        print(f"Simple downsample error: {e}")
        return None

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

# Initialize Porcupine
porcupine = None
try:
    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keywords=[WAKEWORD],
        sensitivities=[0.95]  # High sensitivity
    )
    print("Porcupine initialized successfully.")
    print(f"Frame length: {porcupine.frame_length}, Sample rate: {porcupine.sample_rate}")
except Exception as e:
    print(f"Failed to initialize Porcupine: {e}")
    exit(1)

# Open audio stream
try:
    buffer_size = porcupine.frame_length * 3  # 1536
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=buffer_size,
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

# Global buffers
buffer = np.zeros(int(SAMPLE_RATE * BUFFER_SECONDS), dtype=np.int16)
pre_buffer = np.zeros(int(SAMPLE_RATE * PRE_BUFFER_SECONDS), dtype=np.int16)

def audio_processing():
    global buffer, pre_buffer
    recording = False
    silence_start = None
    full_audio = []
    frame_count = 0
    detection_count = 0
    
    chunk_size = porcupine.frame_length * 3
    
    while True:
        try:
            # Read audio data
            data_48k = np.frombuffer(
                stream.read(chunk_size, exception_on_overflow=False), 
                dtype=np.int16
            )
            data_48k = clean_audio_data(data_48k)
            
            # Debug volume
            if frame_count % 100 == 0:
                volume = np.sqrt(np.mean(data_48k.astype(np.float32)**2))
                max_val = np.max(np.abs(data_48k))
                print(f"Audio Frame {frame_count}: Volume={volume:.2f}, Max={max_val}")
            
            # Prepare frame for Porcupine
            data_16k = simple_downsample(data_48k, factor=3)
            if data_16k is not None and len(data_16k) >= porcupine.frame_length:
                frame_16k = data_16k[:porcupine.frame_length].astype(np.int16)
            else:
                frame_count += 1
                continue
            
            if not recording:
                # UPDATE BUFFERS - chỉ khi không recording
                buffer = np.roll(buffer, -len(data_48k))
                buffer[-len(data_48k):] = data_48k
                pre_buffer = np.roll(pre_buffer, -len(data_48k))
                pre_buffer[-len(data_48k):] = data_48k
                
                # WAKE WORD DETECTION
                try:
                    keyword_index = porcupine.process(frame_16k)
                    if keyword_index >= 0:
                        detection_count += 1
                        print(f"\n🎉 WAKE WORD DETECTED! #{detection_count}")
                        print(f"Detected wake word: {WAKEWORD}")
                        
                        # FIX 1: CHỈ LẤY PRE_BUFFER (0.5s trước wake word)
                        # Thay vì lấy 6s, chỉ lấy 0.5s trước wake word
                        short_pre_buffer_samples = int(SAMPLE_RATE * 0.5)  # 0.5 seconds
                        recording = True
                        audio_processing.recording_start_time = time.time()
                        
                        # Chỉ lấy 0.5s cuối của pre_buffer
                        full_audio = list(pre_buffer[-short_pre_buffer_samples:])
                        print(f"🎙️ Started recording with {len(full_audio)/SAMPLE_RATE:.2f}s pre-buffer")
                        
                        # Save wake word frame
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        sf.write(f"{AUDIO_CHUNKS_DIR}/wake_frame_{timestamp}.wav", frame_16k, PORCUPINE_SAMPLE_RATE)
                        
                        time.sleep(0.3)  # Prevent multiple detections
                        
                except Exception as e:
                    print(f"Porcupine process error: {e}")
                    
            else:
                # RECORDING MODE
                # FIX 2: CHECK FOR NEW WAKE WORD - nếu có thì STOP và SAVE current recording
                try:
                    keyword_index = porcupine.process(frame_16k)
                    if keyword_index >= 0:
                        print(f"\n⚠️ NEW WAKE WORD DETECTED during recording!")
                        print(f"🔄 Stopping current recording and starting new one...")
                        
                        # Save current recording
                        if len(full_audio) > SAMPLE_RATE:  # Ít nhất 1s
                            audio_queue.put(np.array(full_audio, dtype=np.int16))
                            print(f"✅ Previous audio segment saved ({len(full_audio)/SAMPLE_RATE:.2f}s)")
                        
                        # Start new recording
                        detection_count += 1
                        short_pre_buffer_samples = int(SAMPLE_RATE * 0.5)
                        full_audio = list(pre_buffer[-short_pre_buffer_samples:])
                        audio_processing.recording_start_time = time.time()
                        
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        sf.write(f"{AUDIO_CHUNKS_DIR}/wake_frame_{timestamp}.wav", frame_16k, PORCUPINE_SAMPLE_RATE)
                        print(f"🎙️ New recording started #{detection_count}")
                        
                        time.sleep(0.3)
                        frame_count += 1
                        continue
                        
                except Exception as e:
                    print(f"Wake word check error: {e}")
                
                # Continue recording current audio
                full_audio.extend(data_48k)
                
                # VAD processing với improvements
                try:
                    current_time = time.time()
                    if not hasattr(audio_processing, 'last_vad_check'):
                        audio_processing.last_vad_check = current_time
                    
                    if current_time - audio_processing.last_vad_check >= 0.5:
                        audio_processing.last_vad_check = current_time
                        
                        full_audio_16k = resample_audio(np.array(full_audio), SAMPLE_RATE, 16000)
                        
                        if len(full_audio_16k) > 16000:
                            # VAD analysis on recent 2 seconds
                            recent_audio_seconds = 2.0
                            recent_samples = int(recent_audio_seconds * 16000)
                            if len(full_audio_16k) > recent_samples:
                                recent_audio = full_audio_16k[-recent_samples:]
                            else:
                                recent_audio = full_audio_16k
                            
                            # Normalize and create tensor
                            audio_normalized = recent_audio.astype(np.float32) / 32768.0
                            wav = torch.from_numpy(audio_normalized)
                            
                            # Get speech timestamps
                            speech = get_speech_timestamps(
                                wav, model, sampling_rate=16000,
                                threshold=0.5, min_speech_duration_ms=500, min_silence_duration_ms=300
                            )
                            
                            if speech:
                                speech_duration = sum([s["end"] - s["start"] for s in speech]) / 16000
                                speech_ratio = speech_duration / len(recent_audio) * 16000
                                print(f"🗣️ Recent speech: {speech_duration:.2f}s/{len(recent_audio)/16000:.1f}s (ratio: {speech_ratio:.2f})")
                                
                                if speech_ratio < 0.3:
                                    if silence_start is None:
                                        silence_start = time.time()
                                        print("🔇 Silence started...")
                                    elif time.time() - silence_start > SILENCE_DURATION:
                                        # FIX 3: TRIM AUDIO - remove silence at end
                                        trimmed_audio = trim_silence_end(full_audio)
                                        audio_queue.put(np.array(trimmed_audio, dtype=np.int16))
                                        print(f"✅ Audio segment sent to queue ({len(trimmed_audio)/SAMPLE_RATE:.2f}s)")
                                        recording = False
                                        full_audio = []
                                        silence_start = None
                                else:
                                    silence_start = None
                            else:
                                if silence_start is None:
                                    silence_start = time.time()
                                    print("🔇 Complete silence started...")
                                elif time.time() - silence_start > SILENCE_DURATION:
                                    trimmed_audio = trim_silence_end(full_audio)
                                    audio_queue.put(np.array(trimmed_audio, dtype=np.int16))
                                    print(f"✅ Audio segment sent to queue ({len(trimmed_audio)/SAMPLE_RATE:.2f}s)")
                                    recording = False
                                    full_audio = []
                                    silence_start = None
                    
                    # FIX 4: TIMEOUT với trimming
                    if hasattr(audio_processing, 'recording_start_time'):
                        if time.time() - audio_processing.recording_start_time > 15.0:
                            trimmed_audio = trim_silence_end(full_audio)
                            audio_queue.put(np.array(trimmed_audio, dtype=np.int16))
                            print(f"⏰ Timeout: Audio segment sent to queue ({len(trimmed_audio)/SAMPLE_RATE:.2f}s)")
                            recording = False
                            full_audio = []
                            silence_start = None
                    
                except Exception as e:
                    print(f"VAD processing error: {e}")
                
            frame_count += 1
                
        except Exception as e:
            print(f"Audio processing error: {e}")
            time.sleep(0.1)

# FIX 5: Function to trim silence at end
def trim_silence_end(audio_data, threshold=500, min_audio_length=None):
    """Trim silence from end of audio"""
    if min_audio_length is None:
        min_audio_length = SAMPLE_RATE  # 1 second minimum
        
    if len(audio_data) < min_audio_length:
        return audio_data
    
    audio_np = np.array(audio_data)
    
    # Find last non-silent sample
    window_size = int(SAMPLE_RATE * 0.1)  # 100ms windows
    for i in range(len(audio_np) - window_size, 0, -window_size):
        if i < 0:
            break
        window = audio_np[i:i+window_size]
        if np.max(np.abs(window)) > threshold:
            # Keep a bit more after last speech
            end_point = min(i + window_size + int(SAMPLE_RATE * 0.5), len(audio_np))
            return audio_data[:end_point]
    
    return audio_data[:min_audio_length]  # Keep at least minimum length

def main():
    audio_thread = threading.Thread(target=audio_processing)
    audio_thread.daemon = True
    audio_thread.start()
    
    print(f"🎙️ Audio daemon started, listening for '{WAKEWORD}'...")
    print(f"🔧 Using device {DEVICE_INDEX}, sensitivity 0.95")
    print("💡 Speak clearly and close to microphone")
    
    audio_segment_count = 0
    
    try:
        while True:
            # SAVE AUDIO FILES từ queue
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                audio_segment_count += 1
                
                print(f"📥 Retrieved audio segment: {len(audio_data)} samples, {len(audio_data)/SAMPLE_RATE:.2f}s")
                
                # SAVE AUDIO FILE
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                audio_filename = f"{AUDIO_CHUNKS_DIR}/audio_{timestamp}_{audio_segment_count:03d}.wav"
                
                try:
                    sf.write(audio_filename, audio_data, SAMPLE_RATE)
                    print(f"💾 Saved: {audio_filename}")
                    
                    # Optional: Create 16kHz version for STT
                    audio_16k = simple_downsample(audio_data, factor=3)
                    if audio_16k is not None:
                        audio_16k_filename = f"{AUDIO_CHUNKS_DIR}/audio_16k_{timestamp}_{audio_segment_count:03d}.wav"
                        sf.write(audio_16k_filename, audio_16k.astype(np.int16), 16000)
                        print(f"💾 Saved 16kHz: {audio_16k_filename}")
                        
                except Exception as e:
                    print(f"❌ Error saving audio: {e}")
                
                # TODO: Add STT processing here
                # transcription = your_stt_model(audio_data)
                # print(f"🎤 Transcription: {transcription}")
                
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