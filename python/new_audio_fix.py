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

# Suppress ALSA warnings
os.environ["ALSA_LOG_LEVEL"] = "none"

# Initialize PyAudio
pa = pyaudio.PyAudio()

def clean_audio_data(audio_data):
    """Clean audio data - remove NaN, inf, normalize"""
    # Remove NaN and inf values
    audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip to valid int16 range
    audio_data = np.clip(audio_data, -32768, 32767)
    
    return audio_data

def resample_audio_scipy(audio_data, original_sr, target_sr):
    """Resample using scipy - more stable than librosa"""
    try:
        # Clean data first
        audio_data = clean_audio_data(audio_data.astype(np.float32))
        
        # Calculate resampling ratio
        ratio = target_sr / original_sr
        num_samples = int(len(audio_data) * ratio)
        
        # Use scipy.signal.resample - more stable
        resampled = signal.resample(audio_data, num_samples)
        
        # Clean again after resampling
        resampled = clean_audio_data(resampled)
        
        return resampled
    except Exception as e:
        print(f"Resample error: {e}")
        return None

def simple_downsample(audio_data, factor=3):
    """Simple downsampling by taking every nth sample"""
    try:
        # Clean data first
        audio_data = clean_audio_data(audio_data)
        return audio_data[::factor]
    except Exception as e:
        print(f"Simple downsample error: {e}")
        return None

def main_fixed():
    print("=== FIXED WAKE WORD DETECTION ===")
    
    # Initialize Porcupine
    try:
        porcupine = pvporcupine.create(
            access_key=ACCESS_KEY,
            keywords=[WAKEWORD],
            sensitivities=[0.95]  # High sensitivity
        )
        print(f"Porcupine initialized: frame_length={porcupine.frame_length}, sample_rate={porcupine.sample_rate}")
    except Exception as e:
        print(f"Porcupine init error: {e}")
        return
    
    # Calculate correct buffer size
    # 48kHz -> 16kHz = 3:1 ratio
    buffer_48k = porcupine.frame_length * 3
    
    # Open audio stream với buffer lớn hơn để tránh overflow
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=buffer_48k,
            input_device_index=DEVICE_INDEX
        )
        print(f"Audio stream opened: device={DEVICE_INDEX}, buffer_size={buffer_48k}")
    except Exception as e:
        print(f"Stream error: {e}")
        porcupine.delete()
        return
    
    print(f"Listening for '{WAKEWORD}' - speak clearly!")
    print("Press Ctrl+C to stop")
    
    frame_count = 0
    detection_count = 0
    
    try:
        while True:
            try:
                # Read audio data
                data_48k = np.frombuffer(
                    stream.read(buffer_48k, exception_on_overflow=False), 
                    dtype=np.int16
                )
                
                # Clean audio data
                data_48k = clean_audio_data(data_48k)
                
                # Debug volume every 50 frames
                if frame_count % 50 == 0:
                    volume = np.sqrt(np.mean(data_48k.astype(np.float32)**2))
                    max_val = np.max(np.abs(data_48k))
                    print(f"Frame {frame_count}: Volume={volume:.2f}, Max={max_val}, Size={len(data_48k)}")
                
                # Method 1: Try simple downsampling first
                data_16k_simple = simple_downsample(data_48k, factor=3)
                
                if data_16k_simple is not None and len(data_16k_simple) >= porcupine.frame_length:
                    # Take exact frame length
                    frame_16k = data_16k_simple[:porcupine.frame_length].astype(np.int16)
                    
                    # Process with Porcupine
                    keyword_index = porcupine.process(frame_16k)
                    
                    if keyword_index >= 0:
                        detection_count += 1
                        print(f"\n🎉 WAKE WORD DETECTED! Count: {detection_count}")
                        
                        # Save detection audio
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        sf.write(f"{AUDIO_CHUNKS_DIR}/detection_{timestamp}.wav", data_48k, SAMPLE_RATE)
                        sf.write(f"{AUDIO_CHUNKS_DIR}/detection_16k_{timestamp}.wav", frame_16k, PORCUPINE_SAMPLE_RATE)
                        print(f"Saved detection audio: detection_{timestamp}.wav")
                        
                        # Wait a bit before continuing
                        time.sleep(1)
                
                # Fallback: Try scipy resampling if simple method fails
                elif len(data_48k) > 0:
                    data_16k_scipy = resample_audio_scipy(data_48k, SAMPLE_RATE, PORCUPINE_SAMPLE_RATE)
                    
                    if data_16k_scipy is not None and len(data_16k_scipy) >= porcupine.frame_length:
                        frame_16k = data_16k_scipy[:porcupine.frame_length].astype(np.int16)
                        
                        keyword_index = porcupine.process(frame_16k)
                        
                        if keyword_index >= 0:
                            detection_count += 1
                            print(f"\n🎉 WAKE WORD DETECTED (scipy)! Count: {detection_count}")
                            
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            sf.write(f"{AUDIO_CHUNKS_DIR}/detection_scipy_{timestamp}.wav", data_48k, SAMPLE_RATE)
                            print(f"Saved detection audio: detection_scipy_{timestamp}.wav")
                            
                            time.sleep(1)
                
                frame_count += 1
                
            except Exception as e:
                print(f"Processing error: {e}")
                continue
                
    except KeyboardInterrupt:
        print(f"\nStopping... Total detections: {detection_count}")
    finally:
        porcupine.delete()
        stream.close()
        pa.terminate()

# Test nhanh với audio file
def test_with_recorded_audio():
    """Test wake word detection với file audio đã record"""
    print("=== TESTING WITH RECORDED AUDIO ===")
    
    # Record 5 seconds of audio
    print("Recording 5 seconds... Say 'computer' clearly!")
    
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=1024,
            input_device_index=DEVICE_INDEX
        )
        
        audio_data = []
        for _ in range(int(SAMPLE_RATE / 1024 * 5)):
            data = np.frombuffer(stream.read(1024), dtype=np.int16)
            audio_data.extend(data)
        
        stream.close()
        audio_data = np.array(audio_data)
        
        # Save recorded audio
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        sf.write(f"{AUDIO_CHUNKS_DIR}/recorded_{timestamp}.wav", audio_data, SAMPLE_RATE)
        print(f"Recorded audio saved: recorded_{timestamp}.wav")
        
        # Test wake word detection on recorded audio
        porcupine = pvporcupine.create(
            access_key=ACCESS_KEY,
            keywords=[WAKEWORD],
            sensitivities=[0.95]
        )
        
        # Downsample to 16kHz
        audio_16k = simple_downsample(audio_data, factor=3)
        sf.write(f"{AUDIO_CHUNKS_DIR}/recorded_16k_{timestamp}.wav", audio_16k, PORCUPINE_SAMPLE_RATE)
        
        # Process in chunks
        detections = 0
        for i in range(0, len(audio_16k) - porcupine.frame_length, porcupine.frame_length):
            frame = audio_16k[i:i+porcupine.frame_length].astype(np.int16)
            result = porcupine.process(frame)
            if result >= 0:
                detections += 1
                print(f"Detection at frame {i//porcupine.frame_length}: {result}")
        
        print(f"Total detections in recorded audio: {detections}")
        porcupine.delete()
        
    except Exception as e:
        print(f"Recording test error: {e}")

if __name__ == "__main__":
    choice = input("Choose: (1) Live detection, (2) Test with recording: ")
    
    if choice == "2":
        test_with_recorded_audio()
    else:
        main_fixed()