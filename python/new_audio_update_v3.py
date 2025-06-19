# FIX: Clean audio recording logic với wake word trimming

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

# FIX: Add missing trim_silence_end function
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

def find_speech_onset_after_wake_word(audio_data, wake_word_detection_time, sample_rate=48000, vad_model=None, utils=None):
    """
    Tìm điểm bắt đầu user speech sau wake word
    Returns: onset_sample_index
    """
    try:
        # Method 1: Time-based estimation
        estimated_wake_word_duration = 0.6  # seconds for "computer"
        min_search_start = int(wake_word_detection_time * sample_rate)
        estimated_wake_end = min_search_start + int(estimated_wake_word_duration * sample_rate)
        
        # Method 2: Energy-based gap detection
        search_window_start = max(0, estimated_wake_end - int(0.2 * sample_rate))  # 200ms before estimated end
        search_window_end = min(len(audio_data), estimated_wake_end + int(1.0 * sample_rate))  # 1s after estimated end
        
        if search_window_end <= search_window_start:
            return estimated_wake_end
            
        search_audio = audio_data[search_window_start:search_window_end]
        
        # Find silence gaps using energy thresholding
        window_size = int(0.05 * sample_rate)  # 50ms windows
        energy_threshold = np.max(np.abs(search_audio)) * 0.15  # 15% of max energy
        
        silence_gaps = []
        for i in range(0, len(search_audio) - window_size, window_size // 2):
            window = search_audio[i:i + window_size]
            energy = np.sqrt(np.mean(window.astype(np.float32) ** 2))
            if energy < energy_threshold:
                silence_gaps.append(i + search_window_start)
        
        # Method 3: VAD-based speech onset detection
        speech_onset = None
        if vad_model is not None and utils is not None:
            try:
                # Analyze audio from estimated wake word end
                analysis_start = max(0, estimated_wake_end - int(0.1 * sample_rate))
                analysis_audio = audio_data[analysis_start:]
                
                if len(analysis_audio) > sample_rate:  # At least 1 second
                    # Downsample to 16kHz for VAD
                    if sample_rate == 48000:
                        audio_16k = analysis_audio[::3]  # Simple downsampling
                    else:
                        audio_16k = analysis_audio
                    
                    # Normalize and create tensor
                    audio_normalized = audio_16k.astype(np.float32) / 32768.0
                    wav = torch.from_numpy(audio_normalized)
                    
                    # Get speech timestamps
                    get_speech_timestamps, _, _, _, _ = utils
                    speech_segments = get_speech_timestamps(
                        wav, vad_model, 
                        sampling_rate=16000,
                        threshold=0.3,
                        min_speech_duration_ms=200,
                        min_silence_duration_ms=100
                    )
                    
                    if speech_segments:
                        # Find first significant speech segment
                        for segment in speech_segments:
                            segment_duration = (segment["end"] - segment["start"]) / 16000
                            if segment_duration > 0.3:  # At least 300ms of speech
                                # Convert back to original sample rate
                                if sample_rate == 48000:
                                    speech_onset = analysis_start + segment["start"] * 3
                                else:
                                    speech_onset = analysis_start + segment["start"]
                                break
                                
            except Exception as e:
                print(f"VAD analysis error: {e}")
        
        # Method 4: Combined decision
        candidates = []
        
        # Add time-based estimate
        candidates.append(estimated_wake_end)
        
        # Add silence gap candidates
        if silence_gaps:
            post_wake_gaps = [gap for gap in silence_gaps if gap > estimated_wake_end]
            if post_wake_gaps:
                candidates.append(post_wake_gaps[0])
        
        # Add VAD-based onset
        if speech_onset is not None:
            candidates.append(int(speech_onset))
        
        # Choose the most conservative candidate
        candidates = [c for c in candidates if c < len(audio_data) - int(0.5 * sample_rate)]
        
        if candidates:
            min_reasonable = min_search_start + int(0.3 * sample_rate)  # At least 300ms after detection
            max_reasonable = min_search_start + int(2.0 * sample_rate)  # At most 2s after detection
            
            reasonable_candidates = [c for c in candidates if min_reasonable <= c <= max_reasonable]
            
            if reasonable_candidates:
                return max(reasonable_candidates)
            else:
                return max(candidates)
        
        return estimated_wake_end
        
    except Exception as e:
        print(f"Speech onset detection error: {e}")
        return min_search_start + int(0.6 * sample_rate)

def trim_wake_word_from_audio(audio_data, wake_word_detection_time, sample_rate=48000, 
                             vad_model=None, utils=None, method="hybrid"):
    """
    Trim wake word from audio, keeping only user speech
    """
    
    if len(audio_data) < sample_rate:  # Less than 1 second
        return audio_data
    
    try:
        if method == "time":
            # Simple time-based trimming
            trim_duration = 0.8  # seconds
            trim_samples = int(wake_word_detection_time * sample_rate) + int(trim_duration * sample_rate)
            return audio_data[trim_samples:] if trim_samples < len(audio_data) else audio_data
        
        elif method == "energy":
            # Energy-based gap detection
            onset = find_speech_onset_after_wake_word(audio_data, wake_word_detection_time, sample_rate)
            return audio_data[onset:]
        
        elif method == "vad":
            # VAD-based detection
            onset = find_speech_onset_after_wake_word(audio_data, wake_word_detection_time, sample_rate, vad_model, utils)
            return audio_data[onset:]
        
        else:  # hybrid (default)
            # Combined approach
            onset = find_speech_onset_after_wake_word(audio_data, wake_word_detection_time, sample_rate, vad_model, utils)
            
            # Additional safety: ensure we don't cut too aggressively
            min_safe_trim = int(wake_word_detection_time * sample_rate) + int(0.4 * sample_rate)  # Minimum 400ms after detection
            final_onset = max(onset, min_safe_trim)
            
            return audio_data[final_onset:] if final_onset < len(audio_data) else audio_data[min_safe_trim:]
        
    except Exception as e:
        print(f"Wake word trimming error: {e}")
        # Fallback to simple trimming
        fallback_trim = int(wake_word_detection_time * sample_rate) + int(0.6 * sample_rate)
        return audio_data[fallback_trim:] if fallback_trim < len(audio_data) else audio_data

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

# FIX: Audio processing function with corrected references
def audio_processing():
    global buffer, pre_buffer
    recording = False
    silence_start = None
    full_audio = []
    frame_count = 0
    detection_count = 0
    wake_word_detection_timestamp = None
    recording_start_time = None
    last_vad_check = None
    
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
                # UPDATE BUFFERS
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
                        
                        # TRACK WAKE WORD TIMING
                        recording_start_time = time.time()
                        wake_word_detection_timestamp = 0.5  # Wake word is at 0.5s in pre_buffer
                        
                        # Start recording with pre-buffer
                        short_pre_buffer_samples = int(SAMPLE_RATE * 0.5)
                        recording = True
                        
                        full_audio = list(pre_buffer[-short_pre_buffer_samples:])
                        print(f"🎙️ Started recording with {len(full_audio)/SAMPLE_RATE:.2f}s pre-buffer")
                        
                        # Save wake word frame
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        sf.write(f"{AUDIO_CHUNKS_DIR}/wake_frame_{timestamp}.wav", frame_16k, PORCUPINE_SAMPLE_RATE)
                        
                        time.sleep(0.3)
                        
                except Exception as e:
                    print(f"Porcupine process error: {e}")
                    
            else:
                # RECORDING MODE
                # Check for new wake word during recording
                try:
                    keyword_index = porcupine.process(frame_16k)
                    if keyword_index >= 0:
                        print(f"\n⚠️ NEW WAKE WORD DETECTED during recording!")
                        print(f"🔄 Stopping current recording and starting new one...")
                        
                        # Save current recording WITH WAKE WORD TRIMMING
                        if len(full_audio) > SAMPLE_RATE:
                            # Trim wake word before saving
                            trimmed_audio = trim_silence_end(full_audio)
                            speech_only_audio = trim_wake_word_from_audio(
                                trimmed_audio, 
                                wake_word_detection_timestamp, 
                                SAMPLE_RATE, 
                                model,  # VAD model
                                utils,  # VAD utils
                                method="hybrid"
                            )
                            
                            audio_queue.put(np.array(speech_only_audio, dtype=np.int16))
                            print(f"✅ Previous audio segment saved ({len(speech_only_audio)/SAMPLE_RATE:.2f}s, wake word trimmed)")
                        
                        # Start new recording
                        detection_count += 1
                        recording_start_time = time.time()
                        wake_word_detection_timestamp = 0.5
                        short_pre_buffer_samples = int(SAMPLE_RATE * 0.5)
                        full_audio = list(pre_buffer[-short_pre_buffer_samples:])
                        
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        sf.write(f"{AUDIO_CHUNKS_DIR}/wake_frame_{timestamp}.wav", frame_16k, PORCUPINE_SAMPLE_RATE)
                        print(f"🎙️ New recording started #{detection_count}")
                        
                        time.sleep(0.3)
                        frame_count += 1
                        continue
                        
                except Exception as e:
                    print(f"Wake word check error: {e}")
                
                # Continue recording
                full_audio.extend(data_48k)
                
                # VAD processing for silence detection
                try:
                    current_time = time.time()
                    if last_vad_check is None:
                        last_vad_check = current_time
                    
                    if current_time - last_vad_check >= 0.5:
                        last_vad_check = current_time
                        
                        full_audio_16k = resample_audio(np.array(full_audio), SAMPLE_RATE, 16000)
                        
                        if len(full_audio_16k) > 16000:
                            # VAD analysis
                            recent_audio_seconds = 2.0
                            recent_samples = int(recent_audio_seconds * 16000)
                            if len(full_audio_16k) > recent_samples:
                                recent_audio = full_audio_16k[-recent_samples:]
                            else:
                                recent_audio = full_audio_16k
                            
                            audio_normalized = recent_audio.astype(np.float32) / 32768.0
                            wav = torch.from_numpy(audio_normalized)
                            
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
                                        # SAVE WITH WAKE WORD TRIMMING
                                        trimmed_audio = trim_silence_end(full_audio)
                                        speech_only_audio = trim_wake_word_from_audio(
                                            trimmed_audio, 
                                            wake_word_detection_timestamp, 
                                            SAMPLE_RATE, 
                                            model, 
                                            utils, 
                                            method="hybrid"
                                        )
                                        
                                        audio_queue.put(np.array(speech_only_audio, dtype=np.int16))
                                        print(f"✅ Audio segment sent to queue ({len(speech_only_audio)/SAMPLE_RATE:.2f}s, wake word trimmed)")
                                        recording = False
                                        full_audio = []
                                        silence_start = None
                                        wake_word_detection_timestamp = None
                                else:
                                    silence_start = None
                            else:
                                if silence_start is None:
                                    silence_start = time.time()
                                    print("🔇 Complete silence started...")
                                elif time.time() - silence_start > SILENCE_DURATION:
                                    trimmed_audio = trim_silence_end(full_audio)
                                    speech_only_audio = trim_wake_word_from_audio(
                                        trimmed_audio, 
                                        wake_word_detection_timestamp, 
                                        SAMPLE_RATE, 
                                        model, 
                                        utils, 
                                        method="hybrid"
                                    )
                                    
                                    audio_queue.put(np.array(speech_only_audio, dtype=np.int16))
                                    print(f"✅ Audio segment sent to queue ({len(speech_only_audio)/SAMPLE_RATE:.2f}s, wake word trimmed)")
                                    recording = False
                                    full_audio = []
                                    silence_start = None
                                    wake_word_detection_timestamp = None
                    
                    # Timeout handling
                    if recording_start_time is not None:
                        if time.time() - recording_start_time > 15.0:
                            trimmed_audio = trim_silence_end(full_audio)
                            speech_only_audio = trim_wake_word_from_audio(
                                trimmed_audio, 
                                wake_word_detection_timestamp, 
                                SAMPLE_RATE, 
                                model, 
                                utils, 
                                method="hybrid"
                            )
                            
                            audio_queue.put(np.array(speech_only_audio, dtype=np.int16))
                            print(f"⏰ Timeout: Audio segment sent to queue ({len(speech_only_audio)/SAMPLE_RATE:.2f}s, wake word trimmed)")
                            recording = False
                            full_audio = []
                            silence_start = None
                            wake_word_detection_timestamp = None
                    
                except Exception as e:
                    print(f"VAD processing error: {e}")
                
            frame_count += 1
                
        except Exception as e:
            print(f"Audio processing error: {e}")
            time.sleep(0.1)

# Main function with trimmed audio saving
def main():
    audio_thread = threading.Thread(target=audio_processing)
    audio_thread.daemon = True
    audio_thread.start()
    
    print(f"🎙️ Audio daemon started, listening for '{WAKEWORD}'...")
    print(f"🔧 Using device {DEVICE_INDEX}, sensitivity 0.95")
    print("💡 Speak clearly and close to microphone")
    print("✂️ Wake word trimming enabled")
    
    audio_segment_count = 0
    
    try:
        while True:
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                audio_segment_count += 1
                
                print(f"📥 Retrieved audio segment: {len(audio_data)} samples, {len(audio_data)/SAMPLE_RATE:.2f}s")
                
                # Save speech-only audio files
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                
                # Save full quality speech-only
                speech_filename = f"{AUDIO_CHUNKS_DIR}/speech_{timestamp}_{audio_segment_count:03d}.wav"
                sf.write(speech_filename, audio_data, SAMPLE_RATE)
                print(f"💾 Saved speech-only: {speech_filename}")
                
                # Save 16kHz version for STT
                audio_16k = simple_downsample(audio_data, factor=3)
                if audio_16k is not None:
                    speech_16k_filename = f"{AUDIO_CHUNKS_DIR}/speech_16k_{timestamp}_{audio_segment_count:03d}.wav"
                    sf.write(speech_16k_filename, audio_16k.astype(np.int16), 16000)
                    print(f"💾 Saved speech-only 16kHz: {speech_16k_filename}")
                
                # TODO: Add STT processing here - now with clean speech-only audio!
                # transcription = your_stt_model(audio_data)
                # print(f"🎤 Transcription: {transcription}")
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping audio daemon...")
        porcupine.delete()
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("✅ Cleanup completed")

if __name__ == "__main__":
    main()