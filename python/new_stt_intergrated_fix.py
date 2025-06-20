# COMPLETE FIX for new_stt_integrated.py

import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import time
import os
import queue
import librosa
import soundfile as sf
import threading
import subprocess
import sys

# Configuration
PHOWHISPER_MODEL = "models/phowhisper_multistage"
TRANSCRIPTION_DIR = "data/transcriptions"
AUDIO_CHUNKS_DIR = "data/audio_chunks"
OUTPUT_SAMPLE_RATE = 16000  # Required by PhoWhisper - ONLY keep target rate

# Create directories
os.makedirs(TRANSCRIPTION_DIR, exist_ok=True)
os.makedirs(AUDIO_CHUNKS_DIR, exist_ok=True)

# Shared queue (same as audio system)
audio_queue = queue.Queue()

# Initialize STT model
try:
    print("🔄 Loading PhoWhisper model...")
    processor = AutoProcessor.from_pretrained(PHOWHISPER_MODEL)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(PHOWHISPER_MODEL)
    model.eval()
    print("✅ PhoWhisper model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load PhoWhisper model: {e}")
    exit(1)

def resample_audio_smart(audio_data, actual_sr, target_sr):
    """SMART resampling that uses ACTUAL sample rate"""
    try:
        # If already at target rate, no resampling needed!
        if actual_sr == target_sr:
            print(f"✅ Audio already at {target_sr}Hz, no resampling needed")
            return audio_data.astype(np.float32)
        
        # Otherwise, use high-quality librosa resampling
        print(f"🔄 Resampling from {actual_sr}Hz to {target_sr}Hz")
        audio_float = audio_data.astype(np.float32)
        if np.max(np.abs(audio_float)) > 1.0:
            audio_float = audio_float / 32768.0  # Normalize if int16
            
        resampled = librosa.resample(audio_float, orig_sr=actual_sr, target_sr=target_sr)
        return resampled
        
    except Exception as e:
        print(f"❌ Resample error: {e}")
        # If librosa fails, at least don't make it worse
        if actual_sr == target_sr:
            return audio_data.astype(np.float32)
        else:
            # Basic linear interpolation fallback
            from scipy import signal
            num_samples = int(len(audio_data) * target_sr / actual_sr)
            return signal.resample(audio_data.astype(np.float32), num_samples)

def transcribe_fixed(audio_data, actual_sample_rate):
    """FIXED transcribe function that uses ACTUAL sample rate"""
    try:
        print(f"🎤 Transcribing audio: {len(audio_data)} samples @ {actual_sample_rate}Hz")
        
        # Smart resampling using ACTUAL sample rate
        audio_resampled = resample_audio_smart(audio_data, actual_sample_rate, OUTPUT_SAMPLE_RATE)
        
        # Ensure proper normalization for model
        if np.max(np.abs(audio_resampled)) > 1.0:
            audio_resampled = audio_resampled / np.max(np.abs(audio_resampled))
        
        # Process audio for PhoWhisper
        inputs = processor(audio_resampled, sampling_rate=OUTPUT_SAMPLE_RATE, return_tensors="pt")
        
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_features"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=128
            )
        
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

def monitor_audio_files():
    """Monitor for new speech files and transcribe them - FIXED VERSION"""
    processed_files = set()
    
    while True:
        try:
            # Check for new speech files
            if os.path.exists(AUDIO_CHUNKS_DIR):
                # Process BOTH 16kHz and 48kHz files properly
                speech_files = [f for f in os.listdir(AUDIO_CHUNKS_DIR) 
                              if (f.startswith('speech_') or f.startswith('speech_16k_')) 
                              and f.endswith('.wav') and f not in processed_files]
                
                for filename in speech_files:
                    file_path = os.path.join(AUDIO_CHUNKS_DIR, filename)
                    
                    try:
                        # Load audio file and get ACTUAL sample rate
                        audio_data, actual_sample_rate = sf.read(file_path)
                        print(f"📁 Processing: {filename} ({len(audio_data)} samples @ {actual_sample_rate}Hz)")
                        
                        # Convert to int16 if needed (for consistency)
                        if audio_data.dtype != np.int16:
                            if np.max(np.abs(audio_data)) <= 1.0:
                                audio_data = (audio_data * 32767).astype(np.int16)
                            else:
                                audio_data = audio_data.astype(np.int16)
                        
                        # Transcribe using ACTUAL sample rate
                        start_time = time.time()
                        transcription = transcribe_fixed(audio_data, actual_sample_rate)
                        latency = time.time() - start_time
                        
                        print(f"🎤 Transcription: '{transcription}' (Latency: {latency:.2f}s)")
                        
                        if transcription:
                            # Save transcription with timestamp
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            transcription_path = f"{TRANSCRIPTION_DIR}/{timestamp}.txt"
                            with open(transcription_path, "w", encoding='utf-8') as f:
                                f.write(transcription)
                            print(f"💾 Transcription saved: {transcription_path}")
                        
                        processed_files.add(filename)
                        
                    except Exception as e:
                        print(f"❌ Error processing {filename}: {e}")
                        processed_files.add(filename)  # Skip problematic files
                        
            time.sleep(0.5)  # Check every 500ms
            
        except KeyboardInterrupt:
            print("🛑 Stopping STT file monitor...")
            break
        except Exception as e:
            print(f"❌ File monitor error: {e}")
            time.sleep(1)

def stt_queue_processor():
    """Process audio from queue - FIXED VERSION"""
    print("🎧 STT queue processor started...")
    
    while True:
        try:
            # Get audio data from the queue - ASSUME it's 48kHz from live system
            audio_data = audio_queue.get(timeout=1.0)
            
            start_time = time.time()
            # Queue audio comes from live system, likely 48kHz
            transcription = transcribe_fixed(audio_data, 48000)  # Explicit sample rate
            latency = time.time() - start_time
            
            print(f"🎤 Queue Transcription: '{transcription}' (Latency: {latency:.2f}s)")
            
            if transcription:
                # Save transcription with timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_path = f"{TRANSCRIPTION_DIR}/queue_{timestamp}.txt"
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write(transcription)
                print(f"💾 Queue transcription saved: {file_path}")
            
            audio_queue.task_done()
            
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            print("🛑 Stopping STT queue processor...")
            break
        except Exception as e:
            print(f"❌ STT queue error: {e}")
            continue

# BACKWARD COMPATIBILITY: Keep old function name but with warning
def transcribe(audio):
    """DEPRECATED - Use transcribe_fixed() with actual sample rate"""
    print("⚠️ WARNING: Using deprecated transcribe() function")
    print("⚠️ This function assumes 48kHz input - may cause audio distortion!")
    return transcribe_fixed(audio, 48000)  # Default to 48kHz assumption

def main():
    """Main STT service - FIXED VERSION"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='STT Service (FIXED)')
    parser.add_argument('--method', choices=['1', '2', '3'], 
                       help='Processing method: 1=File monitoring, 2=Queue processing, 3=Both')
    args = parser.parse_args()
    
    print("🚀 Starting FIXED STT service...")
    print(f"📁 Monitoring directory: {AUDIO_CHUNKS_DIR}")
    print(f"💾 Saving transcriptions to: {TRANSCRIPTION_DIR}")
    print("🔧 Now using ACTUAL sample rates from files!")
    
    # Determine processing method
    if args.method:
        processing_method = args.method
        print(f"🎯 Using method {processing_method} (from command line)")
    else:
        # Interactive mode
        print("Choose processing method:")
        print("1. File monitoring (recommended)")
        print("2. Queue processing") 
        print("3. Both")
        processing_method = input("Enter (1/2/3): ").strip()
        
        while processing_method not in ['1', '2', '3']:
            print("❌ Invalid choice, please enter 1, 2, or 3")
            processing_method = input("Enter (1/2/3): ").strip()
    
    threads = []
    
    if processing_method in ['1', '3']:
        # Start file monitoring thread
        file_thread = threading.Thread(target=monitor_audio_files)
        file_thread.daemon = True
        file_thread.start()
        threads.append(file_thread)
        print("✅ FIXED file monitoring thread started")
    
    if processing_method in ['2', '3']:
        # Start queue processing thread
        queue_thread = threading.Thread(target=stt_queue_processor)
        queue_thread.daemon = True
        queue_thread.start()
        threads.append(queue_thread)
        print("✅ FIXED queue processing thread started")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping FIXED STT service...")
        print("✅ STT service stopped")

if __name__ == "__main__":
    main()