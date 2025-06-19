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
PHOWHISPER_MODEL = "models/phowhisper_multistage"  # Path to PhoWhisper checkpoint
TRANSCRIPTION_DIR = "data/transcriptions"
AUDIO_CHUNKS_DIR = "data/audio_chunks"  # Same as audio system
INPUT_SAMPLE_RATE = 48000  # Matches new_audio_update_v3
OUTPUT_SAMPLE_RATE = 16000  # Required by PhoWhisper

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

def simple_downsample(audio_data, factor=3):
    """Simple downsampling by taking every nth sample"""
    try:
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
        audio_data = np.clip(audio_data, -32768, 32767)
        return audio_data[::factor]
    except Exception as e:
        print(f"Simple downsample error: {e}")
        return None

def resample_audio(audio_data, original_sr, target_sr):
    """Resample audio data from original_sr to target_sr."""
    try:
        # Method 1: Simple downsampling (fast and stable)
        if original_sr == 48000 and target_sr == 16000:
            return simple_downsample(audio_data, factor=3)
        
        # Method 2: Librosa fallback
        clean_data = audio_data.astype(np.float32)
        clean_data = np.nan_to_num(clean_data, nan=0.0, posinf=0.0, neginf=0.0)
        resampled = librosa.resample(clean_data, orig_sr=original_sr, target_sr=target_sr)
        return np.clip(resampled, -32768, 32767)
    except Exception as e:
        print(f"Resample error: {e}")
        # Last resort - simple decimation
        factor = int(original_sr / target_sr)
        return audio_data[::factor]

def transcribe(audio):
    """Transcribe audio using PhoWhisper model."""
    try:
        # Resample audio from 48 kHz to 16 kHz
        audio_resampled = resample_audio(audio, INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
        if audio_resampled is None:
            return ""
            
        # Convert to float32 and normalize
        audio_float = audio_resampled.astype(np.float32) / 32768.0
        
        # Process audio for PhoWhisper
        inputs = processor(audio_float, sampling_rate=OUTPUT_SAMPLE_RATE, return_tensors="pt")
        
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
    """Monitor for new speech files and transcribe them"""
    processed_files = set()
    
    while True:
        try:
            # Check for new speech files
            if os.path.exists(AUDIO_CHUNKS_DIR):
                speech_files = [f for f in os.listdir(AUDIO_CHUNKS_DIR) 
                              if f.startswith('speech_') and f.endswith('.wav') and f not in processed_files]
                
                for filename in speech_files:
                    file_path = os.path.join(AUDIO_CHUNKS_DIR, filename)
                    
                    try:
                        # Load audio file
                        audio_data, sample_rate = sf.read(file_path)
                        
                        # Convert to int16 if needed
                        if audio_data.dtype != np.int16:
                            audio_data = (audio_data * 32767).astype(np.int16)
                        
                        # Transcribe
                        start_time = time.time()
                        transcription = transcribe(audio_data)
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
    """Process audio from queue (if using shared queue)"""
    print("🎧 STT queue processor started...")
    
    while True:
        try:
            # Get audio data from the queue
            audio_data = audio_queue.get(timeout=1.0)
            
            start_time = time.time()
            transcription = transcribe(audio_data)
            latency = time.time() - start_time
            
            print(f"🎤 Queue Transcription: '{transcription}' (Latency: {latency:.2f}s)")
            
            if transcription:
                # Save transcription with timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_path = f"{TRANSCRIPTION_DIR}/queue_{timestamp}.txt"
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write(transcription)
                print(f"💾 Queue transcription saved: {file_path}")
            
            audio_queue.task_done()  # Mark the task as done
            
        except queue.Empty:
            continue  # Queue is empty, keep waiting
        except KeyboardInterrupt:
            print("🛑 Stopping STT queue processor...")
            break
        except Exception as e:
            print(f"❌ STT queue error: {e}")
            continue

def main():
    """Main STT service"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='STT Service')
    parser.add_argument('--method', choices=['1', '2', '3'], 
                       help='Processing method: 1=File monitoring, 2=Queue processing, 3=Both')
    args = parser.parse_args()
    
    print("🚀 Starting STT service...")
    print(f"📁 Monitoring directory: {AUDIO_CHUNKS_DIR}")
    print(f"💾 Saving transcriptions to: {TRANSCRIPTION_DIR}")
    
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
        print("✅ File monitoring thread started")
    
    if processing_method in ['2', '3']:
        # Start queue processing thread
        queue_thread = threading.Thread(target=stt_queue_processor)
        queue_thread.daemon = True
        queue_thread.start()
        threads.append(queue_thread)
        print("✅ Queue processing thread started")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping STT service...")
        print("✅ STT service stopped")

if __name__ == "__main__":
    main()