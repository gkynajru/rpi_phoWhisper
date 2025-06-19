import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import time
import os
import queue
import librosa
from new_audio import audio_queue  # Import audio_queue from new_audio.py
# Configuration
PHOWHISPER_MODEL = "models/phowhisper_multistage"  # Path to PhoWhisper checkpoint
TRANSCRIPTION_DIR = "data/transcriptions"
INPUT_SAMPLE_RATE = 48000  # Matches new_audio.py
OUTPUT_SAMPLE_RATE = 16000  # Required by PhoWhisper
os.makedirs(TRANSCRIPTION_DIR, exist_ok=True)
# Initialize STT model
try:
    processor = AutoProcessor.from_pretrained(PHOWHISPER_MODEL)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(PHOWHISPER_MODEL)
    model.eval()
except Exception as e:
    print(f"Failed to load PhoWhisper model: {e}")
    exit(1)
def resample_audio(audio_data, original_sr, target_sr):
    """Resample audio data from original_sr to target_sr."""
    return librosa.resample(audio_data.astype(np.float32), orig_sr=original_sr, target_sr=target_sr)
def transcribe(audio):
    """Transcribe audio using PhoWhisper model."""
    try:
        # Resample audio from 48 kHz to 16 kHz
        audio_resampled = resample_audio(audio, INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
        # Convert to float32 and normalize
        audio_float = audio_resampled / 32768.0
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
        print(f"Transcription error: {e}")
        return ""
def main():
    print("Starting STT service...")
    while True:
        try:
            # Get audio data from the queue
            audio_data = audio_queue.get(timeout=1.0)
            start_time = time.time()
            transcription = transcribe(audio_data)
            latency = time.time() - starttime
            print(f"Transcription: {transcription} (Latency: {latency:.2f}s)")
            if transcription:
                # Save transcription with timestamp
                timestamp = time.strftime("%Y%m%d%H%M%S")
                file_path = f"{TRANSCRIPTION_DIR}/{timestamp}.txt"
                with open(file_path, "w") as f:
                    f.write(transcription)
                print(f"Transcription saved: {file_path}")
            audio_queue.task_done()  # Mark the task as done
        except queue.Empty:
            continue  # Queue is empty, keep waiting
        except KeyboardInterrupt:
            print("Stopping STT service...")
            break
        except Exception as e:
            print(f"STT error: {e}")
            continue
if name == "main":
    main()