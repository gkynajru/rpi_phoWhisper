import psutil
import time
import os
import subprocess
import signal
import queue
import numpy as np
from datetime import datetime
from new_audio import audio_queue
from new_stt import transcribe
from threading import Thread, Event

# Configuration
TEST_DURATION = 60  # Seconds to run test
METRICS_INTERVAL = 1  # Seconds between metrics collection
LOG_DIR = "data/test_metrics"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{LOG_DIR}/{timestamp}_metrics.txt"

def log_metrics(message):
    """Write metrics to log file."""
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()}: {message}\n")
    print(message)

def get_process_metrics(pid):
    """Get CPU and RAM usage for a process."""
    try:
        proc = psutil.Process(pid)
        cpu_percent = proc.cpu_percent(interval=0.1)
        mem_info = proc.memory_info()
        ram_mb = mem_info.rss / 1024 / 1024  # Convert to MB
        return cpu_percent, ram_mb
    except psutil.NoSuchProcess:
        return 0.0, 0.0

def monitor_metrics(audio_pid, stt_pid, stop_event):
    """Monitor CPU and RAM usage of audio and STT processes."""
    start_time = time.time()
    while not stop_event.is_set():
        audio_cpu, audio_ram = get_process_metrics(audio_pid)
        stt_cpu, stt_ram = get_process_metrics(stt_pid)
        total_cpu = audio_cpu + stt_cpu
        total_ram = audio_ram + stt_ram
        elapsed = time.time() - start_time
        log_metrics(
            f"CPU: Audio={audio_cpu:.2f}%, STT={stt_cpu:.2f}%, Total={total_cpu:.2f}% | "
            f"RAM: Audio={audio_ram:.2f}MB, STT={stt_ram:.2f}MB, Total={total_ram:.2f}MB | "
            f"Elapsed={elapsed:.2f}s"
        )
        time.sleep(METRICS_INTERVAL)

def process_transcriptions():
    """Process audio from queue and measure transcription latency."""
    latencies = []
    while True:
        try:
            audio_data = audio_queue.get(timeout=1.0)
            start_time = time.time()
            transcription = transcribe(audio_data)
            latency = time.time() - start_time
            if transcription:
                latencies.append(latency)
                log_metrics(
                    f"Transcription: '{transcription}' | Latency={latency:.2f}s | "
                    f"Avg Latency={np.mean(latencies):.2f}s (n={len(latencies)})"
                )
                # Save transcription
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"data/transcriptions/{timestamp}.txt"
                with open(file_path, "w") as f:
                    f.write(transcription)
                log_metrics(f"Transcription saved: {file_path}")
            audio_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            log_metrics(f"Transcription error: {e}")

def main():
    # Start new_audio.py and new_stt.py as subprocesses
    audio_proc = subprocess.Popen(["python3", "new_audio.py"])
    stt_proc = subprocess.Popen(["python3", "new_stt.py"])

    # Create stop event for metrics monitoring
    stop_event = Event()

    # Start metrics monitoring in a separate thread
    metrics_thread = Thread(target=monitor_metrics, args=(audio_proc.pid, stt_proc.pid, stop_event))
    metrics_thread.daemon = True
    metrics_thread.start()

    # Start transcription processing in a separate thread
    transcription_thread = Thread(target=process_transcriptions)
    transcription_thread.daemon = True
    transcription_thread.start()

    log_metrics("Starting test...")
    try:
        # Run for TEST_DURATION or until interrupted
        time.sleep(TEST_DURATION)
    except KeyboardInterrupt:
        log_metrics("Test interrupted by user")
    finally:
        # Signal stop and cleanup
        stop_event.set()
        audio_proc.send_signal(signal.SIGINT)
        stt_proc.send_signal(signal.SIGINT)
        audio_proc.wait()
        stt_proc.wait()
        log_metrics("Test completed")

if __name__ == "__main__":
    main()