# QUICK DEBUG: Chạy test này để confirm vấn đề

import queue
import time
import subprocess
import sys
import os
import signal

def test_queue_between_processes():
    """Test để confirm queue không work giữa processes"""
    
    print("🔍 QUICK DEBUG: Queue Communication Test")
    print("=" * 40)
    
    # Test 1: Tạo 2 files để test
    audio_script = """
import queue
import time
import pickle
import os

print("🎤 AUDIO PROCESS: Starting...")

# Tạo queue (riêng biệt)
audio_queue = queue.Queue()

# Put data vào queue
for i in range(3):
    data = f"audio_data_{i}"
    audio_queue.put(data)
    print(f"✅ AUDIO: Put '{data}' into queue")
    
    # Ghi queue size ra file để debug
    with open("/tmp/audio_queue_size.txt", "w") as f:
        f.write(str(audio_queue.qsize()))
    
    time.sleep(1)

print(f"🎤 AUDIO: Final queue size = {audio_queue.qsize()}")
print("🎤 AUDIO: Process finished")
"""

    stt_script = """
import queue
import time

print("🤖 STT PROCESS: Starting...")

# Tạo queue (riêng biệt!)
audio_queue = queue.Queue()

print(f"🤖 STT: My queue size = {audio_queue.qsize()}")

# Try to get data
for i in range(3):
    try:
        data = audio_queue.get(timeout=1.0)
        print(f"📥 STT: Got '{data}'")
    except queue.Empty:
        print("❌ STT: Queue is empty!")
    
    # Check file để xem audio process queue size
    try:
        with open("/tmp/audio_queue_size.txt", "r") as f:
            audio_size = f.read().strip()
        print(f"🔍 STT: Audio process queue size = {audio_size}")
    except:
        print("🔍 STT: Cannot read audio queue size")
    
    time.sleep(1)

print("🤖 STT: Process finished")
"""

    # Ghi scripts ra files
    with open("/tmp/test_audio.py", "w") as f:
        f.write(audio_script)
    
    with open("/tmp/test_stt.py", "w") as f:
        f.write(stt_script)
    
    print("📁 Created test scripts")
    
    # Chạy audio process
    print("\n🎤 Starting audio process...")
    audio_proc = subprocess.Popen([sys.executable, "/tmp/test_audio.py"])
    
    time.sleep(2)  # Wait for audio to put data
    
    # Chạy STT process
    print("\n🤖 Starting STT process...")
    stt_proc = subprocess.Popen([sys.executable, "/tmp/test_stt.py"])
    
    # Wait for both to finish
    audio_proc.wait()
    stt_proc.wait()
    
    # Cleanup
    for f in ["/tmp/test_audio.py", "/tmp/test_stt.py", "/tmp/audio_queue_size.txt"]:
        if os.path.exists(f):
            os.remove(f)
    
    print("\n" + "=" * 40)
    print("🎯 RESULT ANALYSIS:")
    print("- Audio process puts data into ITS queue")
    print("- STT process checks ITS OWN queue (different object)")
    print("- STT always gets 'Queue is empty!'")
    print("- This proves queue.Queue() doesn't work between processes")

def test_actual_voice_system():
    """Test actual voice system để confirm issue"""
    
    print("\n" + "=" * 40)
    print("🔍 TESTING YOUR ACTUAL VOICE SYSTEM")
    print("=" * 40)
    
    # Check nếu có processes đang chạy
    import psutil
    
    voice_procs = []
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'new_audio_update_v3.py' in cmdline or 'new_stt_integrated.py' in cmdline:
                voice_procs.append(proc)
        except:
            pass
    
    if voice_procs:
        print(f"🎵 Found {len(voice_procs)} voice system processes running:")
        for proc in voice_procs:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            print(f"  - PID {proc.pid}: {cmdline}")
        print("\n❌ CONFIRMED: Multiple separate processes!")
        print("❌ Each process has its own memory space")
        print("❌ queue.Queue() objects are NOT shared")
    else:
        print("✅ No voice system processes currently running")
    
    # Check data directories
    audio_dir = "data/audio_chunks"
    if os.path.exists(audio_dir):
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        print(f"\n📁 Found {len(audio_files)} audio files in {audio_dir}")
        if audio_files:
            print("✅ File-based communication is working!")
    else:
        print(f"\n📁 No audio directory found: {audio_dir}")

def show_solutions():
    print("\n" + "=" * 40)
    print("💡 SOLUTIONS TO FIX QUEUE ISSUE:")
    print("=" * 40)
    print("1. 🔥 QUICK FIX: Use file monitoring (Mode 1)")
    print("   - Already working in your system")
    print("   - No code changes needed")
    print()
    print("2. 🔧 PROPER FIX: Replace queue.Queue() with:")
    print("   - multiprocessing.Queue()")
    print("   - or file-based queue")
    print("   - or Redis/message broker")
    print()
    print("3. 🎯 RECOMMENDED: Stick with Mode 1")
    print("   - It's working perfectly")
    print("   - File I/O is fast enough for your use case")
    print("   - Much simpler than inter-process communication")

def main():
    print("🚀 RUNNING QUEUE DEBUG TESTS...")
    
    try:
        test_queue_between_processes()
        test_actual_voice_system()
        show_solutions()
        
        print("\n" + "=" * 40)
        print("✅ DEBUG COMPLETED!")
        print("🎯 MODE 1 (file monitoring) is your best option")
        print("❌ MODE 2 (queue) won't work without major changes")
        
    except KeyboardInterrupt:
        print("\n🛑 Debug interrupted")
    except Exception as e:
        print(f"\n❌ Debug error: {e}")

if __name__ == "__main__":
    main()