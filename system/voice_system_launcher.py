#!/usr/bin/env python3
"""
Voice System Launcher
Khởi động cả hệ thống Wake Word Detection + STT
"""
import subprocess
import time
import os
import sys
import signal
import threading
from pathlib import Path

# Configuration
AUDIO_SCRIPT = "python/new_audio_update_v3.py"  # Your wake word detection script
STT_SCRIPT = "python/new_stt_update.py"     # STT service script

class VoiceSystemLauncher:
    def __init__(self):
        self.processes = []
        self.running = True
        
    def start_process(self, script_name, name):
        """Start a subprocess"""
        try:
            print(f"🚀 Starting {name}...")
            process = subprocess.Popen([
                sys.executable, script_name
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
               universal_newlines=True, bufsize=1)
            
            self.processes.append((process, name))
            print(f"✅ {name} started (PID: {process.pid})")
            return process
        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return None
    
    def monitor_process(self, process, name):
        """Monitor process output"""
        try:
            while self.running and process.poll() is None:
                line = process.stdout.readline()
                if line:
                    print(f"[{name}] {line.rstrip()}")
        except Exception as e:
            print(f"❌ Monitor error for {name}: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.shutdown()
    
    def shutdown(self):
        """Shutdown all processes"""
        self.running = False
        print("🛑 Shutting down voice system...")
        
        for process, name in self.processes:
            try:
                print(f"🔄 Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                    print(f"✅ {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    print(f"⚠️ Force killing {name}...")
                    process.kill()
                    process.wait()
                    print(f"✅ {name} force stopped")
                    
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
        
        print("✅ Voice system shutdown complete")
        sys.exit(0)
    
    def run(self):
        """Main launcher function"""
        print("=" * 60)
        print("🎤 VOICE SYSTEM LAUNCHER")
        print("=" * 60)
        print("Components:")
        print(f"  • Wake Word Detection: {AUDIO_SCRIPT}")
        print(f"  • Speech-to-Text: {STT_SCRIPT}")
        print("=" * 60)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Check if scripts exist
        if not os.path.exists(AUDIO_SCRIPT):
            print(f"❌ Audio script not found: {AUDIO_SCRIPT}")
            return
            
        if not os.path.exists(STT_SCRIPT):
            print(f"❌ STT script not found: {STT_SCRIPT}")
            return
        
        # Start audio detection process
        audio_process = self.start_process(AUDIO_SCRIPT, "Audio Detection")
        if not audio_process:
            print("❌ Failed to start audio detection, exiting...")
            return
        
        # Wait a bit for audio system to initialize
        time.sleep(3)
        
        # Start STT process
        stt_process = self.start_process(STT_SCRIPT, "STT Service")
        if not stt_process:
            print("❌ Failed to start STT service")
            self.shutdown()
            return
        
        # Start monitoring threads
        audio_monitor = threading.Thread(
            target=self.monitor_process, 
            args=(audio_process, "Audio Detection")
        )
        audio_monitor.daemon = True
        audio_monitor.start()
        
        stt_monitor = threading.Thread(
            target=self.monitor_process, 
            args=(stt_process, "STT Service")
        )
        stt_monitor.daemon = True
        stt_monitor.start()
        
        print("\n✅ Voice system fully operational!")
        print("💡 Say 'computer' to activate voice recording")
        print("🎤 Speech will be automatically transcribed")
        print("🛑 Press Ctrl+C to stop")
        print("-" * 60)
        
        try:
            # Keep main thread alive and monitor processes
            while self.running:
                # Check if any process has died
                dead_processes = []
                for process, name in self.processes:
                    if process.poll() is not None:
                        dead_processes.append((process, name))
                
                if dead_processes:
                    for process, name in dead_processes:
                        return_code = process.returncode
                        print(f"💀 {name} has died (exit code: {return_code})")
                    
                    print("🛑 One or more processes have died, shutting down system...")
                    self.shutdown()
                    break
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.shutdown()

if __name__ == "__main__":
    launcher = VoiceSystemLauncher()
    launcher.run()