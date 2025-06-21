#!/usr/bin/env python3
"""
Clean Pipeline Monitor
Chỉ hiển thị kết quả quan trọng, filter bỏ noise
"""

import os
import time
import json
from datetime import datetime
import subprocess
import signal
import sys

class CleanMonitor:
    def __init__(self):
        self.audio_dir = "data/audio_chunks"
        self.transcript_dir = "data/transcriptions"
        self.nlu_dir = "data/nlu_results"
        self.running = True
        
        # Track processed files
        self.last_counts = self.get_file_counts()
        self.session_start = datetime.now()
        
        # Results storage
        self.results = []
    
    def get_file_counts(self):
        """Get current file counts"""
        counts = {}
        dirs = {
            "audio": self.audio_dir,
            "transcripts": self.transcript_dir,
            "nlu": self.nlu_dir
        }
        
        for name, path in dirs.items():
            if os.path.exists(path):
                if name == "audio":
                    counts[name] = len([f for f in os.listdir(path) if f.endswith('.wav') and 'speech_' in f])
                elif name == "transcripts":
                    counts[name] = len([f for f in os.listdir(path) if f.endswith('.txt')])
                else:  # nlu
                    counts[name] = len([f for f in os.listdir(path) if f.endswith('.json')])
            else:
                counts[name] = 0
        return counts
    
    def get_latest_file(self, directory, extension):
        """Get the most recent file in directory"""
        if not os.path.exists(directory):
            return None
            
        files = [f for f in os.listdir(directory) if f.endswith(extension)]
        if not files:
            return None
            
        files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))
        return files[-1]
    
    def read_latest_transcript(self):
        """Read the latest transcription"""
        latest_file = self.get_latest_file(self.transcript_dir, '.txt')
        if not latest_file:
            return None
            
        try:
            with open(os.path.join(self.transcript_dir, latest_file), 'r', encoding='utf-8') as f:
                return f.read().strip()
        except:
            return None
    
    def read_latest_nlu(self):
        """Read the latest NLU result"""
        latest_file = self.get_latest_file(self.nlu_dir, '.json')
        if not latest_file:
            return None
            
        try:
            with open(os.path.join(self.nlu_dir, latest_file), 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    
    def format_nlu_result(self, nlu_data):
        """Format NLU result for clean display"""
        if not nlu_data:
            return "Error reading NLU data"
            
        intent_info = nlu_data.get('intent', {})
        intent_name = intent_info.get('intent', 'unknown')
        confidence = intent_info.get('confidence', 0.0)
        
        entities = nlu_data.get('entities', [])
        entity_summary = []
        for entity in entities:
            entity_type = entity.get('entity', 'unknown')
            entity_text = entity.get('text', '')
            entity_summary.append(f"{entity_type}:{entity_text}")
        
        return {
            'intent': intent_name,
            'confidence': f"{confidence:.2f}",
            'entities': entity_summary
        }
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_header(self):
        """Print clean header"""
        now = datetime.now().strftime("%H:%M:%S")
        session_duration = datetime.now() - self.session_start
        duration_str = str(session_duration).split('.')[0]  # Remove microseconds
        
        print("=" * 70)
        print("🎤 VOICE PIPELINE MONITOR")
        print("=" * 70)
        print(f"⏰ Time: {now} | Session: {duration_str}")
        print(f"📊 Status: {'🟢 Running' if self.running else '🔴 Stopped'}")
        print("=" * 70)
    
    def print_summary(self, counts):
        """Print file count summary"""
        total_processed = counts['audio'] - self.last_counts['audio']
        if total_processed > 0:
            stt_efficiency = (counts['transcripts'] - self.last_counts['transcripts']) / total_processed * 100
            nlu_efficiency = (counts['nlu'] - self.last_counts['nlu']) / total_processed * 100
        else:
            stt_efficiency = nlu_efficiency = 0
        
        print(f"📁 Files: Audio({counts['audio']}) → STT({counts['transcripts']}) → NLU({counts['nlu']})")
        print(f"📈 Efficiency: STT={stt_efficiency:.0f}% | NLU={nlu_efficiency:.0f}%")
        print()
    
    def print_latest_results(self):
        """Print only the latest results"""
        print("🎯 LATEST RESULTS:")
        print("-" * 50)
        
        # Latest transcript
        transcript = self.read_latest_transcript()
        if transcript:
            print(f"📝 Transcription: '{transcript}'")
        else:
            print("📝 Transcription: No data yet")
        
        # Latest NLU
        nlu_data = self.read_latest_nlu()
        if nlu_data:
            nlu_formatted = self.format_nlu_result(nlu_data)
            print(f"🎯 Intent: {nlu_formatted['intent']} (confidence: {nlu_formatted['confidence']})")
            if nlu_formatted['entities']:
                entities_str = ', '.join(nlu_formatted['entities'])
                print(f"📍 Entities: {entities_str}")
            else:
                print("📍 Entities: None detected")
        else:
            print("🎯 Intent: No data yet")
            print("📍 Entities: No data yet")
        
        print("-" * 50)
    
    def print_recent_history(self):
        """Print recent results history"""
        if len(self.results) == 0:
            print("📚 History: No commands processed yet")
            return
            
        print("📚 RECENT HISTORY:")
        print("-" * 30)
        
        # Show last 3 results
        recent = self.results[-3:] if len(self.results) >= 3 else self.results
        for i, result in enumerate(recent, 1):
            time_str = result['time'].strftime("%H:%M:%S")
            print(f"{i}. [{time_str}] '{result['transcript']}'")
            print(f"   → {result['intent']} ({result['confidence']})")
            if result['entities']:
                print(f"   → Entities: {', '.join(result['entities'])}")
        print("-" * 30)
    
    def print_instructions(self):
        """Print usage instructions"""
        print()
        print("💡 INSTRUCTIONS:")
        print("• Say 'computer' to activate")
        print("• Speak your Vietnamese command")
        print("• Results will appear here automatically")
        print("• Press Ctrl+C to stop monitoring")
        print()
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        self.running = False
        print("\n🛑 Stopping monitor...")
        sys.exit(0)
    
    def run(self):
        """Main monitoring loop"""
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("🚀 Starting clean pipeline monitor...")
        print("💡 This will show only important results, no debug logs")
        time.sleep(2)
        
        while self.running:
            try:
                self.clear_screen()
                self.print_header()
                
                # Get current counts
                current_counts = self.get_file_counts()
                self.print_summary(current_counts)
                
                # Check for new results
                if (current_counts['transcripts'] > self.last_counts['transcripts'] or 
                    current_counts['nlu'] > self.last_counts['nlu']):
                    
                    # Store new result
                    transcript = self.read_latest_transcript()
                    nlu_data = self.read_latest_nlu()
                    
                    if transcript and nlu_data:
                        nlu_formatted = self.format_nlu_result(nlu_data)
                        self.results.append({
                            'time': datetime.now(),
                            'transcript': transcript,
                            'intent': nlu_formatted['intent'],
                            'confidence': nlu_formatted['confidence'],
                            'entities': nlu_formatted['entities']
                        })
                
                # Display current results
                self.print_latest_results()
                self.print_recent_history()
                self.print_instructions()
                
                # Update last counts
                self.last_counts = current_counts
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"❌ Monitor error: {e}")
                time.sleep(1)

def main():
    monitor = CleanMonitor()
    monitor.run()

if __name__ == "__main__":
    main()