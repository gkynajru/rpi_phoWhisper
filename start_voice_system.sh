#!/bin/bash

# Voice System Startup Script
# Makes it super easy to start the voice system

echo "============================================================"
echo "🎤 VOICE SYSTEM STARTUP"
echo "============================================================"

# Check if files exist
if [ ! -f "python/new_audio_update_v3.py" ]; then
    echo "❌ Audio script not found: python/new_audio_update_v3.py"
    exit 1
fi

if [ ! -f "python/new_stt_update.py" ]; then
    echo "❌ STT script not found: python/new_stt_update.py"
    exit 1
fi

# Create directories
mkdir -p data/audio_chunks
mkdir -p data/transcriptions

echo "📁 Directories created"

# Ask user for STT method
echo ""
echo "STT Processing Method:"
echo "1. File monitoring (recommended)"
echo "2. Queue processing"
echo "3. Both"
echo ""
read -p "Enter (1/2/3) [default=1]: " method
method=${method:-1}

# Validate method
if [[ ! "$method" =~ ^[123]$ ]]; then
    echo "❌ Invalid method, using default (1)"
    method=1
fi

echo ""
echo "🚀 Starting Voice System..."
echo "Method: $method"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down voice system..."
    kill $AUDIO_PID 2>/dev/null
    kill $STT_PID 2>/dev/null
    wait $AUDIO_PID 2>/dev/null
    wait $STT_PID 2>/dev/null
    echo "✅ Voice system stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start audio detection in background
echo "🎤 Starting audio detection..."
python3 python/new_audio_update_v3.py &
AUDIO_PID=$!
echo "✅ Audio detection started (PID: $AUDIO_PID)"

# Wait for audio system to initialize
echo "⏳ Waiting for audio system to initialize..."
sleep 3

# Start STT service in background
echo "🤖 Starting STT service..."
python3 python/new_stt_integrated.py --method $method &
STT_PID=$!
echo "✅ STT service started (PID: $STT_PID)"

# Wait a bit more
sleep 2

echo ""
echo "============================================================"
echo "✅ VOICE SYSTEM READY!"
echo "============================================================"
echo "🎤 Say 'computer' to activate voice recording"
echo "🤖 Speech will be automatically transcribed"
echo "📁 Audio files: data/audio_chunks/"
echo "📝 Transcriptions: data/transcriptions/"
echo "🛑 Press Ctrl+C to stop"
echo "============================================================"

# Wait for processes to finish
wait $AUDIO_PID $STT_PID