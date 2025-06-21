#!/bin/bash

# Quiet Voice System Launcher
# Starts the complete pipeline with minimal console output

echo "🎤 Starting Voice System (Quiet Mode)..."

# Create log directory
mkdir -p logs

# Get timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping voice system..."
    kill $AUDIO_PID $STT_PID $NLU_PID 2>/dev/null
    wait $AUDIO_PID $STT_PID $NLU_PID 2>/dev/null
    echo "✅ Voice system stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start components with output redirected to log files
echo "🚀 Starting components..."

# Audio Detection (redirect output to log)
python3 python/new_audio_update_v3.py > logs/audio_${TIMESTAMP}.log 2>&1 &
AUDIO_PID=$!
echo "✅ Audio Detection started (PID: $AUDIO_PID)"

sleep 3

# STT Service (redirect output to log)
python3 python/new_stt_integrated.py --method 1 > logs/stt_${TIMESTAMP}.log 2>&1 &
STT_PID=$!
echo "✅ STT Service started (PID: $STT_PID)"

sleep 5

# NLU Service (redirect output to log)
python3 python/new_nlu_integrated.py --method 1 > logs/nlu_${TIMESTAMP}.log 2>&1 &
NLU_PID=$!
echo "✅ NLU Service started (PID: $NLU_PID)"

sleep 3

echo ""
echo "🎯 VOICE SYSTEM READY (Quiet Mode)"
echo "=" * 40
echo "🔇 Console output redirected to logs/"
echo "📁 Log files:"
echo "   • Audio: logs/audio_${TIMESTAMP}.log"
echo "   • STT: logs/stt_${TIMESTAMP}.log" 
echo "   • NLU: logs/nlu_${TIMESTAMP}.log"
echo ""
echo "💡 Usage:"
echo "   1. Say 'computer' to activate"
echo "   2. Speak your Vietnamese command"
echo "   3. Check results with monitor scripts"
echo ""
echo "📊 Monitor options:"
echo "   • python3 clean_pipeline_monitor.py"
echo "   • python3 simple_dashboard.py"
echo "   • python3 results_only_monitor.py"
echo ""
echo "🛑 Press Ctrl+C to stop system"
echo "=" * 40

# Keep running until interrupted
while true; do
    # Check if processes are still running
    if ! kill -0 $AUDIO_PID 2>/dev/null; then
        echo "💀 Audio process died"
        cleanup
    fi
    if ! kill -0 $STT_PID 2>/dev/null; then
        echo "💀 STT process died"
        cleanup
    fi
    if ! kill -0 $NLU_PID 2>/dev/null; then
        echo "💀 NLU process died"
        cleanup
    fi
    
    sleep 10
done