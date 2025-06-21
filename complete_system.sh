#!/bin/bash

# Complete Voice System Startup Script
# Full pipeline: Audio Detection -> STT -> NLU

echo "============================================================"
echo "🎤 COMPLETE VOICE SYSTEM STARTUP"
echo "============================================================"
echo "Pipeline: Audio Detection -> Speech-to-Text -> NLU"
echo "Components: Wake Word + Recording + Transcription + Intent/NER"
echo "============================================================"

# Check if all required files exist
echo "🔍 Checking required components..."

if [ ! -f "python/new_audio_update_v3.py" ]; then
    echo "❌ Audio script not found: python/new_audio_update_v3.py"
    exit 1
fi

if [ ! -f "python/new_stt_integrated.py" ]; then
    echo "❌ STT script not found: python/new_stt_integrated.py"
    exit 1
fi

if [ ! -f "python/new_nlu_integrated.py" ]; then
    echo "❌ NLU script not found: python/new_nlu_integrated.py"
    exit 1
fi

# Check model directories
echo "🔍 Checking model directories..."

if [ ! -d "models/phowhisper_multistage" ]; then
    echo "⚠️ PhoWhisper model not found: models/phowhisper_multistage"
    echo "   STT may not work properly"
fi

if [ ! -d "models/phobert/intent_classifier_final" ]; then
    echo "⚠️ Intent model not found: models/phobert/intent_classifier_final"
    echo "   NLU may not work properly"
fi

if [ ! -d "models/phobert/ner_model_final" ]; then
    echo "⚠️ NER model not found: models/phobert/ner_model_final"
    echo "   NLU may not work properly"
fi

echo "✅ Component check completed"

# Create all necessary directories
echo "📁 Creating data directories..."
mkdir -p data/audio_chunks
mkdir -p data/transcriptions
mkdir -p data/nlu_results

echo "✅ Directories created"

# Ask user for processing method
echo ""
echo "Processing Method Selection:"
echo "============================================================"
echo "1. File monitoring (RECOMMENDED)"
echo "   - Audio saves files -> STT monitors files -> NLU monitors transcriptions"
echo "   - Most reliable, proven to work"
echo "   - Latency: ~2-3 seconds total"
echo ""
echo "2. Queue processing (EXPERIMENTAL)"
echo "   - Faster but requires process communication fixes"
echo "   - May not work due to inter-process queue limitations"
echo ""
echo "3. Both (NOT RECOMMENDED)"
echo "   - Runs both methods simultaneously"
echo "   - Resource intensive"
echo "============================================================"
echo ""
read -p "Enter method (1/2/3) [default=1]: " method
method=${method:-1}

# Validate method
if [[ ! "$method" =~ ^[123]$ ]]; then
    echo "❌ Invalid method, using default (1)"
    method=1
fi

echo ""
echo "🎯 Selected method: $method"
case $method in
    1) echo "   📁 File monitoring - Most reliable" ;;
    2) echo "   📦 Queue processing - Experimental" ;;
    3) echo "   🔄 Both methods - Resource intensive" ;;
esac

echo ""
echo "🚀 Starting Complete Voice System..."
echo "============================================================"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down complete voice system..."
    echo "Stopping all components..."
    
    # Kill all background processes
    if [ ! -z "$AUDIO_PID" ]; then
        echo "🔄 Stopping Audio Detection (PID: $AUDIO_PID)..."
        kill $AUDIO_PID 2>/dev/null
        wait $AUDIO_PID 2>/dev/null
        echo "✅ Audio Detection stopped"
    fi
    
    if [ ! -z "$STT_PID" ]; then
        echo "🔄 Stopping STT Service (PID: $STT_PID)..."
        kill $STT_PID 2>/dev/null
        wait $STT_PID 2>/dev/null
        echo "✅ STT Service stopped"
    fi
    
    if [ ! -z "$NLU_PID" ]; then
        echo "🔄 Stopping NLU Service (PID: $NLU_PID)..."
        kill $NLU_PID 2>/dev/null
        wait $NLU_PID 2>/dev/null
        echo "✅ NLU Service stopped"
    fi
    
    echo "✅ Complete voice system shutdown completed"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Step 1: Start Audio Detection
echo "🎤 STEP 1: Starting Audio Detection..."
echo "   - Listening for wake word 'computer'"
echo "   - Recording speech with VAD"
echo "   - Saving audio files to data/audio_chunks/"
echo ""

python3 python/new_audio_update_v3.py &
AUDIO_PID=$!
echo "✅ Audio Detection started (PID: $AUDIO_PID)"

# Wait for audio system to fully initialize
echo "⏳ Waiting for audio system to initialize..."
echo "   - Loading Porcupine wake word detection..."
echo "   - Loading Silero VAD model..."
echo "   - Opening audio stream..."
sleep 5

# Step 2: Start STT Service  
echo ""
echo "🤖 STEP 2: Starting STT Service..."
echo "   - Loading PhoWhisper model..."
echo "   - Monitoring audio files in data/audio_chunks/"
echo "   - Saving transcriptions to data/transcriptions/"
echo ""

python3 python/new_stt_integrated.py --method $method &
STT_PID=$!
echo "✅ STT Service started (PID: $STT_PID)"

# Wait for STT models to load
echo "⏳ Waiting for STT models to load..."
echo "   - This may take 30-60 seconds on first run..."
sleep 8

# Step 3: Start NLU Service
echo ""
echo "🧠 STEP 3: Starting NLU Service..."
echo "   - Loading PhoBERT Intent Classifier..."
echo "   - Loading PhoBERT NER model..."  
echo "   - Monitoring transcriptions in data/transcriptions/"
echo "   - Saving NLU results to data/nlu_results/"
echo ""

python3 python/new_nlu_integrated.py --method $method &
NLU_PID=$!
echo "✅ NLU Service started (PID: $NLU_PID)"

# Wait for NLU models to load
echo "⏳ Waiting for NLU models to load..."
echo "   - Loading intent classification model..."
echo "   - Loading named entity recognition model..."
sleep 5

# System ready
echo ""
echo "============================================================"
echo "✅ COMPLETE VOICE SYSTEM READY!"
echo "============================================================"
echo "🎯 FULL PIPELINE ACTIVE:"
echo ""
echo "1. 🎤 Audio Detection:"
echo "   - Say 'computer' to trigger recording"
echo "   - Speak your command clearly"
echo "   - System automatically detects end of speech"
echo ""
echo "2. 🤖 Speech-to-Text:"
echo "   - Converts audio to Vietnamese text"
echo "   - Saves transcription files"
echo ""
echo "3. 🧠 Natural Language Understanding:"
echo "   - Extracts intent from your command"
echo "   - Identifies named entities"
echo "   - Saves structured results"
echo ""
echo "📁 DATA FLOW:"
echo "   Audio files: data/audio_chunks/"
echo "   Transcriptions: data/transcriptions/"  
echo "   NLU results: data/nlu_results/"
echo ""
echo "📊 EXPECTED LATENCY:"
case $method in
    1) echo "   Total processing time: 2-4 seconds" ;;
    2) echo "   Total processing time: 1-2 seconds" ;;
    3) echo "   Total processing time: 1-4 seconds" ;;
esac
echo ""
echo "💡 USAGE:"
echo "   1. Say 'computer' clearly"
echo "   2. Wait for recording indicator"  
echo "   3. Speak your command (e.g., 'bật điện phòng khách')"
echo "   4. Wait for processing to complete"
echo "   5. Check results in data/nlu_results/"
echo ""
echo "🛑 Press Ctrl+C to stop the complete system"
echo "============================================================"

# Monitor system health
echo ""
echo "🔍 System monitoring started..."
echo "   Monitoring all three components for failures..."
echo ""

# Keep script running and monitor processes
while true; do
    # Check if any process has died
    if ! kill -0 $AUDIO_PID 2>/dev/null; then
        echo "💀 Audio Detection process has died!"
        cleanup
    fi
    
    if ! kill -0 $STT_PID 2>/dev/null; then
        echo "💀 STT Service process has died!"
        cleanup
    fi
    
    if ! kill -0 $NLU_PID 2>/dev/null; then
        echo "💀 NLU Service process has died!"
        cleanup
    fi
    
    # Optional: Show periodic status
    sleep 30
    echo "📊 System status: All components running ($(date))"
    
    # Show file counts
    audio_count=$(ls data/audio_chunks/*.wav 2>/dev/null | wc -l)
    transcript_count=$(ls data/transcriptions/*.txt 2>/dev/null | wc -l)
    nlu_count=$(ls data/nlu_results/*.json 2>/dev/null | wc -l)
    
    echo "   📁 Files: Audio($audio_count) -> Transcripts($transcript_count) -> NLU($nlu_count)"
done