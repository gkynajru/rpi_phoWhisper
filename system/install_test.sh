#!/bin/bash
# Install system packages
sudo apt-get update
sudo apt-get install -y portaudio19-dev

# Install Python dependencies for pipeline and testing
pip install pvporcupine transformers torch torchaudio scipy sounddevice silero-vad psutil librosa

# Create data directories
mkdir -p data/audio_chunks data/transcriptions data/nlu_results data/test_metrics