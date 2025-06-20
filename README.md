# rpi_phoWhisper

A lightweight pipeline on Raspberry Pi 5 for recording audio with wake word detection, transcribing with a custom PhoWhisper model, and (optionally) processing transcriptions via a PhoBERT NLU model.

---

## 📖 Overview

This repository provides scripts to:

1. **Wake Word Detection** - Listen for "computer" using Porcupine engine
2. **Smart Audio Recording** - Record and trim wake word from user speech using VAD
3. **Speech-to-Text** - Transcribe audio using custom-finetuned PhoWhisper model
4. **(Optional) NLU** - Perform intent classification using custom-finetuned PhoBERT model
5. **Easy Startup** - One-command system launch with automated process management

Designed for ease of use: clone, install dependencies, add your models, and run with a single script.

---

## 🚀 Prerequisites

* **Hardware**: Raspberry Pi 5 (16 GB RAM) with USB microphone
* **OS**: Raspberry Pi OS (64‑bit)
* **Python**: 3.11+ (installed via system)
* **Git**: for cloning the repo

---

## 📂 Repository Structure

```plaintext
rpi_phoWhisper/
├─ README.md
├─ start_voice_system.sh          ← Main startup script
├─ system/
│  ├─ install_test.sh             ← Dependency installation
│  └─ voice_system_launcher_v2.py ← Alternative launcher
├─ python/
│  ├─ new_audio_update_v3.py      ← Wake word detection + audio recording
│  ├─ new_stt_intergrated_fix.py  ← Speech-to-text service
│  └─ daemon_nlu.py               ← (Optional) NLU service
├─ models/
│  ├─ phowhisper_multistage/      ← Custom PhoWhisper model files
│  └─ phobert/                    ← (Optional) Custom PhoBERT model files
├─ data/                          ← Auto-created directories
│  ├─ audio_chunks/               ← Raw + processed audio files
│  ├─ transcriptions/             ← STT output files
│  └─ nlu_results/                ← NLU output files
└─ venv/                          ← Python virtual environment
```

---

## ⚙️ Setup

### 1. Clone and Setup Environment

```bash
git clone https://github.com/quanghuytv12/rpi_phoWhisper.git
cd rpi_phoWhisper

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install system and Python dependencies
bash system/install_test.sh
```

This installs:
- **System packages**: `portaudio19-dev`
- **Python libraries**: `pvporcupine`, `transformers`, `torch`, `torchaudio`, `scipy`, `sounddevice`, `silero-vad`, `psutil`, `librosa`

### 3. Add Your Models

**Required**: PhoWhisper model
```bash
# Copy your fine-tuned PhoWhisper model to:
models/phowhisper_multistage/
├─ config.json
├─ pytorch_model.bin
├─ preprocessor_config.json
├─ tokenizer.json
└─ ...
```

**Optional**: PhoBERT NLU model
```bash
# Copy your fine-tuned PhoBERT model to:
models/phobert/
├─ intent_classifier_final/
├─ ner_model_final/
├─ intent_encoder.pkl
└─ ...
```

### 4. Configure Audio Device (if needed)

```bash
# List available audio devices
arecord -l

# Edit python/new_audio_update_v3.py if needed:
# DEVICE_INDEX = 1  # Change to your USB mic index
```

---

## ▶️ Usage

### Quick Start (Recommended)

```bash
# Ensure virtual environment is active
source venv/bin/activate

# Start the complete voice system
bash start_voice_system.sh
```

The script will:
1. ✅ Check all required files exist
2. 📁 Create necessary directories
3. 🤖 Ask you to choose STT processing method:
   - **1. File monitoring** (recommended)
   - **2. Queue processing** 
   - **3. Both**
4. 🚀 Start audio detection and STT services
5. 🎤 Begin listening for wake word "computer"

### System Operation

1. **Say "computer"** - Wake word activates recording
2. **Speak your command** - System records until silence detected
3. **Automatic processing**:
   - ✂️ Wake word automatically trimmed from audio
   - 🎤 Speech transcribed using PhoWhisper
   - 💾 Results saved to `data/transcriptions/`

### Manual Testing

**Test wake word detection only:**
```bash
python3 python/new_audio_update_v3.py
# Say "computer" and speak - check data/audio_chunks/
```

**Test transcription only:**
```bash
python3 python/new_stt_intergrated_fix.py --method 1
# Place .wav files in data/audio_chunks/
```

**View results:**
```bash
# Check generated audio files
ls data/audio_chunks/

# Check transcriptions
cat data/transcriptions/*.txt
```

---

## 🔧 Configuration

### Wake Word Settings
```python
# In python/new_audio_update_v3.py
WAKEWORD = "computer"           # Change wake word
DEVICE_INDEX = 1                # USB microphone device
sensitivities=[0.95]           # Wake word sensitivity (0.0-1.0)
```

### Audio Processing
```python
SILENCE_DURATION = 1.0          # Seconds of silence before stopping
BUFFER_SECONDS = 5              # Rolling buffer size
VAD_THRESHOLD = 0.02            # Voice activity threshold
```

### STT Model Path
```python
# In python/new_stt_intergrated_fix.py
PHOWHISPER_MODEL = "models/phowhisper_multistage"
```

---

## 🔍 Troubleshooting

### Audio Issues
```bash
# Check microphone
arecord -l
arecord -D hw:1,0 -d 5 test.wav

# Check audio permissions
sudo usermod -a -G audio $USER
```

### Model Loading Issues
```bash
# Verify model structure
ls -la models/phowhisper_multistage/
# Should contain: config.json, pytorch_model.bin, etc.

# Check Python dependencies
pip list | grep transformers
```

### Process Management
```bash
# Check running processes
ps aux | grep python.*audio
ps aux | grep python.*stt

# Force stop if needed
pkill -f new_audio_update_v3.py
pkill -f new_stt_intergrated_fix.py
```

### File Monitoring Mode (Recommended)
- ✅ Most reliable method
- ✅ Works across separate processes
- ✅ No inter-process communication issues
- ⚡ Files processed in ~500ms

---

## 📊 Performance

**Raspberry Pi 5 (16GB) Performance:**
- **Wake word detection**: <100ms latency
- **Audio processing**: Real-time with VAD
- **STT transcription**: 1-3 seconds (depends on audio length)
- **Wake word trimming**: Automatic using hybrid VAD+energy detection

---

## 🛠️ Development

### Adding New Features
```bash
# Test individual components
python3 python/new_audio_update_v3.py     # Audio recording
python3 python/new_stt_intergrated_fix.py # STT service
python3 python/daemon_nlu.py              # NLU service
```

### Debug Mode
```bash
# Run with verbose output
python3 quick_debug_test.py  # Test queue communication
```

---

## 🤝 Contributing

Feel free to open issues or pull requests for improvements:
- Better audio preprocessing
- Additional language models
- Docker containerization
- systemd service integration
- Performance optimizations

---

## 📄 License

MIT License. See [LICENSE](LICENSE) if provided.

---

## 🔗 Related Projects

- [PhoWhisper](https://github.com/VinAIResearch/PhoWhisper) - Vietnamese Speech Recognition
- [PhoBERT](https://github.com/VinAIResearch/PhoBERT) - Vietnamese BERT
- [Porcupine](https://github.com/Picovoice/porcupine) - Wake Word Detection
