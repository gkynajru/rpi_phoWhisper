import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal
import librosa

def debug_audio_pipeline(original_file, processed_file):
    """Compare original vs processed audio visually"""
    
    # Load audio files
    orig_audio, orig_sr = sf.read(original_file)
    proc_audio, proc_sr = sf.read(processed_file)
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Time domain comparison
    orig_time = np.linspace(0, len(orig_audio)/orig_sr, len(orig_audio))
    proc_time = np.linspace(0, len(proc_audio)/proc_sr, len(proc_audio))
    
    axes[0,0].plot(orig_time, orig_audio)
    axes[0,0].set_title(f'Original Audio ({orig_sr}Hz)')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Amplitude')
    
    axes[0,1].plot(proc_time, proc_audio)
    axes[0,1].set_title(f'Processed Audio ({proc_sr}Hz)')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Amplitude')
    
    # Frequency spectrum
    orig_fft = np.abs(np.fft.fft(orig_audio))
    proc_fft = np.abs(np.fft.fft(proc_audio))
    
    orig_freqs = np.fft.fftfreq(len(orig_audio), 1/orig_sr)
    proc_freqs = np.fft.fftfreq(len(proc_audio), 1/proc_sr)
    
    axes[1,0].plot(orig_freqs[:len(orig_freqs)//2], 
                   20*np.log10(orig_fft[:len(orig_fft)//2] + 1e-10))
    axes[1,0].set_title('Original Frequency Spectrum')
    axes[1,0].set_xlabel('Frequency (Hz)')
    axes[1,0].set_ylabel('Magnitude (dB)')
    axes[1,0].set_xlim(0, orig_sr/2)
    
    axes[1,1].plot(proc_freqs[:len(proc_freqs)//2], 
                   20*np.log10(proc_fft[:len(proc_fft)//2] + 1e-10))
    axes[1,1].set_title('Processed Frequency Spectrum')
    axes[1,1].set_xlabel('Frequency (Hz)')
    axes[1,1].set_ylabel('Magnitude (dB)')
    axes[1,1].set_xlim(0, proc_sr/2)
    
    # Spectrograms
    f_orig, t_orig, Sxx_orig = signal.spectrogram(orig_audio, orig_sr, nperseg=1024)
    f_proc, t_proc, Sxx_proc = signal.spectrogram(proc_audio, proc_sr, nperseg=1024)
    
    im1 = axes[2,0].pcolormesh(t_orig, f_orig, 10*np.log10(Sxx_orig + 1e-10))
    axes[2,0].set_title('Original Spectrogram')
    axes[2,0].set_xlabel('Time (s)')
    axes[2,0].set_ylabel('Frequency (Hz)')
    
    im2 = axes[2,1].pcolormesh(t_proc, f_proc, 10*np.log10(Sxx_proc + 1e-10))
    axes[2,1].set_title('Processed Spectrogram')
    axes[2,1].set_xlabel('Time (s)')
    axes[2,1].set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print("🔍 AUDIO ANALYSIS REPORT")
    print("=" * 40)
    print(f"Original: {len(orig_audio)} samples @ {orig_sr}Hz")
    print(f"Processed: {len(proc_audio)} samples @ {proc_sr}Hz")
    print(f"Duration change: {len(proc_audio)/proc_sr:.3f}s vs {len(orig_audio)/orig_sr:.3f}s")
    
    # Check for aliasing (high frequency content in processed)
    nyquist_proc = proc_sr / 2
    high_freq_content = np.sum(proc_fft[proc_freqs > nyquist_proc * 0.8])
    total_content = np.sum(proc_fft)
    aliasing_ratio = high_freq_content / total_content
    
    print(f"High frequency content ratio: {aliasing_ratio:.4f}")
    if aliasing_ratio > 0.1:
        print("⚠️  WARNING: High frequency content detected - possible aliasing!")
    else:
        print("✅ No significant aliasing detected")

def test_pipeline_with_known_audio():
    """Test your current pipeline with a known good audio file"""
    print("🧪 PIPELINE TEST WITH KNOWN AUDIO")
    print("=" * 45)
    
    # Generate test tone (known signal)
    duration = 2.0  # seconds
    sr_high = 48000
    sr_low = 16000
    
    # Create test signal: mix of frequencies
    t = np.linspace(0, duration, int(duration * sr_high))
    test_signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +      # A4 note
        0.3 * np.sin(2 * np.pi * 880 * t) +      # A5 note  
        0.2 * np.sin(2 * np.pi * 1760 * t) +     # A6 note
        0.1 * np.sin(2 * np.pi * 3520 * t)       # A7 note (near Nyquist for 16kHz)
    )
    
    # Add some noise
    test_signal += 0.05 * np.random.randn(len(test_signal))
    
    # Normalize
    test_signal = test_signal / np.max(np.abs(test_signal)) * 0.8
    
    # Save original
    sf.write("test_signal_48k.wav", test_signal, sr_high)
    
    # Test your current method
    simple_downsampled = test_signal[::3]  # Your current method
    sf.write("test_signal_simple.wav", simple_downsampled, sr_low)
    
    # Test proper method
    proper_resampled = librosa.resample(test_signal, orig_sr=sr_high, target_sr=sr_low)
    sf.write("test_signal_proper.wav", proper_resampled, sr_low)
    
    # Compare
    debug_audio_pipeline("test_signal_48k.wav", "test_signal_simple.wav")
    debug_audio_pipeline("test_signal_48k.wav", "test_signal_proper.wav")
    
    print("\n📝 Files created for manual listening:")
    print("  - test_signal_48k.wav (original)")
    print("  - test_signal_simple.wav (your current method)")  
    print("  - test_signal_proper.wav (proper resampling)")
    print("\n🎧 Listen to these files to hear the difference!")

if __name__ == "__main__":
    test_pipeline_with_known_audio()