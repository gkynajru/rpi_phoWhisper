import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

class ResampleTester:
    def __init__(self, model_path):
        """Initialize STT model for testing"""
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
        self.model.eval()
    
    def simple_downsample(self, audio_data, factor=3):
        """Original method - potentially problematic"""
        return audio_data[::factor]
    
    def librosa_resample(self, audio_data, original_sr, target_sr):
        """Librosa with anti-aliasing"""
        return librosa.resample(audio_data.astype(np.float32), 
                               orig_sr=original_sr, target_sr=target_sr)
    
    def scipy_resample(self, audio_data, original_sr, target_sr):
        """Scipy signal resample"""
        num_samples = int(len(audio_data) * target_sr / original_sr)
        return signal.resample(audio_data, num_samples)
    
    def butter_lowpass_resample(self, audio_data, original_sr, target_sr):
        """Manual lowpass + downsample"""
        # Anti-aliasing filter
        nyquist = target_sr / 2
        cutoff = nyquist * 0.95  # 95% of Nyquist
        
        # Design filter
        sos = signal.butter(5, cutoff / (original_sr/2), 
                           btype='low', output='sos')
        
        # Apply filter
        filtered = signal.sosfilt(sos, audio_data)
        
        # Downsample
        factor = int(original_sr / target_sr)
        return filtered[::factor]
    
    def analyze_frequency_content(self, audio_data, sr, title="Audio"):
        """Generate spectrogram for visual inspection"""
        plt.figure(figsize=(12, 4))
        
        # Time domain
        plt.subplot(1, 2, 1)
        time = np.linspace(0, len(audio_data)/sr, len(audio_data))
        plt.plot(time, audio_data)
        plt.title(f"{title} - Time Domain")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        # Frequency domain
        plt.subplot(1, 2, 2)
        f, t, Sxx = signal.spectrogram(audio_data, sr, nperseg=1024)
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10))
        plt.title(f"{title} - Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label='Power (dB)')
        
        plt.tight_layout()
        plt.show()
    
    def calculate_audio_metrics(self, original, resampled, sr_orig, sr_new):
        """Calculate audio quality metrics"""
        # Align lengths for comparison (if needed)
        if len(original) != len(resampled):
            min_len = min(len(original), len(resampled))
            original = original[:min_len]
            resampled = resampled[:min_len]
        
        # Normalize for comparison
        orig_norm = original / np.max(np.abs(original))
        resamp_norm = resampled / np.max(np.abs(resampled))
        
        # Calculate metrics
        mse = np.mean((orig_norm - resamp_norm) ** 2)
        snr = 10 * np.log10(np.var(orig_norm) / mse) if mse > 0 else float('inf')
        
        # Frequency content analysis
        orig_fft = np.abs(np.fft.fft(orig_norm))
        resamp_fft = np.abs(np.fft.fft(resamp_norm))
        
        # High frequency content (potential aliasing)
        high_freq_ratio = np.sum(resamp_fft[len(resamp_fft)//2:]) / np.sum(resamp_fft)
        
        return {
            'mse': mse,
            'snr_db': snr,
            'high_freq_ratio': high_freq_ratio
        }
    
    def transcribe(self, audio_data, sample_rate=16000):
        """Transcribe audio using STT model"""
        try:
            # Normalize
            audio_float = audio_data.astype(np.float32)
            if np.max(np.abs(audio_float)) > 1.0:
                audio_float = audio_float / 32768.0
            
            # Process
            inputs = self.processor(audio_float, sampling_rate=sample_rate, return_tensors="pt")
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    max_new_tokens=128
                )
            
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            return f"ERROR: {e}"
    
    def test_resample_methods(self, audio_file_path, reference_text=None):
        """Compare different resampling methods"""
        print("🔬 RESAMPLING COMPARISON TEST")
        print("=" * 50)
        
        # Load original audio
        audio_48k, sr_orig = sf.read(audio_file_path)
        print(f"📁 Original: {len(audio_48k)} samples @ {sr_orig}Hz")
        
        target_sr = 16000
        methods = {
            'simple_downsample': lambda x: self.simple_downsample(x, factor=3),
            'librosa_resample': lambda x: self.librosa_resample(x, sr_orig, target_sr),
            'scipy_resample': lambda x: self.scipy_resample(x, sr_orig, target_sr),
            'butter_lowpass': lambda x: self.butter_lowpass_resample(x, sr_orig, target_sr)
        }
        
        results = {}
        
        for method_name, method_func in methods.items():
            print(f"\n🔄 Testing: {method_name}")
            
            # Resample
            resampled = method_func(audio_48k)
            
            # Calculate audio metrics
            metrics = self.calculate_audio_metrics(audio_48k[::3], resampled, sr_orig, target_sr)
            
            # Transcribe
            transcription = self.transcribe(resampled, target_sr)
            
            # Save resampled audio for manual inspection
            sf.write(f"resampled_{method_name}.wav", resampled, target_sr)
            
            results[method_name] = {
                'transcription': transcription,
                'audio_metrics': metrics,
                'audio_length': len(resampled)
            }
            
            print(f"  📝 Transcription: '{transcription}'")
            print(f"  📊 SNR: {metrics['snr_db']:.2f} dB")
            print(f"  📊 MSE: {metrics['mse']:.6f}")
            print(f"  📊 High freq ratio: {metrics['high_freq_ratio']:.4f}")
        
        # Compare results
        print(f"\n📋 SUMMARY COMPARISON")
        print("=" * 50)
        
        if reference_text:
            print(f"🎯 Reference: '{reference_text}'")
            print()
        
        for method_name, result in results.items():
            transcription = result['transcription']
            metrics = result['audio_metrics']
            
            # Calculate WER if reference provided
            wer = "N/A"
            if reference_text:
                wer = self.calculate_wer(reference_text, transcription)
            
            print(f"{method_name:20} | WER: {wer:6} | SNR: {metrics['snr_db']:6.2f}dB | Transcription: '{transcription}'")
        
        return results
    
    def calculate_wer(self, reference, hypothesis):
        """Calculate Word Error Rate"""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Simple WER calculation
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
        
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
            
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitution = d[i-1][j-1] + 1
                    insertion = d[i][j-1] + 1
                    deletion = d[i-1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)
        
        return f"{d[len(ref_words)][len(hyp_words)] / len(ref_words):.3f}"

# Usage example
if __name__ == "__main__":
    # Initialize tester
    tester = ResampleTester("models/phowhisper_multistage")
    
    # Test with your audio file
    audio_file = "data/audio_chunks/speech_20250620_083741_001.wav" #48k
    reference_text = "bật đèn phòng khách"  # What was actually said
    
    # Run comparison test
    results = tester.test_resample_methods(audio_file, reference_text)