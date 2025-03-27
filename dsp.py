import torch
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, hilbert
from logger import logger

class DSP:
    def __init__(self, sr=44100, freq=600, bandwidth=200, device='cuda'):
        self.sr = sr
        self.freq = freq
        self.bandwidth = bandwidth
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"DSP initialized (SR={self.sr}, Freq={self.freq}Hz, Device={self.device})")

    def bandpass(self, signal):
        nyq = 0.5 * self.sr
        low = (self.freq - self.bandwidth / 2) / nyq
        high = (self.freq + self.bandwidth / 2) / nyq
        sos = butter(6, [low, high], btype='band', output='sos')
        filtered = sosfiltfilt(sos, signal.cpu().numpy())
        logger.debug("Applied robust bandpass filter")
        return torch.from_numpy(filtered.copy()).to(self.device)

    def envelope(self, signal):
        analytic = hilbert(signal.cpu().numpy())
        envelope = np.abs(analytic)
        envelope = envelope / np.max(envelope)
        logger.debug("Calculated analytic envelope")
        return torch.from_numpy(envelope).to(self.device)

    def adaptive_threshold(self, envelope):
        median = np.median(envelope.cpu().numpy())
        mad = np.median(np.abs(envelope.cpu().numpy() - median))
        threshold = median + 3 * mad
        logger.debug(f"Adaptive threshold calculated: {threshold:.4f}")
        return threshold

    def synthesize_tone(self, duration, amplitude=0.7):
        t = torch.linspace(0, duration, int(self.sr * duration), device=self.device)
        omega = 2 * np.pi * self.freq
        tone = amplitude * torch.sin(omega * t)
        logger.debug(f"Synthesized tone: duration={duration:.4f}s")
        return tone

