import torch
from dsp import DSP
from audio_interface import AudioInterface
from logger import logger
import librosa
import soundfile as sf
import numpy as np
import threading

class Morse(AudioInterface):
    MORSE_CODE = {
        # letters
        'A': '.-',     'B': '-...',   'C': '-.-.',  'D': '-..',
        'E': '.',      'F': '..-.',   'G': '--.',   'H': '....',
        'I': '..',     'J': '.---',   'K': '-.-',   'L': '.-..',
        'M': '--',     'N': '-.',     'O': '---',   'P': '.--.',
        'Q': '--.-',   'R': '.-.',    'S': '...',   'T': '-',
        'U': '..-',    'V': '...-',   'W': '.--',   'X': '-..-',
        'Y': '-.--',   'Z': '--..',
    
        # numbers
        '0': '-----',  '1': '.----',  '2': '..---', '3': '...--',
        '4': '....-',  '5': '.....',  '6': '-....', '7': '--...',
        '8': '---..',  '9': '----.',
    
        # punctuation
        '.': '.-.-.-', ',': '--..--', '?': '..--..',  "'": '.----.',
        '!': '-.-.--', '/': '-..-.',  '(': '-.--.',   ')': '-.--.-',
        '&': '.-...',  ':': '---...', ';': '-.-.-.',  '=': '-...-',
        '+': '.-.-.',  '-': '-....-', '_': '..--.-',  '"': '.-..-.',
        '$': '...-..-', '@': '.--.-.',
    
        # special whitespace characters
        ' ': '/',               # space between words
        '\n': '.-.-',           # "Newline" (commonly AR prosign for new message line)
        '\r': '.-.-',           # carriage return as AR prosign for robustness
    }

    REVERSE_MORSE_CODE = {v: k for k, v in MORSE_CODE.items()}

    def __init__(self, wpm=20, sr=44100, freq=600):
        self.unit = 1.2 / wpm
        self.dsp = DSP(sr=sr, freq=freq)
        self.sr = sr
        self._lock = threading.Lock()
        logger.info(f"Morse system initialized (WPM={wpm})")

    def text_to_morse(self, text):
        morse = ' '.join(self.MORSE_CODE.get(c.upper(), '') for c in text if c.upper() in self.MORSE_CODE)
        logger.debug(f"Converted text '{text}' to Morse '{morse}'")
        return morse

    def morse_to_text(self, morse):
        text = ''.join(self.REVERSE_MORSE_CODE.get(m, '') for m in morse.split())
        logger.debug(f"Converted Morse '{morse}' to text '{text}'")
        return text

    def text_to_audio(self, text, padding_seconds=0.0):
        morse = self.text_to_morse(text)
        audio = torch.tensor([], device=self.dsp.device)
        symbol_silence = torch.zeros(int(self.sr * self.unit), device=self.dsp.device)
    
        for symbol in morse:
            if symbol == '.':
                audio = torch.cat([audio, self.dsp.synthesize_tone(self.unit), symbol_silence])
            elif symbol == '-':
                audio = torch.cat([audio, self.dsp.synthesize_tone(3*self.unit), symbol_silence])
            elif symbol == ' ':
                audio = torch.cat([audio, torch.zeros(int(self.sr * 2 * self.unit), device=self.dsp.device)])
            elif symbol == '/':
                audio = torch.cat([audio, torch.zeros(int(self.sr * 6 * self.unit), device=self.dsp.device)])
    
        # add configurable silence padding at the beginning and end
        padding = torch.zeros(int(self.sr * padding_seconds), device=self.dsp.device)
        audio_padded = torch.cat([padding, audio, padding])
    
        logger.info(f"Generated audio from text '{text}' with {padding_seconds}s silence padding")
        return audio_padded

    def audio_to_text(self, audio):
        with self._lock:
            filtered = self.dsp.bandpass(audio)
            envelope = self.dsp.envelope(filtered)
            threshold = self.dsp.adaptive_threshold(envelope)
            binary = (envelope > threshold).cpu().numpy().astype(int)
            transitions = np.diff(binary, prepend=0)
            indexes = np.where(transitions)[0]
            durations = np.diff(np.append(indexes, len(binary))) / self.sr

            dot = self.unit
            dash = 3 * self.unit
            symbols = ''
            for dur, state in zip(durations, binary[indexes]):
                if state:
                    symbols += '.' if dur < (dot + dash)/2 else '-'
                else:
                    if dur > 6 * dot:
                        symbols += ' / '
                    elif dur > 2 * dot:
                        symbols += ' '

            text = self.morse_to_text(symbols)
            logger.info(f"Decoded text from audio: '{text}'")
            return text.strip()

    def load_audio(self, filepath):
        audio, _ = librosa.load(filepath, sr=self.sr, mono=True)
        logger.info(f"Loaded audio from '{filepath}'")
        return torch.tensor(audio, device=self.dsp.device)

    def save_audio(self, audio, filepath):
        sf.write(filepath, audio.cpu().numpy(), self.sr)
        logger.info(f"Saved audio to '{filepath}'")

