# Morse-Audio

A high-performance Morse code encoder and decoder leveraging advanced digital signal processing (DSP) and GPU-accelerated audio synthesis. Built using PyTorch, this project offers precision and speed suitable for both educational use and research-grade signal analysis.

## Features

- **Morse Code Translation**: Encode plain text to Morse audio and decode Morse audio back to text accurately.
- **Robust DSP Pipeline**: Implements adaptive thresholding, Hilbert envelope extraction, and Butterworth bandpass filtering.
- **GPU Acceleration**: Utilizes CUDA via PyTorch for efficient processing.
- **Thread-Safe and Concurrent**: Thread-safe design allows simultaneous decoding and encoding operations.
- **Configurable WPM and Frequency**: Adjust Morse code speed and audio tone frequency easily.

## Installation

### Prerequisites

- Python 3.10+
- PyTorch with CUDA support
- Librosa and SoundFile for audio processing
- NumPy, SciPy

### Install dependencies

```bash
pip install torch librosa soundfile numpy scipy
```

### Clone the repository

```bash
git clone https://github.com/joe-crowley/morse-audio.git
cd morse-audio
```

## Usage

Run the example provided in `main.py`:

```bash
python main.py
```

This will:

- Encode a sample message to Morse audio and save it (`hello_world.wav`).
- Load the audio from disk and decode it back into text.
- Log the original and decoded messages.

### Customizing Morse Generation

Adjust parameters directly in the `main.py` or instantiate the `Morse` class as follows:

```python
from morse import Morse

morse_system = Morse(wpm=25, sr=48000, freq=750)
audio_tensor = morse_system.text_to_audio("Custom Message")
morse_system.save_audio(audio_tensor, "custom_message.wav")
```

## Code Structure

- **`morse.py`**: Core logic for Morse code translation and audio synthesis.
- **`dsp.py`**: DSP functions including bandpass filtering, envelope detection, and adaptive thresholding.
- **`audio_interface.py`**: Abstract interface defining standard audio operations.
- **`logger.py`**: Thread-safe logging to both console and file with configurable verbosity.
- **`main.py`**: Example demonstrating end-to-end encoding and decoding.

## DSP Methodology

The signal processing approach involves:

- Applying a Butterworth bandpass filter around a configurable Morse tone frequency.
- Using Hilbert transform for envelope extraction.
- Calculating adaptive thresholds to robustly separate Morse signals from background noise.

## Logging

Logs are stored in `logs/morse_system.log` and output to the console, capturing key processing steps and facilitating debugging.

## Contributions

Contributions and suggestions are welcomed. Please open an issue or submit a pull request on GitHub.

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

© 2025 Joe Crowley | Fat Tailed Solutions
