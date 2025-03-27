import torch
from abc import ABC, abstractmethod

class AudioInterface(ABC):

    @abstractmethod
    def text_to_audio(self, text: str) -> torch.Tensor:
        pass

    @abstractmethod
    def audio_to_text(self, audio: torch.Tensor) -> str:
        pass

    @abstractmethod
    def load_audio(self, filepath: str) -> torch.Tensor:
        pass

    @abstractmethod
    def save_audio(self, audio: torch.Tensor, filepath: str):
        pass

