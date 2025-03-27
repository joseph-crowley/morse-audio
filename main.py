from morse import Morse
from logger import logger

def main():
    morse_system = Morse(wpm=20)

    message = "Hello world - Fat Tailed Solutions."
    audio = morse_system.text_to_audio(message)
    morse_system.save_audio(audio, "hello_world.wav")

    loaded_audio = morse_system.load_audio("hello_world.wav")
    decoded_message = morse_system.audio_to_text(loaded_audio)

    logger.info(f"Original Message: '{message}' | Decoded Message: '{decoded_message}'")

if __name__ == "__main__":
    main()

