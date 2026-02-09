from google.cloud import speech_v1 as speech
from google.cloud.speech import enums, types
import io

def transcribe_audio(file_path):
    client = speech.SpeechClient()
    with io.open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = types.RecognitionAudio(content=content)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)
    
    transcript = []
    for result in response.results:
        transcript.append(result.alternatives[0].transcript)

    return ' '.join(transcript)

# Example usage
file_path = 'path_to_audio_file.wav'
transcription = transcribe_audio(file_path)
print(transcription)
