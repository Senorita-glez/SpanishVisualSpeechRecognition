import whisper

def transcribe_audio(file_path, language="Spanish"):
    # Load model directly
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    processor = AutoProcessor.from_pretrained("openai/whisper-large")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large")

    # Transcribe the audio
    print("Transcribing audio...")
    result = model.transcribe(file_path, language="es")

    # Extract the transcription text
    transcription = result['text']
    return transcription

if __name__ == "__main__":
    # Provide the path to your audio file
    audio_file = "audio-NoticiaPeru.wav"  # Replace with your file path
    
    try:
        transcription = transcribe_audio(audio_file)
        print("Transcription:")
        print(transcription)
    except Exception as e:
        print("An error occurred:", e)
