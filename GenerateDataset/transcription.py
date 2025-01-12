import whisper
import ffmpeg

def extract_audio(path, title):
    extracted_audio = f"/{title}/audio-{title}.wav"
    stream = ffmpeg.input(path)
    stream = ffmpeg.output(stream, extracted_audio)
    ffmpeg.run(stream, overwrite_output=True)
    return extracted_audio

def transcribe_with_timestamps(file_path):
    # Load the Whisper model
    model = whisper.load_model("large")  # Use "large" for higher accuracy if resources allow

    # Transcribe the audio with timestamps
    print("Transcribing audio with timestamps...")
    result = model.transcribe(file_path, language="es", task="transcribe")

    # Extract segments with timestamps
    segments = result["segments"]
    return segments

def save_as_srt(segments):
    output_file ="transcription.srt"
    print(f"Saving transcription to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(segments):
            # Convert timestamps to SRT format
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"]

            # Write each subtitle entry
            srt_file.write(f"{i+1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")

def format_timestamp(seconds):
    # Convert seconds to H:M:S,ms format
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

if __name__ == "__main__":
    audio_file = "audio-NoticiaPeru.wav"  # Replace with your WAV file path
    output_srt = "transcriptionPeru.srt"  # Output SRT file name

    try:
        segments = transcribe_with_timestamps(audio_file)
        save_as_srt(segments, output_srt)
        print(f"Transcription saved to {output_srt}")
    except Exception as e:
        print("An error occurred:", e)

def getSubtitles(pathVideo, title):
    extract_audio(title)