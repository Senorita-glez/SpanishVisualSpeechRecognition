import os
import whisper
import ffmpeg

# Optionally, ensure PyTorch warning mitigation
import torch
original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = True  # Ensures weights-only loading when possible
    return original_torch_load(*args, **kwargs)

torch.load = safe_torch_load  # Override torch.load globally

def cleanpathvideo(path_video):
    # Extract the filename from the full path
    return os.path.basename(path_video)

def extract_audio(input_file, output_audio):
    stream = ffmpeg.input(input_file)
    stream = ffmpeg.output(stream, output_audio)
    ffmpeg.run(stream, overwrite_output=True)
    return output_audio

def transcribe_with_timestamps(file_path):
    # Load the Whisper model
    model = whisper.load_model("base")  # Use "large" for higher accuracy if resources allow

    # Transcribe the audio with timestamps
    print("Transcribing audio with timestamps...")
    result = model.transcribe(file_path, language="es", task="transcribe")

    # Extract segments with timestamps
    segments = result["segments"]
    return segments

def save_as_txt_with_seconds(segments, output_file):
    print(f"Saving timestamps and text to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as txt_file:
        for segment in segments:
            # Extract start time, end time, and text
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            # Write times and text to the file
            txt_file.write(f"{start_time:.3f},{end_time:.3f} -> {text}\n")


def get_subtitles(path_video, number):
    filename = cleanpathvideo(path_video)
    output_audio = f"../Data/video{number}/audio{number}.wav"
    output_txt = f"../Data/video{number}/timestamps{number}.txt"

    try:
        # Extract audio from video
        extract_audio(path_video, output_audio)
        # Transcribe audio and get timestamps
        segments = transcribe_with_timestamps(output_audio)
        # Save timestamps to a text file
        save_as_txt_with_seconds(segments, output_txt)
        print(f"Timestamps saved to {output_txt}")
        return output_txt
    except Exception as e:
        print("An error occurred:", e)
