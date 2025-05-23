import os
import subprocess
import json
import yt_dlp
import re
import whisper
import ffmpeg
import torch
import time

torch.cuda.empty_cache()
import threading
original_torch_load = torch.load
folder_path = "./Data"

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
    #print("Transcribing audio with timestamps...")
    result = model.transcribe(file_path, language="es", task="transcribe")

    # Extract segments with timestamps
    segments = result["segments"]
    return segments

def transcribe_with_word_timestamps(file_path):
    # Load Whisper model
    model = whisper.load_model("large")  # Large model supports word-level timestamps

    # Transcribe audio with timestamps
    print("Transcribing audio with word timestamps...")
    result = model.transcribe(file_path, language="es", word_timestamps=True)
    
    return result["segments"] 

def save_as_txt_with_seconds(segments, output_file):
    #print(f"Saving timestamps and text to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as txt_file:
        for segment in segments:
            # Extract start time, end time, and text
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            # Write times and text to the file
            txt_file.write(f"{start_time:.3f},{end_time:.3f} -> {text}\n")

def get_subtitles(path_video, number, lock):
    output_audio = f"{folder_path}/video{number}/audio{number}.wav"
    output_txt = f"./Data/video{number}/timestamps{number}.txt"

    with lock:  # Prevent multiple threads from writing at the same time
        try:
            extract_audio(path_video, output_audio)
            segments = transcribe_with_timestamps(output_audio)
            save_as_txt_with_seconds(segments, output_txt)

            if os.path.exists(output_txt):
                return output_txt
            else:
                return None
        except Exception as e:
            print(f"An error occurred in get_subtitles(): {e}")
            return None


def sanitize_filename(filename):
    """Replace special characters in the filename with underscores."""
    sanitized = re.sub(r'[^\w\s]', '_', filename)  # Replace non-alphanumeric characters
    sanitized = sanitized.replace(' ', '_')       # Replace spaces with underscores
    return sanitized

def DownloadVideo(url, platform, save_path):
    if isinstance(url, str):
        result = {"file_path": None, "title": None, "duration": None}
        if platform == 'yt':  # YouTube platform
            try:
                # Extract video information without downloading first to sanitize title
                ydl_opts_info = {
                    'quiet': True,
                    'skip_download': True,  # Only fetch metadata
                }
                with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                    info = ydl.extract_info(url, download=False)
                    original_title = info['title']
                    sanitized_title = sanitize_filename(original_title)

                # Use sanitized title in download options
                ydl_opts = {
                    'outtmpl': f'{save_path}/{sanitized_title}.mp4',  # Sanitized title
                    'format': 'mp4',
                    'quiet': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                sanitized_file_path = f"{save_path}/{sanitized_title}.mp4"
                result["file_path"] = sanitized_file_path
                result["title"] = sanitized_title
                result["duration"] = info.get('duration')
                return result

            except Exception as e:
                print(f"An error occurred with yt-dlp: {e}")
                #print("Falling back to pytubefix...")

                # Fallback to pytubefix
                try:
                    from pytubefix import YouTube
                    from pytubefix.cli import on_progress
                    yt = YouTube(url, on_progress_callback=on_progress)
                    video_stream = yt.streams.get_highest_resolution()
                    original_title = yt.title
                    sanitized_title = sanitize_filename(original_title)
                    
                    file_path = video_stream.download(output_path=save_path)
                    
                    # Rename the file with sanitized title
                    sanitized_file_path = f"{save_path}/{sanitized_title}.mp4"
                    if os.path.exists(file_path):
                        os.rename(file_path, sanitized_file_path)

                    result["file_path"] = sanitized_file_path
                    result["title"] = sanitized_title
                    result["duration"] = yt.length
                    return result
                except Exception as fallback_error:
                    print(f"An error occurred with pytubefix: {fallback_error}")
                    return None
        else:  # TikTok or other platforms
            try:
                # Similar logic as YouTube
                ydl_opts_info = {
                    'quiet': True,
                    'skip_download': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                    info = ydl.extract_info(url, download=False)
                    original_title = info['title']
                    sanitized_title = sanitize_filename(original_title)

                ydl_opts = {
                    'outtmpl': f'{save_path}/{sanitized_title}.mp4',
                    'format': 'mp4',
                    'quiet': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                sanitized_file_path = f"{save_path}/{sanitized_title}.mp4"
                result["file_path"] = sanitized_file_path
                result["title"] = sanitized_title
                result["duration"] = info.get('duration')
                return result
            except Exception as e:
                print(f"An error occurred with TikTok video: {e}")
                return None
    else:
        print("URL is not valid, must be a string.")
        return None

def recortar_video(transcription_file, video_file, save_path, id):
    """
    Recorta un video en segmentos basados en las marcas de tiempo de un archivo de texto y guarda los clips.

    Args:
        transcription_file (str): Ruta al archivo de texto con las marcas de tiempo y descripciones.
        video_file (str): Ruta al archivo de video original.
        save_path (str): Directorio donde se guardarán los clips recortados.
        id (str): Identificador único para los clips.

    Returns:
        dict: Diccionario con los paths de los archivos creados, texto, duración, tiempo de inicio y fin de cada fragmento.
    """

    results = {}

    if transcription_file is None or not os.path.exists(transcription_file):
        raise ValueError(f"Invalid path_subtitles: {transcription_file}")

    # Leer el archivo de transcripción y procesar las líneas
    with open(transcription_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Procesar cada línea para extraer las marcas de tiempo y el texto
    for index, line in enumerate(lines):
        try:
            # Dividir en marcas de tiempo y texto
            time_range, text = line.split("->")
            start_time, end_time = [float(t) for t in time_range.split(",")]

            # Calcular duración del clip
            duration = end_time - start_time

            # Generar un nombre único para cada clip
            output_filename = os.path.join(save_path, f"clip_{id}-{index + 1}.mp4")

            # Ejecutar FFmpeg para recortar el segmento
            subprocess.run([
                "ffmpeg",
                "-i", video_file,
                "-ss", str(start_time),
                "-to", str(end_time),
                "-c:v", "libx264",
                "-c:a", "aac",
                output_filename
            ], check=True)

            # Agregar al diccionario de resultados
            results[output_filename] = {
                "text": text.strip(),
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration
            }

            #(f"Clip guardado: {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar FFmpeg en la línea {index + 1}: {e}")
        except Exception as e:
            print(f"Error procesando la línea {index + 1}: {e}")

    return results

def save_video_data_to_json(new_result, new_subclips, json_file_path, lock):
    """Guarda los metadatos del video y subclips en un archivo JSON."""
    with lock:
        if os.path.exists(json_file_path):
            # Cargar datos existentes
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
        else:
            # Crear estructura inicial si el archivo no existe
            data = {"videos": []}

        # Crear la estructura del nuevo video
        video_data = {
            "video_metadata": {
                "file_path": new_result["file_path"],
                "title": new_result["title"],
                "duration": new_result["duration"],
                "subclips": new_subclips
            }
        }

        # Agregar el nuevo video a la lista de videos
        data["videos"].append(video_data)

        # Guardar los datos actualizados en el archivo JSON
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f"Datos actualizados guardados en {json_file_path}")

import time
import os

def process_video(dupla, platform, json_file_path, lock):
    """Procesa un video, obteniendo subtítulos, subclips y guardando datos."""
    video_id, url = dupla
    savepath = f"./Data/video{video_id}"

    # Crear el directorio si no existe
    os.makedirs(savepath, exist_ok=True)

    # Descargar el video
    try:
        result = DownloadVideo(url, platform, savepath)
        print(f"✅ Download result: {result}")
    except Exception as e:
        print(f"❌ An error occurred while downloading: {e}")
        return

    # Obtener subtítulos
    path_subtitles = get_subtitles(result['file_path'], video_id, lock)

    # 🔹 Ensure the subtitles file exists before continuing
    wait_time = 0
    max_wait = 900  # Wait up to 15 minutes (adjust if needed)
    
    while not path_subtitles or not os.path.exists(path_subtitles):
        if wait_time > max_wait:
            print(f"❌ ERROR: Subtitles file {path_subtitles} did not appear after {max_wait} seconds!")
            return  # Stop processing this video
        print(f"⏳ Waiting for subtitles file {path_subtitles}... ({wait_time}s)")
        time.sleep(5)  # Wait for 5 seconds before checking again
        wait_time += 5
    
    # ✅ Now that we are sure subtitles exist, process the video
    print(f"🎬 Subtitles file ready: {path_subtitles}")

    # Recortar el video usando los subtítulos
    subclips = recortar_video(path_subtitles, result['file_path'], savepath, video_id)
    print(f"✅ Subclips created: {subclips}")

    # Guardar los datos en el archivo JSON
    save_video_data_to_json(result, subclips, json_file_path, lock)
    print(f"📁 Video metadata saved: {json_file_path}")


def divide_links(links, num_threads):
    """Divide la lista de enlaces en sublistas para cada hilo."""
    return [links[i::num_threads] for i in range(num_threads)]


def get_last_video_id(json_file_path):
    """Obtiene el último ID de video existente en el archivo JSON."""
    if os.path.exists(json_file_path):
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            # Obtener todos los IDs de video del archivo JSON
            json_ids = [
                int(os.path.basename(os.path.dirname(video["video_metadata"]["file_path"]))[5:])
                for video in data.get("videos", [])
                if "video_metadata" in video and "file_path" in video["video_metadata"]
            ]
            return max(json_ids) if json_ids else -1
    return -1  # Si el archivo JSON no existe, retornar -1


def fromPlaylistToLinks(playlist_url):
    ydl_opts = {
        'extract_flat': True,  # Extract metadata without downloading
        'quiet': True,         # Suppress output
    }

    links = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract playlist information
        playlist_info = ydl.extract_info(playlist_url, download=False)

        # Check if the playlist contains entries
        if 'entries' in playlist_info:
            # Iterate through each video in the playlist
            for video in playlist_info['entries']:
                link = video['url']  # Get the video URL
                links.append(link)  # Add the URL to the list
        else:
            print("No videos found in the playlist.")

    return links


def create_dataset():
    """Crea un dataset procesando videos de enlaces proporcionados."""
    json_file_path = "./Data/videos_metadata.json"
    folder_path = "./Data"
    '''
    PLxep90LjGgt8ipF2bO5LPWKigA4ckBlyh  # Playlist de videos de prueba
    PLxep90LjGgt9P35CN-bjH2Pd3ny6MV35W  # Playlist de videos de Persona1
    PLxep90LjGgt_LXgT287-5kHkuknkaQWHe  # Playlist de videos de Persona2
    PLxep90LjGgt908HRdOd787bpH5PFeDBqU  # Playlist de videos de Persona3
    PLxep90LjGgt9fLG2ZoGbk7xVOKydpuj4C  # Playlist de videos de Persona4
    PLAAC7fSa2ntKUjtze8ufLhDj8l8MnXhOF  # Playlist de videos de Persona5
    PLxep90LjGgt8_k2OgTTpGFbmW-VAm4ysY  # Playlist de videos de Persona6
    
    PLxep90LjGgt8bqPpkMv4d81zLibPEq_lE  # Playlist de videos de Persona8
    '''
    links = fromPlaylistToLinks("https://www.youtube.com/playlist?list=PLxep90LjGgt8ipF2bO5LPWKigA4ckBlyh")
    # Crear la carpeta si no existe
    os.makedirs(folder_path, exist_ok=True)

    # Obtener el último ID de video existente del archivo JSON
    last_video_id = get_last_video_id(json_file_path)

    # Enumerar los enlaces con IDs continuando desde el último ID + 1
    enumerated_links = [(last_video_id + 1 + i, link) for i, link in enumerate(links)]

    # Determinar el número de hilos
    num_threads = min(len(links), 4)
    lock = threading.Lock()

    # Dividir los enlaces enumerados en sublistas
    sublists = divide_links(enumerated_links, num_threads)
    threads = []

    # Crear hilos para procesar los videos
    for sublist in sublists:
        for dupla in sublist:
            thread = threading.Thread(
                target=process_video,
                args=(dupla, 'yt', json_file_path, lock)
            )
            threads.append(thread)
            thread.start()

    # Esperar a que todos los hilos terminen
    for thread in threads:
        thread.join()

    print('Proceso completado con éxito.')


if __name__ == "__main__":
    create_dataset()
