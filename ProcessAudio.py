import os
import librosa
import numpy as np
import soundfile as sf
import concurrent.futures
import librosa.display
import cv2
import matplotlib.pyplot as plt
import json
import threading
lock = threading.Lock()

def process_video_folder(subdir_path, video_num, fps=30):
    audio_path = os.path.join(subdir_path, f"audio{video_num}.wav")
    text_path = os.path.join(subdir_path, f"timestamps{video_num}.txt")
    
    if not (os.path.exists(audio_path) and os.path.exists(text_path)):
        print(f"Error: Faltan archivos en {subdir_path}, saltando.")
        return
    
    print(f"Procesando: {audio_path} y {text_path}")
    
    # Cargar el audio
    audio, sr = librosa.load(audio_path, sr=None)
    hop_length = int(sr / fps)
    
    # Leer timestamps
    with open(text_path, 'r') as file:
        lines = file.readlines()
    
    # Procesar cada línea del archivo de timestamps
    for i, line in enumerate(lines):
        try:
            time_range, text = line.strip().split('->')
            start_time, end_time = map(float, time_range.split(','))
            
            # Convertir tiempos a muestras
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Cortar el segmento de audio
            audio_segment = audio[start_sample:end_sample]
            
            # Guardar el segmento en la misma carpeta con el nuevo formato
            segment_path = os.path.join(subdir_path, f"audio_{video_num}-{i+1}.wav")
            sf.write(segment_path, audio_segment, sr)
            print(f"Segmento guardado: {segment_path}")
        except ValueError:
            print(f"Error procesando línea en {text_path}: {line}")

def process_audio_segments(base_path="./Data/"):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for subdir in os.listdir(base_path):
            subdir_path = os.path.join(base_path, subdir)
            
            if not os.path.isdir(subdir_path):
                print(f"Error: {subdir_path} no es un directorio.")
                continue
            
            video_num = ''.join(filter(str.isdigit, subdir))  # Extrae solo los números del nombre
            if not video_num:
                print(f"Error: No se encontró número de video en {subdir}, saltando.")
                continue
            
            futures.append(executor.submit(process_video_folder, subdir_path, video_num))
        
        # Esperar a que todos los procesos terminen
        concurrent.futures.wait(futures)

def get_filtered_subclip_paths(file_path, min_duration=1, max_duration=9):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Extraer subclips con sus duraciones
    filtered_subclip_paths = []
    for video in data["videos"]:
        subclips = video["video_metadata"]["subclips"]
        for clip_path, clip_data in subclips.items():
            if min_duration <= clip_data["duration"] <= max_duration:
                filtered_subclip_paths.append(clip_path)

    # Retornar la porción solicitada de la lista
    return filtered_subclip_paths

def replace_clip_with_audio(paths):
    return [path.replace("clip", "audio").replace(".mp4", ".wav") for path in paths]

import threading

lock = threading.Lock()

def generate_mel_spectrogram(audio_path, json_file, target_width=256, n_mels=80, sr=16000, hop_length=512, fmax=8000):
    try:
        if not os.path.exists(audio_path):
            print(f"Error: El archivo de audio {audio_path} no existe.")
            return

        output_dir = "./DataProcessed/Mel/"
        os.makedirs(output_dir, exist_ok=True)
        video_path = audio_path.replace(".wav", ".mp4").replace("audio", "clip")

        y, sr = librosa.load(audio_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, fmax=fmax)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_resized = cv2.resize(mel_spec_db, (target_width, mel_spec_db.shape[0]), interpolation=cv2.INTER_LINEAR)

        output_path = os.path.join(output_dir, os.path.basename(audio_path).replace(".wav", "_mel.npz"))
        np.savez_compressed(output_path, mel_spec=mel_spec_resized)
        print(f"Espectrograma guardado: {output_path}")

        with lock:
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    metadata = json.load(f)
            else:
                print("Error: El archivo JSON no existe.")
                return

            updated = False
            for video_entry in metadata.get("videos", []):
                subclips = video_entry.get("video_metadata", {}).get("subclips", {})
                if video_path in subclips:
                    subclips[video_path]["mel_spectrogram"] = output_path
                    updated = True
                    break

            if not updated:
                print(f"Error: No se encontró {video_path} en el archivo JSON.")
            else:
                with open(json_file, "r+") as f:
                    existing_metadata = json.load(f)
                    f.seek(0)
                    f.truncate()
                    json.dump(existing_metadata, f, indent=4)
                print(f"Actualizado JSON con {output_path}")
    except Exception as e:
        print(f"Error al procesar {video_path}: {e}")
        import traceback
        traceback.print_exc()

def process_mel_spectrograms(audio_paths, json_file):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(generate_mel_spectrogram, path, json_file) for path in audio_paths]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error en un proceso de generación de espectrograma: {e}")


if __name__ == "__main__":
    json_file = "./Data/videos_metadata.json"
    process_audio_segments()
    print("Proceso de segmentación completado con éxito.")
    audio_paths = replace_clip_with_audio(get_filtered_subclip_paths("./Data/videos_metadata.json"))
    print(f"Rutas de los archivos de audio: {audio_paths[0:30]}")
    process_mel_spectrograms(audio_paths, json_file)
    print("Proceso de generación de espectrogramas completado con éxito.")