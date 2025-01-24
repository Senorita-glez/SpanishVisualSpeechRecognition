import os
import json
import threading
from download import DownloadVideo
from transcription import get_subtitles
from cutNmouth import recortar_video


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


def process_video(dupla, platform, json_file_path, lock):
    """Procesa un video, obteniendo subtítulos, subclips y guardando datos."""
    video_id, url = dupla
    savepath = f"../Data/video{video_id}"

    # Crear el directorio si no existe
    os.makedirs(savepath, exist_ok=True)

    # Descargar el video
    try:
        result = DownloadVideo(url, platform, savepath)
        print(f"Download result: {result}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Obtener subtítulos
    path_subtitles = get_subtitles(result['file_path'], video_id)

    # Recortar el video usando los subtítulos
    subclips = recortar_video(path_subtitles, result['file_path'], savepath, video_id)
    print(subclips)

    # Guardar los datos en el archivo JSON
    save_video_data_to_json(result, subclips, json_file_path, lock)


def divide_links(links, num_threads):
    """Divide la lista de enlaces en sublistas para cada hilo."""
    return [links[i::num_threads] for i in range(num_threads)]


def create_dataset():
    """Crea un dataset procesando videos de enlaces proporcionados."""
    json_file_path = "../Data/videos_metadata.json"
    links = [
        'https://youtu.be/rWlY7JiMXHs?si=xwVo0_WPVoZ92hsD',
        'https://youtube.com/shorts/qiZBTNSaNsg?si=iafHTuLrjFs0dq9E',
        'https://youtube.com/shorts/GyYZZxlvW8g?si=xN2ikFKY4FeWKBTt',
        'https://youtube.com/shorts/6BqkgEvbyug?si=JzjHGytwfPrCECKe'
    ]

    # Enumerar los enlaces con IDs
    enumerated_links = [(i, link) for i, link in enumerate(links)]

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
