import os
import subprocess

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

            print(f"Clip guardado: {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar FFmpeg en la línea {index + 1}: {e}")
        except Exception as e:
            print(f"Error procesando la línea {index + 1}: {e}")

    return results


def cutNLabel(filetext, path_video, output_path, id): 
    recortar_video(filetext, path_video, output_path, id)
