import os
import skvideo.io
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import multiprocessing
import json
import numpy as np
import cv2
import skvideo.io
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

np.int = int 
np.float = float

class VideoProcessor:
    def __init__(self, path):
        self.video_path = path
        self.mouth_frames = []
        self.face_frames = []
        self.min_detected_frames = 10  # mínimo para considerar válido

        base_options = python.BaseOptions(model_asset_path='models/face_landmarker_v2_with_blendshapes.task')
        self.detector = vision.FaceLandmarker.create_from_options(

            vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,  
                output_facial_transformation_matrixes=True,
                num_faces=1
            )
        )

    def getFileName(self):
        """Returns the filename of the video."""
        return os.path.basename(self.video_path)
    
    def temporal_resample(self, frames, target_frames=256):
        """Interpola la secuencia de frames a target_frames"""
        num_frames = len(frames)
        indices = np.linspace(0, num_frames - 1, target_frames).astype(np.float32)
        resampled_frames = []

        for i in indices:
            lower = int(np.floor(i))
            upper = min(lower + 1, num_frames - 1)
            alpha = i - lower
            frame = (1 - alpha) * frames[lower] + alpha * frames[upper]
            resampled_frames.append(frame)

        return np.array(resampled_frames)

    def normalize_frames(self, frames):
        """Normaliza cada frame (z-score)."""
        norm_frames = []
        for frame in frames:
            mean = np.mean(frame)
            std = np.std(frame) if np.std(frame) > 0 else 1.0
            norm_frame = (frame - mean) / std
            norm_frames.append(norm_frame)
        return np.array(norm_frames)

    def process_video(self):
        self.mouth_frames = []
        self.face_frames = []

        videogen = skvideo.io.vreader(self.video_path)

        for frame in videogen:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            detection_result = self.detect_face(frame)
            if detection_result is None:
                continue

            mouth_crop = self.extract_mouth(gray_frame, detection_result)
            face_crop = self.extract_face(gray_frame, detection_result)

            if face_crop is not None:
                self.face_frames.append(face_crop)
            if mouth_crop is not None:
                self.mouth_frames.append(mouth_crop)

        if len(self.mouth_frames) < self.min_detected_frames or len(self.face_frames) < self.min_detected_frames:
            print(f"[{self.video_path}] Descartado por pocos frames detectados (mouth: {len(self.mouth_frames)}, face: {len(self.face_frames)})")
            return np.array([]), np.array([])

        # Normalización
        self.mouth_frames = self.normalize_frames(self.mouth_frames)
        self.face_frames = self.normalize_frames(self.face_frames)

        # Interpolación directa a 256
        self.mouth_frames = self.temporal_resample(self.mouth_frames, target_frames=256)
        self.face_frames = self.temporal_resample(self.face_frames, target_frames=256)

        return self.mouth_frames, self.face_frames

    def detect_face(self, frame):
        """Detects face landmarks in a grayscale frame using MediaPipe."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.detector.detect(mp_image)
        return detection_result if detection_result.face_landmarks else None

    def extract_face(self, image, detection_result, final_size=(100, 100)):
        """Extracts the face region and resizes it to a fixed size."""
        for face_landmarks in detection_result.face_landmarks:
            img_height, img_width = image.shape

            x_min = int(min(landmark.x * img_width for landmark in face_landmarks))
            x_max = int(max(landmark.x * img_width for landmark in face_landmarks))
            y_min = int(min(landmark.y * img_height for landmark in face_landmarks))
            y_max = int(max(landmark.y * img_height for landmark in face_landmarks))

            if x_max <= x_min or y_max <= y_min:
                return None

            face_region = image[y_min:y_max, x_min:x_max]
            resized_face = cv2.resize(face_region, final_size, interpolation=cv2.INTER_AREA)

            return resized_face  

    def extract_mouth(self, image, detection_result, target_size=(100, 50), margin_factor=0.2):
        """Extracts the mouth region with additional margin while maintaining a 2:1 aspect ratio."""
        MOUTH_LANDMARKS = [0, 37, 267, 39, 40, 41, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

        for face_landmarks in detection_result.face_landmarks:
            img_height, img_width = image.shape
            mouth_coords = [(int(landmark.x * img_width), int(landmark.y * img_height)) 
                            for i, landmark in enumerate(face_landmarks) if i in MOUTH_LANDMARKS]

            if not mouth_coords:
                return None  

            x_min, x_max = min(x for x, _ in mouth_coords), max(x for x, _ in mouth_coords)
            y_min, y_max = min(y for _, y in mouth_coords), max(y for _, y in mouth_coords)

            mouth_width, mouth_height = x_max - x_min, y_max - y_min
            margin_w = int(mouth_width * margin_factor)
            margin_h = int(mouth_height * margin_factor)

            x_min = max(x_min - margin_w, 0)
            x_max = min(x_max + margin_w, img_width)
            y_min = max(y_min - margin_h, 0)
            y_max = min(y_max + margin_h, img_height)

            mouth_width, mouth_height = x_max - x_min, y_max - y_min

            desired_width = max(mouth_width, 2 * mouth_height)
            desired_height = desired_width // 2  

            center_x, center_y = (x_max + x_min) // 2, (y_max + y_min) // 2

            x_min_new = max(center_x - desired_width // 2, 0)
            x_max_new = min(center_x + desired_width // 2, img_width)
            y_min_new = max(center_y - desired_height // 2, 0)
            y_max_new = min(center_y + desired_height // 2, img_height)

            mouth_region = image[y_min_new:y_max_new, x_min_new:x_max_new]
            resized_mouth = cv2.resize(mouth_region, target_size, interpolation=cv2.INTER_AREA)

            return resized_mouth

    def saveFaceFramesNumpy(self, output_file="face_frames.npz"):
        """Guarda los frames de la cara con padding al final si es necesario."""
        print(f"Saved face frames as {output_file}, Shape: {self.face_frames.shape}")
        np.savez_compressed(output_file, face_frames=self.face_frames)

    def saveMouthFramesNumpy(self, output_file="mouth_frames.npz"):
        """Guarda los frames de la boca con padding al final si es necesario."""
        np.savez_compressed(output_file, mouth_frames=self.mouth_frames)
        

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

json_file = "./Data/videos_metadata.json"
output_dirMouth = "./DataProcessed/Mouth"
output_dirFace = "./DataProcessed/Face"

list_of_videos = get_filtered_subclip_paths(json_file)
num_processes = min(multiprocessing.cpu_count(), len(list_of_videos))  

# Usamos un Manager para manejar el Lock de manera segura
manager = multiprocessing.Manager()
lock = manager.Lock()

def process_video_parallel(video_path):
    """Procesa un video y guarda los frames de bocas y caras en archivos npz separados."""
    video_processor = VideoProcessor(video_path)
    processed_mouths, processed_faces = video_processor.process_video()

    if len(processed_mouths) > 0 and len(processed_faces) > 0:
        # Crear carpeta de salida por video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(output_dirMouth, exist_ok=True)
        os.makedirs(output_dirFace, exist_ok=True)

        # Rutas de los archivos npz
        mouth_npy = os.path.join(output_dirMouth, f"{video_name}_mouth.npz")
        face_npy = os.path.join(output_dirFace, f"{video_name}_face.npz")

        # Guardar los frames de bocas y caras en archivos npz 
        video_processor.saveFaceFramesNumpy(face_npy)
        video_processor.saveMouthFramesNumpy(mouth_npy)

        # Obtener el FPS del video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()  # Liberar el video después de obtener el FPS

        # Bloquear escritura para evitar corrupción de datos en JSON
        with lock:
            print(f"[{video_path}] Mouth frames: {len(processed_mouths)}, Face frames: {len(processed_faces)}, FPS: {fps}")

            # Leer el JSON existente
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    metadata = json.load(f)
            else:
                print("El archivo JSON no existe.")
                return  # Si no hay JSON, no se puede actualizar

            # Buscar el video correcto en la estructura del JSON
            updated = False
            for video_entry in metadata["videos"]:
                subclips = video_entry["video_metadata"].get("subclips", {})
                if video_path in subclips:
                    subclips[video_path]["mouth_numpy"] = mouth_npy
                    subclips[video_path]["face_numpy"] = face_npy
                    subclips[video_path]["fps"] = fps  # Agregar el fps al subclip
                    updated = True
                    break  # Salimos del loop, ya que encontramos y actualizamos el clip

            if not updated:
                print(f"Advertencia: No se encontró {video_path} en el JSON, no se actualizó.")

            # Escribir de vuelta el JSON con la actualización
            with open(json_file, "w") as f:
                json.dump(metadata, f, indent=4)

    else:
        print(f"[{video_path}] No faces or mouths detected.")


if __name__ == "__main__":
    with multiprocessing.Pool(num_processes) as pool:
        pool.map(process_video_parallel, list_of_videos)

