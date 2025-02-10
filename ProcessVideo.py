import os
import skvideo.io
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import multiprocessing
import json

class VideoProcessor:
    def __init__(self, path):
        self.video_path = path
        self.mouth_frames = []
        self.face_frames = []

        base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
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

    def process_video(self):
        """Extracts mouth and face regions from video frames after converting to grayscale."""
        self.mouth_frames = []
        self.face_frames = []
        
        videogen = skvideo.io.vreader(self.video_path)

        for frame in videogen:
            # Convert frame to grayscale first
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Detect face landmarks
            detection_result = self.detect_face(frame)
            if detection_result is None:
                print('No face detected')
                return [], []

            # Extract mouth and face regions
            mouth_crop = self.extract_mouth(gray_frame, detection_result)
            face_crop = self.extract_face(gray_frame, detection_result)

            # Store extracted regions
            if face_crop is not None:
                self.face_frames.append(face_crop)
            if mouth_crop is not None:
                self.mouth_frames.append(mouth_crop)

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

            # Get bounding box coordinates
            x_min = int(min(landmark.x * img_width for landmark in face_landmarks))
            x_max = int(max(landmark.x * img_width for landmark in face_landmarks))
            y_min = int(min(landmark.y * img_height for landmark in face_landmarks))
            y_max = int(max(landmark.y * img_height for landmark in face_landmarks))

            # Ensure the region is valid
            if x_max <= x_min or y_max <= y_min:
                return None

            # Crop and resize face region
            face_region = image[y_min:y_max, x_min:x_max]
            resized_face = cv2.resize(face_region, final_size, interpolation=cv2.INTER_AREA)

            return resized_face  

    def extract_mouth(self, image, detection_result, target_size=(100, 50), margin_factor=0.2):
        """Extrae la región de la boca con margen adicional, manteniendo una proporción 2:1."""
        MOUTH_LANDMARKS = [0, 37, 267, 39, 40, 41, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

        for face_landmarks in detection_result.face_landmarks:
            img_height, img_width = image.shape
            mouth_coords = [(int(landmark.x * img_width), int(landmark.y * img_height)) 
                            for i, landmark in enumerate(face_landmarks) if i in MOUTH_LANDMARKS]

            if not mouth_coords:
                return None  

            x_min, x_max = min(x for x, _ in mouth_coords), max(x for x, _ in mouth_coords)
            y_min, y_max = min(y for _, y in mouth_coords), max(y for _, y in mouth_coords)

            # Calcular ancho y alto originales
            mouth_width, mouth_height = x_max - x_min, y_max - y_min

            # Aplicar margen
            margin_w = int(mouth_width * margin_factor)
            margin_h = int(mouth_height * margin_factor)

            x_min = max(x_min - margin_w, 0)
            x_max = min(x_max + margin_w, img_width)
            y_min = max(y_min - margin_h, 0)
            y_max = min(y_max + margin_h, img_height)

            # Recalcular tamaño con margen
            mouth_width, mouth_height = x_max - x_min, y_max - y_min

            # Ajustar para mantener proporción 2:1 (ancho = 2 * alto)
            desired_width = max(mouth_width, 2 * mouth_height)
            desired_height = desired_width // 2  

            center_x, center_y = (x_max + x_min) // 2, (y_max + y_min) // 2

            # Definir nuevas coordenadas asegurando que se mantengan dentro de la imagen
            x_min_new = max(center_x - desired_width // 2, 0)
            x_max_new = min(center_x + desired_width // 2, img_width)
            y_min_new = max(center_y - desired_height // 2, 0)
            y_max_new = min(center_y + desired_height // 2, img_height)

            # Extraer y redimensionar la región de la boca
            mouth_region = image[y_min_new:y_max_new, x_min_new:x_max_new]
            resized_mouth = cv2.resize(mouth_region, target_size, interpolation=cv2.INTER_AREA)

            return resized_mouth

    def save_video(self, frames, output_path, fps=30):
        """Saves extracted frames as a grayscale video."""
        if not frames:
            print(f"No frames available to save for {output_path}.")
            return

        frame_height, frame_width = frames[0].shape  
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)

        for frame in frames:
            video_writer.write(frame)

        video_writer.release()
        print(f"Saved video as {output_path}")

    def save_mouth_video(self, output_path="output_mouths.mp4", fps=30):
        """Saves the extracted mouth frames as a video."""
        self.save_video(self.mouth_frames, output_path, fps)

    def save_face_video(self, output_path="output_faces.mp4", fps=30):
        """Saves the extracted face frames as a video."""
        self.save_video(self.face_frames, output_path, fps)

    def saveFaceFramesNumpy(self, output_file="face_frames.npz"):
        """Saves the extracted face frames as a compressed NumPy file for neural network input."""
        if not self.face_frames:
            print("No face frames available to save.")
            return
        
        np.savez_compressed(output_file, face_frames=np.array(self.face_frames))
        print(f"Saved face frames as {output_file}")

    def saveMouthFramesNumpy(self, output_file="mouth_frames.npz"):
        """Saves the extracted mouth frames as a compressed NumPy file for neural network input."""
        if not self.mouth_frames:
            print("No mouth frames available to save.")
            return
        
        np.savez_compressed(output_file, mouth_frames=np.array(self.mouth_frames))
        print(f"Saved mouth frames as {output_file}")

def get_subclip_paths(file_path, start, end):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    subclip_paths = [
        clip_path
        for video in data["videos"]
        for clip_path in video["video_metadata"]["subclips"].keys()
    ]
    return subclip_paths[start:end]

json_file = "./Data/videos_metadata.json"
output_dirMouth = "./Data/processed/Mouth"
output_dirFace = "./Data/processed/Face"

list_of_videos = get_subclip_paths(json_file, 0, 500)
num_processes = min(multiprocessing.cpu_count(), len(list_of_videos))  

# Usamos un Manager para manejar el Lock de manera segura
manager = multiprocessing.Manager()
lock = manager.Lock()

def process_video_parallel(video_path):
    """Procesa un video y guarda los frames de bocas y caras en archivos npz separados."""
    video_processor = VideoProcessor(video_path)
    processed_mouths, processed_faces = video_processor.process_video()

    if processed_mouths and processed_faces:
        # Crear carpeta de salida por video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(output_dirMouth, exist_ok=True)
        os.makedirs(output_dirFace, exist_ok=True)

        # Rutas de los archivos npz
        mouth_npy = os.path.join(output_dirMouth, f"{video_name}_mouth.npz")
        face_npy = os.path.join(output_dirFace, f"{video_name}_face.npz")

        # Guardar los datos en archivos separados
        np.savez_compressed(mouth_npy, mouth_frames=np.array(processed_mouths))
        np.savez_compressed(face_npy, face_frames=np.array(processed_faces))

        # Bloquear escritura para evitar corrupción de datos en JSON
        with lock:
            print(f"[{video_path}] Mouth frames: {len(processed_mouths)}, Face frames: {len(processed_faces)}")

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

