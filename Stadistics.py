import json

# Cargar el JSON desde un archivo
with open("Data/videos_metadata.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Extraer datos
num_videos = len(data["videos"])
subclip_durations = []
num_subclips = 0

for video in data["videos"]:
    subclips = video["video_metadata"]["subclips"]
    num_subclips += len(subclips)
    subclip_durations.extend([clip["duration"] for clip in subclips.values()])

# Calcular estadísticas
min_duration = min(subclip_durations)
avg_duration = sum(subclip_durations) / len(subclip_durations)
max_duration = max(subclip_durations)

# Mostrar resultados
print(f"Número total de videos principales: {num_videos}")
print(f"Número total de subclips: {num_subclips}")
print(f"Duración mínima de un subclip: {min_duration:.2f} segundos")
print(f"Duración promedio de los subclips: {avg_duration:.2f} segundos")
print(f"Duración máxima de un subclip: {max_duration:.2f} segundos")
