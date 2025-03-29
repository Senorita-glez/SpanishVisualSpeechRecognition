import numpy as np

# Ruta a tu archivo .npz
ruta_archivo = "DataProcessed/Mouth/clip_4-1_mouth.npz"

# Cargar archivo
data = np.load(ruta_archivo)

# Mostrar claves disponibles
print("Claves:", data.files)

# Mostrar shape de cada arreglo
print("frames shape:", data['mouth_frames'].shape)
