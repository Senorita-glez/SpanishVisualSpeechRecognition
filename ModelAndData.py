import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class LipToMelDataset(Dataset):
    def __init__(self, mouth_folder, mel_folder, frame_len=5, mel_len=6, stride=1):
        self.samples = []  # Guardamos solo índices, no datos
        self.frame_len = frame_len
        self.mel_len = mel_len

        mouth_files = [f for f in os.listdir(mouth_folder) if f.endswith(".npz")]

        for clip_file in mouth_files:
            audio_file = clip_file.replace("clip", "audio").replace("mouth", "mel")
            mouth_path = os.path.join(mouth_folder, clip_file)
            mel_path = os.path.join(mel_folder, audio_file)

            if not os.path.exists(mel_path):
                continue

            # Cargamos solo los headers
            frames_npz = np.load(mouth_path)
            mel_npz = np.load(mel_path)

            frames_len = frames_npz['mouth_frames'].shape[0]
            mel_len_actual = mel_npz['mel_spec'].shape[1]

            # Ajustar para trabajar con los datos reales
            min_len = min(frames_len, mel_len_actual)

            # Guardamos solo las referencias, no los datos
            num_samples = min_len - max(frame_len, mel_len) + 1

            for i in range(0, num_samples, stride):
                self.samples.append({
                    'mouth_path': mouth_path,
                    'mel_path': mel_path,
                    'start_idx': i
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        npz_frames = np.load(sample['mouth_path'])
        npz_mel = np.load(sample['mel_path'])

        frames = npz_frames['mouth_frames']  # (T,112,112)
        mel = npz_mel['mel_spec']                 # (80,T)

        i = sample['start_idx']

        f = frames[i:i+self.frame_len]
        m = mel[:, i:i+self.mel_len]

        # Preparar tensor
        f = torch.tensor(f, dtype=torch.float32).unsqueeze(0) / 255.0   # (1,5,112,112)
        m = torch.tensor(m, dtype=torch.float32).unsqueeze(0)           # (1,80,6)
        
        return f, m


# Definir las carpetas donde están los npz
mouth_folder = "DataProcessed/Mouth"
mel_folder = "DataProcessed/Mel"

# Crear dataset
dataset = LipToMelDataset(
    mouth_folder=mouth_folder,
    mel_folder=mel_folder,
    frame_len=5,
    mel_len=6,
    stride=1  # o lo que necesites
)

# Crear dataloader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for frames, mels in dataloader:
    print("Frames shape:", frames.shape)  # Esperado: [16,1,5,112,112]
    print("Mels shape:", mels.shape)      # Esperado: [16,1,80,6]
    break


