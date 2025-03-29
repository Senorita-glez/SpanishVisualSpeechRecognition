import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# Dataset con normalización incluida
# ---------------------------

class LipToMelDataset(Dataset):
    def __init__(self, mouth_folder, mel_folder, frame_len=5, mel_len=6, stride=1):
        self.samples = []  
        self.frame_len = frame_len
        self.mel_len = mel_len

        mouth_files = [f for f in os.listdir(mouth_folder) if f.endswith(".npz")]

        for clip_file in mouth_files:
            audio_file = clip_file.replace("clip", "audio").replace("mouth", "mel")
            mouth_path = os.path.join(mouth_folder, clip_file)
            mel_path = os.path.join(mel_folder, audio_file)

            if not os.path.exists(mel_path):
                continue

            frames_npz = np.load(mouth_path)
            mel_npz = np.load(mel_path)

            frames_len = frames_npz['mouth_frames'].shape[0]
            mel_len_actual = mel_npz['mel_spec'].shape[1]
            min_len = min(frames_len, mel_len_actual)

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

        frames = npz_frames['mouth_frames'] 
        mel = npz_mel['mel_spec']

        i = sample['start_idx']

        f = frames[i:i+self.frame_len]
        m = mel[:, i:i+self.mel_len]

        m = (m - m.mean()) / (m.std() + 1e-8)

        f = torch.tensor(f, dtype=torch.float32).unsqueeze(0)
        m = torch.tensor(m, dtype=torch.float32).unsqueeze(0)

        return f, m


# ---------------------------
# Model
# ---------------------------

class LipToMel_Transformer(nn.Module):
    def __init__(self, frame_size=(50, 100)):
        super(LipToMel_Transformer, self).__init__()
        self.frame_size = frame_size

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        H, W = frame_size
        self.H_out = H // 4
        self.W_out = W // 4
        cnn_output_dim = 64 * self.H_out * self.W_out

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_output_dim, 128),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.decoder = nn.Sequential(
            nn.Linear(128 * 5, 512),
            nn.ReLU(),
            nn.Linear(512, 480),
            nn.ReLU(),
            nn.Unflatten(1, (1, 80, 6))
        )

    def forward(self, x):
        B, C, T, H, W = x.size()
        assert H == self.frame_size[0] and W == self.frame_size[1]
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)
        features = self.fc(features)
        features = features.view(B, T, -1)
        encoded = self.transformer_encoder(features)
        encoded = encoded.view(B, -1)
        mel = self.decoder(encoded)
        return mel


# ---------------------------
# Entrenamiento
# ---------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = LipToMelDataset(
    mouth_folder="DataProcessed/Mouth",
    mel_folder="DataProcessed/Mel",
    frame_len=5,
    mel_len=6,
    stride=1
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = LipToMel_Transformer(frame_size=(50, 100)).to(device)

# Cambiamos a L1Loss que suele ser más estable para mel
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_loss = float('inf')

for epoch in range(50):
    model.train()
    total_loss = 0.0

    for frames, mels in dataloader:
        frames, mels = frames.to(device), mels.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, mels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/50] - Loss: {avg_loss:.6f}")

    # Guardar el mejor modelo
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "lip_to_mel_best.pt")
        print(f"Nuevo mejor modelo guardado (loss: {best_loss:.4f})")
