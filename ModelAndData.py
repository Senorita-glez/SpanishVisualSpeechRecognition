import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import datetime
import sys
import datetime

# ---------------------------
# Dataset con normalización incluida
# ---------------------------

class LipToMelDataset(Dataset):
    def __init__(self, mouth_folder, mel_folder, frame_len=5, mel_len=6, stride=1, per_clip=False, selected_files=None):
        self.samples = []  
        self.frame_len = frame_len
        self.mel_len = mel_len
        self.stride = stride
        self.mel_folder = mel_folder
        self.per_clip = per_clip
        mouth_files = selected_files if selected_files is not None else [f for f in os.listdir(mouth_folder) if f.endswith(".npz")]
        self.global_std, self.global_mean, self.max, self.min, self.subs = self.compute_global_values()

        mouth_files = [f for f in os.listdir(mouth_folder) if f.endswith(".npz")]

        for clip_file in mouth_files:
            audio_file = clip_file.replace("clip", "audio").replace("mouth", "mel")
            mouth_path = os.path.join(mouth_folder, clip_file)
            mel_path = os.path.join(mel_folder, audio_file)

            if not os.path.exists(mel_path):
                print(f"[SKIP] Falta mel: {mel_path}")
                continue

            try:
                frames_npz = np.load(mouth_path)
                mel_npz = np.load(mel_path)

                frames_len = frames_npz['mouth_frames'].shape[0]
                mel_len_actual = mel_npz['mel_spec'].shape[1]
                min_len = min(frames_len, mel_len_actual)

                if self.per_clip:
                    self.samples.append({
                        'mouth_path': mouth_path,
                        'mel_path': mel_path,
                        'full_clip': True
                    })
                else:
                    num_samples = min_len - max(frame_len, mel_len) + 1
                    if num_samples <= 0:
                        print(f"SKIP Archivo demasiado corto: {clip_file}")
                        continue

                    for i in range(0, num_samples, stride):
                        self.samples.append({
                            'mouth_path': mouth_path,
                            'mel_path': mel_path,
                            'start_idx': i
                        })

            except Exception as e:
                print(f"ERROR Archivo problemático: {clip_file} -- {str(e)}")
                continue

    def compute_global_values(self):
        stats_path = os.path.join("DataProcessed", "global_stats.txt")
        
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    std, mean, max_val, min_val, subs = [float(line.strip()) for line in lines[:5]]
                    return std, mean, max_val, min_val, subs
                else:
                    print("Archivo global_stats.txt está incompleto. Recalculando...")
        
        # Si no existe o está incompleto, calcular y guardar
        mel_values = []
        for file in os.listdir(self.mel_folder):
            if file.endswith('.npz'):
                mel_path = os.path.join(self.mel_folder, file)
                try:
                    mel = np.load(mel_path)['mel_spec']
                    mel_values.append(mel)
                except:
                    continue

        if not mel_values:
            raise ValueError("No se encontraron espectrogramas válidos para calcular mean y std.")

        all_mels = np.concatenate(mel_values, axis=1)
        mean = np.mean(all_mels)
        std = np.std(all_mels)
        max_val = np.max(all_mels)
        min_val = np.min(all_mels)
        subs = max_val - min_val

        os.makedirs("DataProcessed", exist_ok=True)
        with open(stats_path, "w") as f:
            f.write(f"{std}\n{mean}\n{max_val}\n{min_val}\n{subs}\n")

        return std, mean, max_val, min_val, subs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        npz_frames = np.load(sample['mouth_path'])
        npz_mel = np.load(sample['mel_path'])

        frames = npz_frames['mouth_frames']
        mel = npz_mel['mel_spec']

        if sample.get('full_clip', False):
            frames = frames[:mel.shape[1]] 
            mel_norm = (mel - self.min) / self.subs
            frames_norm = frames / 255
            print('max n min ', mel_norm.max(), mel_norm.min())
            print('frames', frames_norm.max(), frames_norm.min())
            frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)  # [1, T, H, W]
            mel_norm = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)  # [1, 80, T]
            return frames, mel_norm
        
        else:
            i = sample['start_idx']
            f = frames[i:i+self.frame_len]
            m = mel[:, i:i+self.mel_len]
            m_norm = (m - self.min) / self.subs
            f_norm = f / 255
            f_norm = torch.tensor(f, dtype=torch.float32).unsqueeze(0)     # [1, 5, H, W]
            m_norm = torch.tensor(m_norm, dtype=torch.float32).unsqueeze(0)  # [1, 80, 6]

            return f_norm, m_norm

# ---------------------------
# Model
# ---------------------------

class LipToMel_Transformer(nn.Module):
    def __init__(self, frame_size=(50, 100)):
        super(LipToMel_Transformer, self).__init__()
        self.frame_size = frame_size

        self.cnn = nn.Sequential(
            #entra 50x100
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            #entra 25x50
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        H, W = frame_size
        self.H_out = H // 4
        self.W_out = W // 4
        cnn_output_dim = 64 * self.H_out * self.W_out

        self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(cnn_output_dim, 240),
        nn.ReLU(),
        nn.Linear(240, 80 * 6),
        nn.Sigmoid(),
        nn.Unflatten(1, (1, 80, 6))
    )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.view(B * T, C, H, W)           # procesa cada frame por separado
        features = self.cnn(x)
        mel = self.fc(features)              # [B*T, 1, 80, 6]
        mel = mel.view(B, T, 1, 80, 6)       # reacomoda por batch
        mel = mel.mean(dim=1)               # promedio sobre los T frames
        return mel                          # salida: [B, 1, 80, 6]


# ---------------------------
# Entrenamiento
# ---------------------------
if __name__ == "__main__":
    CONTINUAR_ENTRENAMIENTO = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_clips = sorted([f for f in os.listdir("DataProcessed/Mouth") if f.endswith(".npz")])
    torch.manual_seed(42)

    # Cálculo de tamaños
    total = len(all_clips)
    n_train = int(0.8 * total)
    n_val = int(0.1 * total)
    n_test = total - n_train - n_val  # Asegura que la suma dé exacto

    # División
    train_clip_files = all_clips[:n_train]
    val_clip_files = all_clips[n_train:n_train + n_val]
    test_clip_files = all_clips[n_train + n_val:]

    print(f"Total de clips: {total}")
    print(f"Clips de entrenamiento: {len(train_clip_files)}")
    print(f"Clips de validación: {len(val_clip_files)}")
    print(f"Clips de prueba: {len(test_clip_files)}")

    # Guardar índices para reutilizarlos en evaluación
    os.makedirs("splits", exist_ok=True)
    if not (os.path.exists("splits/train_clip_files.npy") and os.path.exists("splits/val_clip_files.npy") and os.path.exists("splits/test_clip_files.npy")):
        np.save("splits/train_clip_files.npy", train_clip_files)
        np.save("splits/val_clip_files.npy", val_clip_files)
        np.save("splits/test_clip_files.npy", test_clip_files)
    else:
        print("Índices ya existentes. Se usará la separación previa.")
 

    # Dataset completo (por fragmento para entrenamiento, per_clip=False)
    train_dataset = LipToMelDataset(
        mouth_folder="DataProcessed/Mouth",
        mel_folder="DataProcessed/Mel",
        frame_len=5,
        mel_len=6,
        stride=1,
        per_clip=False, 
        selected_files = train_clip_files
    )

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=os.cpu_count() - 1)

    val_dataset = LipToMelDataset(
    mouth_folder="DataProcessed/Mouth",
    mel_folder="DataProcessed/Mel",
    frame_len=5,
    mel_len=6,
    stride=1,
    per_clip=False,
    selected_files=val_clip_files
    )

    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=os.cpu_count() - 1)
    print("Datos cargados correctamente")

    model = LipToMel_Transformer(frame_size=(50, 100)).to(device)
    model_path = "lip_to_mel_best.pt"
    best_loss_path = "best_loss.txt"
    best_loss = float('inf')

    # ---------------------------
    # Cargar modelo anterior si se desea
    # ---------------------------
    if os.path.exists(model_path) and CONTINUAR_ENTRENAMIENTO:
        print(f"Cargando modelo desde {model_path}")
        model.load_state_dict(torch.load(model_path))
        if os.path.exists(best_loss_path):
            with open(best_loss_path, "r") as f:
                best_loss = float(f.read().strip())
            print(f"best_loss cargado: {best_loss:.6f}")
        else:
            best_loss = 9999
    elif not CONTINUAR_ENTRENAMIENTO:
        print("ENTRENAMIENTO FORZADO DESDE CERO (CONTINUAR_ENTRENAMIENTO=False)")
    else:
        print("No hay modelo previo. Entrenamiento desde cero.")

    # ---------------------------
    # Configuración
    # ---------------------------
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("="*50)
    print(f"🕐 Fecha y hora de inicio: {datetime.datetime.now()}")
    print(f"🖥️  Python ejecutado desde: {sys.executable}")
    print(f"💾 CUDA disponible: {torch.cuda.is_available()}")
    print(f"💻 Dispositivo en uso: {device}")
    print("="*50)

    def evaluar(model, dataloader, device, criterion):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for frames, mels_norm in dataloader:
                frames, mels_norm = frames.to(device), mels_norm.to(device)
                outputs = model(frames)
                loss = criterion(outputs, mels_norm)
                total_loss += loss.item()
        return total_loss / len(dataloader)


    # ---------------------------
    # Entrenamiento
    # ---------------------------
    losses_entrenamiento = []
    losses_validacion = []

    for epoch in range(50):
        model.train()
        total_loss = 0.0

        for frames, mels_norm in train_dataloader:
            frames, mels_norm = frames.to(device), mels_norm.to(device)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, mels_norm)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = evaluar(model, val_dataloader, device, criterion)

        losses_entrenamiento.append(avg_train_loss)
        losses_validacion.append(avg_val_loss)

        print(f"📉 Epoch [{epoch+1}/50] - Loss train: {avg_train_loss:.6f} | val: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            with open(best_loss_path, "w") as f:
                f.write(str(best_loss))
            print(f"✅ Nuevo mejor modelo guardado (val_loss: {best_loss:.4f})")


    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(losses_entrenamiento, marker='o', label="Entrenamiento")
    plt.plot(losses_validacion, marker='x', label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Pérdida promedio (L1)")
    plt.title("Evolución del Loss durante el entrenamiento")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_evolucion.png"))
    plt.close()

