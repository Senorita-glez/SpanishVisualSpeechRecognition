from ModelAndData import LipToMel_Transformer, LipToMelDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

# ----------------------------
# Configuración
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset
dataset = LipToMelDataset(
    mouth_folder="DataProcessed/Mouth",
    mel_folder="DataProcessed/Mel",
    frame_len=5,
    mel_len=6,
    stride=1
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count() - 1)

# ----------------------------
# Modelo
# ----------------------------
model = LipToMel_Transformer(frame_size=(50, 100))
model.load_state_dict(torch.load("modelbest1.pt", map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# Reconstrucción completa
# ----------------------------
def reconstruct_full_mel_clip(frames, model, frame_len=5, mel_len=6, stride=1):
    B, C, T, H, W = frames.shape
    mel_pred = torch.zeros((B, 1, 80, T)).to(device)
    mel_count = torch.zeros((B, 1, 80, T)).to(device)

    with torch.no_grad():
        for i in range(0, T - frame_len + 1, stride):
            frames_window = frames[:, :, i:i+frame_len]
            mel_out = model(frames_window)
            mel_out = mel_out[:, :, :, :min(mel_len, T - i)]
            mel_pred[:, :, :, i:i+mel_out.shape[-1]] += mel_out
            mel_count[:, :, :, i:i+mel_out.shape[-1]] += 1.0

    mel_count[mel_count == 0] = 1.0
    mel_final = mel_pred / mel_count
    return mel_final

# ----------------------------
# Métricas
# ----------------------------
def l1_loss(pred, gt):
    return torch.mean(torch.abs(pred - gt)).item()

def l2_loss(pred, gt):
    return torch.mean((pred - gt) ** 2).item()

def snr(pred, gt):
    signal_power = torch.sum(gt ** 2)
    noise_power = torch.sum((gt - pred) ** 2) + 1e-8
    ratio = signal_power / noise_power
    return 10 * torch.log10(ratio).item()

# ----------------------------
# Evaluación clip por clip
# ----------------------------
total_l1, total_l2, total_snr, n = 0, 0, 0, 0
best_snr, worst_snr = -float('inf'), float('inf')
best_clip, worst_clip = -1, -1
best_mel = None
os.makedirs("outputs", exist_ok=True)

for idx, (frames, mels_norm, clip_mean, clip_std) in enumerate(dataloader):
    frames, mels_norm = frames.to(device), mels_norm.to(device)
    clip_mean, clip_std = clip_mean.to(device), clip_std.to(device)

    mel_pred_norm = reconstruct_full_mel_clip(frames, model)
    mel_pred = mel_pred_norm * clip_std + clip_mean
    mel_gt = mels_norm * clip_std + clip_mean

    T_pred = mel_pred.shape[-1]
    mel_gt = mel_gt[..., :T_pred]

    clip_l1 = l1_loss(mel_pred, mel_gt)
    clip_l2 = l2_loss(mel_pred, mel_gt)
    clip_snr = snr(mel_pred, mel_gt)

    total_l1 += clip_l1
    total_l2 += clip_l2
    total_snr += clip_snr
    n += 1

    print(f"Clip {idx} - L1: {clip_l1:.4f} - L2: {clip_l2:.4f} - SNR: {clip_snr:.2f} dB")

    if clip_snr > best_snr:
        best_snr = clip_snr
        best_clip = idx
        best_mel = mel_pred[0,0].cpu().numpy()

    if clip_snr < worst_snr:
        worst_snr = clip_snr
        worst_clip = idx

    plt.imshow(mel_pred[0,0].cpu().numpy(), aspect='auto', origin='lower')
    plt.title(f"Predicted Mel Clip {idx}")
    plt.colorbar()
    plt.savefig(f'outputs/predicted_mel_clip_{idx}.png')
    plt.close()

# ----------------------------
# Métricas globales
# ----------------------------
print("\nMétricas globales (dataset completo):")
print(f"Avg L1 Loss: {total_l1/n:.6f}")
print(f"Avg L2 Loss: {total_l2/n:.6f}")
print(f"Avg SNR (dB): {total_snr/n:.2f} dB")

print(f"\nMejor clip: {best_clip} con SNR = {best_snr:.2f} dB")
print(f"Peor clip: {worst_clip} con SNR = {worst_snr:.2f} dB")

# ----------------------------
# Guardar imagen del mejor mel
plt.imshow(best_mel, aspect='auto', origin='lower')
plt.title("Best Clip Predicted Mel")
plt.colorbar()
plt.savefig("outputs/best_clip_mel.png")
plt.close()
print("Imagen del mejor mel guardada en outputs/best_clip_mel.png")
