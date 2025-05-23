import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ModelAndData import LipToMel_Transformer, LipToMelDataset
import csv
from skimage.metrics import structural_similarity as ssim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_clip_files = np.load("splits/test_clip_files.npy")

full_dataset = LipToMelDataset(
    mouth_folder="DataProcessed/Mouth",
    mel_folder="DataProcessed/Mel",
    frame_len=5,
    mel_len=6,
    stride=1,
    per_clip=True,
    selected_files=test_clip_files
)

test_dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count() - 1)

model = LipToMel_Transformer(frame_size=(50, 100))
model.load_state_dict(torch.load("lip_to_mel_best.pt", map_location=device))
model = model.to(device)
model.eval()

# Cargar estadísticas de normalización
stats_path = "DataProcessed/global_stats.txt"
assert os.path.exists(stats_path), "No se encontró el archivo de estadísticas globales."
with open(stats_path, "r") as f:
    lines = f.readlines()
    if len(lines) < 5:
        raise ValueError("Archivo global_stats.txt incompleto.")
    global_std = float(lines[0].strip())
    global_mean = float(lines[1].strip())
    global_max = float(lines[2].strip())
    global_min = float(lines[3].strip())
    global_subs = float(lines[4].strip())

print(f"📊 Global mean: {global_mean:.4f}, std: {global_std:.4f}")

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
    return mel_pred / mel_count

def l1_loss(pred, gt):
    return torch.mean(torch.abs(pred - gt)).item()

def l2_loss(pred, gt):
    return torch.mean((pred - gt) ** 2).item()

def snr(pred, gt):
    signal_power = torch.sum(gt ** 2)
    noise_power = torch.sum((gt - pred) ** 2) + 1e-8
    return 10 * torch.log10(signal_power / noise_power).item()

def ssim_score(pred, gt):
    pred_np = pred.squeeze().cpu().numpy()
    gt_np = gt.squeeze().cpu().numpy()
    score, _ = ssim(pred_np, gt_np, full=True, data_range=1.0)
    return score

# Métricas y control
total_l1 = total_l2 = total_snr = total_ssim = n = 0
best_snr, worst_snr = -float('inf'), float('inf')
best_clip, worst_clip = -1, -1
os.makedirs("outputs", exist_ok=True)

best_metrics = {
    "l1": {"val": float("inf"), "idx": -1, "mel": None, "gt": None},
    "l2": {"val": float("inf"), "idx": -1, "mel": None, "gt": None},
    "snr": {"val": -float("inf"), "idx": -1, "mel": None, "gt": None},
    "ssim": {"val": -float("inf"), "idx": -1, "mel": None, "gt": None}
}

csv_path = os.path.join("outputs", "clip_metrics.csv")
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Clip", "L1 Loss", "L2 Loss", "SNR (dB)", "SSIM"])

for idx, (frames, mels_norm) in enumerate(test_dataloader):
    frames, mels_norm = frames.to(device), mels_norm.to(device)
    mel_pred_norm = reconstruct_full_mel_clip(frames, model)
    mel_pred = mel_pred_norm * global_subs + global_min
    mel_gt = mels_norm * global_subs + global_min
    T_pred = mel_pred.shape[-1]
    mel_gt = mel_gt[..., :T_pred]

    clip_l1 = l1_loss(mel_pred, mel_gt)
    clip_l2 = l2_loss(mel_pred, mel_gt)
    clip_snr = snr(mel_pred, mel_gt)
    clip_ssim = ssim_score(mel_pred[0, 0], mel_gt[0, 0])

    total_l1 += clip_l1
    total_l2 += clip_l2
    total_snr += clip_snr
    total_ssim += clip_ssim
    n += 1

    print(f"Clip {idx} - L1: {clip_l1:.4f} - L2: {clip_l2:.4f} - SNR: {clip_snr:.2f} dB - SSIM: {clip_ssim:.4f}")

    if clip_snr > best_snr:
        best_snr = clip_snr
        best_clip = idx
    if clip_snr < worst_snr:
        worst_snr = clip_snr
        worst_clip = idx

    for metric, value in zip(["l1", "l2", "snr", "ssim"], [clip_l1, clip_l2, clip_snr, clip_ssim]):
        if ((metric in ["l1", "l2"] and value < best_metrics[metric]["val"]) or
            (metric in ["snr", "ssim"] and value > best_metrics[metric]["val"])):
            best_metrics[metric] = {
                "val": value,
                "idx": idx,
                "mel": mel_pred[0, 0].cpu().numpy(),
                "gt": mel_gt[0, 0].cpu().numpy()
            }

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([idx, f"{clip_l1:.6f}", f"{clip_l2:.6f}", f"{clip_snr:.2f}", f"{clip_ssim:.4f}"])

# Métricas globales
avg_l1 = total_l1 / n
avg_l2 = total_l2 / n
avg_snr = total_snr / n
avg_ssim = total_ssim / n

print("\nMétricas globales (dataset completo):")
print(f"Avg L1 Loss: {avg_l1:.6f}")
print(f"Avg L2 Loss: {avg_l2:.6f}")
print(f"Avg SNR (dB): {avg_snr:.2f} dB")
print(f"Avg SSIM: {avg_ssim:.4f}")
print(f"\nMejor clip: {best_clip} con SNR = {best_snr:.2f} dB")
print(f"Peor clip: {worst_clip} con SNR = {worst_snr:.2f} dB")

with open(csv_path, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([])
    writer.writerow(["Promedios", f"{avg_l1:.6f}", f"{avg_l2:.6f}", f"{avg_snr:.2f}", f"{avg_ssim:.4f}"])

# Comparación visual en una sola figura
fig, axes = plt.subplots(4, 2, figsize=(12, 16))
fig.suptitle("Comparación de mejores clips por métrica", fontsize=16)
for i, metric in enumerate(["l1", "l2", "snr", "ssim"]):
    pred = best_metrics[metric]["mel"]
    gt = best_metrics[metric]["gt"]
    if pred is not None and gt is not None:
        axes[i, 0].imshow(pred, aspect='auto', origin='lower')
        axes[i, 0].set_title(f"{metric.upper()} - Reconstruido")
        axes[i, 1].imshow(gt, aspect='auto', origin='lower')
        axes[i, 1].set_title(f"{metric.upper()} - Ground Truth")
for ax in axes.flatten():
    ax.label_outer()
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig("outputs/comparacion_general.png")
plt.close()

print("✅ Comparación visual guardada en outputs/comparacion_general.png")
print("✅ Métricas guardadas en outputs/clip_metrics.csv")
