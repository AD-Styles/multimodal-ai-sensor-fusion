"""
🌐 Multimodal AI: From Sensor Fusion to Contrastive Pre-training
================================================================
Based on concepts from NVIDIA Deep Learning Institute (DLI)
 — Multimodal AI course —

Pipeline:
  1. Synthetic RGB + LiDAR data generation & exploration
  2. 3D Point-cloud visualization
  3. Single-modal vs Fusion model comparison
     (Early / Late / Intermediate Fusion)
  4. CLIP-style Contrastive Pre-training (FashionMNIST + Sobel outlines)
  5. Cross-modal Projection (LiDAR → RGB embedding space)
  6. Final results summary

"""

# ── Libraries ─────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 – 3-D projection
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE

# ── Global Setup ──────────────────────────────────────────────────────────────
os.makedirs('plots', exist_ok=True)

plt.rcParams['figure.dpi']    = 100
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titlesize'] = 12
sns.set_style('darkgrid')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch {torch.__version__}  |  Device: {DEVICE}\n")

# ──────────────────────────────────────────────────────────────────────────────
# Section 1. Synthetic Multimodal Dataset
# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE  = 32
N_CLASSES = 3          # 0 = Cube, 1 = Sphere, 2 = Torus
N_TRAIN   = 900
N_VAL     = 300
CLASS_NAMES = ['Cube', 'Sphere', 'Torus']


def _make_scene(label: int, cx: int, cy: int, size: int, img_size: int = 32):
    """Return (rgb [3,H,W], depth [H,W]) for a single synthetic scene."""
    rgb   = np.zeros((3, img_size, img_size), dtype=np.float32)
    depth = np.full((img_size, img_size), 10.0, dtype=np.float32)

    colors = [(0.90, 0.20, 0.20),   # Cube   — red
              (0.20, 0.80, 0.20),   # Sphere — green
              (0.20, 0.40, 0.90)]   # Torus  — blue

    yy, xx = np.mgrid[:img_size, :img_size]
    dist   = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    if label == 0:          # square → cube
        mask = (np.abs(xx - cx) < size / 2) & (np.abs(yy - cy) < size / 2)
    elif label == 1:        # circle → sphere
        mask = dist < size / 2
    else:                   # ring   → torus
        mask = (dist > size / 3) & (dist < size / 2)

    for ch, val in enumerate(colors[label]):
        rgb[ch][mask] = val

    depth_obj            = 5.0 - 2.0 * np.clip((size / 2 - dist) / (size / 2 + 1e-6), 0, 1)
    depth[mask]          = depth_obj[mask]

    rgb   += np.random.normal(0, 0.02, rgb.shape).astype(np.float32)
    depth += np.random.normal(0, 0.05, depth.shape).astype(np.float32)
    rgb    = np.clip(rgb, 0, 1)
    return rgb, depth


def build_dataset(n: int = 1200, seed: int = 42):
    np.random.seed(seed)
    rgb_l, lidar_l, label_l = [], [], []

    for i in range(n):
        lbl  = i % N_CLASSES
        cx   = np.random.randint(8, IMG_SIZE - 8)
        cy   = np.random.randint(8, IMG_SIZE - 8)
        sz   = np.random.randint(6, 12)
        rgb, depth = _make_scene(lbl, cx, cy, sz, IMG_SIZE)
        rgb_l.append(rgb)
        lidar_l.append(depth[None])          # add channel dim
        label_l.append(lbl)

    return (
        torch.tensor(np.array(rgb_l)),
        torch.tensor(np.array(lidar_l), dtype=torch.float32),
        torch.tensor(label_l, dtype=torch.long),
    )


print("⏳  Generating synthetic multimodal dataset …")
rgb_all, lidar_all, labels_all = build_dataset(N_TRAIN + N_VAL)
rgb_tr, rgb_va       = rgb_all[:N_TRAIN],   rgb_all[N_TRAIN:]
lidar_tr, lidar_va   = lidar_all[:N_TRAIN], lidar_all[N_TRAIN:]
labels_tr, labels_va = labels_all[:N_TRAIN],labels_all[N_TRAIN:]
print(f"  Train: {len(rgb_tr)}   Val: {len(rgb_va)}\n")

BATCH = 32
train_ds = TensorDataset(rgb_tr, lidar_tr, labels_tr)
val_ds   = TensorDataset(rgb_va, lidar_va, labels_va)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH)

# ─── Plot 1: Data exploration ─────────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
fig.suptitle('Synthetic Multimodal Dataset — RGB vs LiDAR Depth Comparison',
             fontsize=14, fontweight='bold')

for row, cls_name in enumerate(CLASS_NAMES):
    idxs = (labels_all == row).nonzero(as_tuple=True)[0][:2]
    for col, idx in enumerate(idxs):
        # RGB
        ax = axes[row, col * 2]
        ax.imshow(rgb_all[idx].permute(1, 2, 0).numpy())
        ax.set_title(f'{cls_name} — RGB (sample {col+1})', fontsize=9)
        ax.axis('off')
        # LiDAR depth
        ax = axes[row, col * 2 + 1]
        im = ax.imshow(lidar_all[idx][0].numpy(), cmap='plasma')
        ax.set_title(f'{cls_name} — LiDAR Depth (sample {col+1})', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.7, label='depth')

plt.tight_layout()
plt.savefig('plots/01_data_exploration.png', bbox_inches='tight')
plt.close()
print("✅  plots/01_data_exploration.png")

# ─── Plot 2: 3-D Point Cloud ─────────────────────────────────────────────────
# 배경(depth ~10)을 제거하고 오브젝트 포인트만 강조하여 시각화
BG_DEPTH   = 9.5    # 이 값 이상은 배경으로 간주
OBJ_COLORS = ['#E74C3C', '#27AE60', '#2980B9']   # Cube / Sphere / Torus

fig = plt.figure(figsize=(16, 6))
fig.patch.set_facecolor('#1A1A2E')
fig.suptitle('LiDAR → 3-D Point Cloud Reconstruction\n'
             '(Object points highlighted · Background filtered)',
             fontsize=13, fontweight='bold', color='white')

for subplot_idx, label in enumerate(range(N_CLASSES)):
    # 같은 클래스 샘플 3개를 합쳐 포인트 수를 늘림
    idxs  = (labels_all == label).nonzero(as_tuple=True)[0][:3]
    ax    = fig.add_subplot(1, 3, subplot_idx + 1, projection='3d')
    ax.set_facecolor('#0D1117')

    for i, idx in enumerate(idxs):
        depth = lidar_all[int(idx)][0].numpy()
        yy, xx = np.mgrid[:IMG_SIZE, :IMG_SIZE]

        # 오브젝트 포인트 (depth < BG_DEPTH)
        obj_mask = depth.flatten() < BG_DEPTH
        x_obj = (xx.flatten() / IMG_SIZE)[obj_mask]
        y_obj = (yy.flatten() / IMG_SIZE)[obj_mask]
        z_obj = depth.flatten()[obj_mask]

        # 배경 포인트 (아주 연하게)
        bg_mask = ~obj_mask
        x_bg = (xx.flatten() / IMG_SIZE)[bg_mask]
        y_bg = (yy.flatten() / IMG_SIZE)[bg_mask]
        z_bg = depth.flatten()[bg_mask]

        # 배경 — 반투명 회색 점 (sparse)
        ax.scatter(x_bg[::8], y_bg[::8], z_bg[::8],
                   c='#444466', alpha=0.08, s=3, marker='.')

        # 오브젝트 — 클래스별 색상, 깊이로 명도 조절
        if len(z_obj) > 0:
            z_norm = (z_obj - z_obj.min()) / (z_obj.max() - z_obj.min() + 1e-6)
            # 깊이가 낮을수록 (앞쪽) 밝게
            alpha_vals = 0.6 + 0.4 * (1 - z_norm)
            ax.scatter(x_obj, y_obj, z_obj,
                       c=OBJ_COLORS[label], alpha=0.85,
                       s=28, marker='o', edgecolors='none')

    ax.set_title(f'{CLASS_NAMES[label]}', fontweight='bold',
                 color='white', fontsize=12, pad=8)
    ax.set_xlabel('X', color='#AAAAAA', fontsize=8)
    ax.set_ylabel('Y', color='#AAAAAA', fontsize=8)
    ax.set_zlabel('Depth', color='#AAAAAA', fontsize=8)
    ax.tick_params(colors='#888888', labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#333355')
    ax.yaxis.pane.set_edgecolor('#333355')
    ax.zaxis.pane.set_edgecolor('#333355')
    ax.grid(True, alpha=0.15)
    ax.view_init(elev=30, azim=210)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('plots/02_pointcloud_3d.png', bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("✅  plots/02_pointcloud_3d.png")

# ──────────────────────────────────────────────────────────────────────────────
# Section 2. Fusion Model Architectures
# ──────────────────────────────────────────────────────────────────────────────
# Flat size after two MaxPool2d(2) on IMG_SIZE=32:  32 → 16 → 8 → 32*8*8 = 2048
_FLAT = 32 * 8 * 8   # 2048


class RGBEncoder(nn.Module):
    def __init__(self, emb=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(_FLAT, emb), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class LiDAREncoder(nn.Module):
    def __init__(self, emb=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(_FLAT, emb), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


# ── Single-modal baselines ───────────────────────────────────────────────────
class RGBOnlyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc  = RGBEncoder(64)
        self.head = nn.Linear(64, N_CLASSES)
    def forward(self, rgb, _): return self.head(self.enc(rgb))


class LiDAROnlyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc  = LiDAREncoder(64)
        self.head = nn.Linear(64, N_CLASSES)
    def forward(self, _, lidar): return self.head(self.enc(lidar))


# ── Fusion architectures ─────────────────────────────────────────────────────
class EarlyFusionNet(nn.Module):
    """Concatenate RGB + LiDAR at pixel level (4 input channels)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(_FLAT, 64), nn.ReLU(),
            nn.Linear(64, N_CLASSES),
        )
    def forward(self, rgb, lidar):
        return self.net(torch.cat([rgb, lidar], dim=1))


class LateFusionNet(nn.Module):
    """Train two separate encoders; fuse their embeddings at the head."""
    def __init__(self):
        super().__init__()
        self.rgb_enc   = RGBEncoder(64)
        self.lidar_enc = LiDAREncoder(64)
        self.head      = nn.Linear(128, N_CLASSES)
    def forward(self, rgb, lidar):
        return self.head(torch.cat([self.rgb_enc(rgb),
                                    self.lidar_enc(lidar)], dim=1))


class IntermediateFusionNet(nn.Module):
    """CatNet — fuse at intermediate feature-map level."""
    def __init__(self):
        super().__init__()
        self.rgb_branch   = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.lidar_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.combined = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(_FLAT, 64), nn.ReLU(),
            nn.Linear(64, N_CLASSES),
        )
    def forward(self, rgb, lidar):
        return self.combined(
            torch.cat([self.rgb_branch(rgb),
                       self.lidar_branch(lidar)], dim=1)
        )


# ── Training helpers ─────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, train: bool):
    model.train() if train else model.eval()
    tot_loss = tot_correct = tot_n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for rgb, lidar, lbl in loader:
            rgb, lidar, lbl = rgb.to(DEVICE), lidar.to(DEVICE), lbl.to(DEVICE)
            if train:
                optimizer.zero_grad()
            out  = model(rgb, lidar)
            loss = criterion(out, lbl)
            if train:
                loss.backward()
                optimizer.step()
            tot_loss    += loss.item() * len(lbl)
            tot_correct += (out.argmax(1) == lbl).sum().item()
            tot_n       += len(lbl)
    return tot_loss / tot_n, tot_correct / tot_n


EPOCHS    = 25
criterion = nn.CrossEntropyLoss()

model_zoo = {
    'RGB-Only':              RGBOnlyNet(),
    'LiDAR-Only':            LiDAROnlyNet(),
    'Early Fusion':          EarlyFusionNet(),
    'Late Fusion':           LateFusionNet(),
    'Intermediate Fusion':   IntermediateFusionNet(),
}

print("⏳  Training fusion architectures …")
hist_all = {}
for name, model in model_zoo.items():
    model = model.to(DEVICE)
    opt   = Adam(model.parameters(), lr=1e-3)
    hist  = {'tl': [], 'vl': [], 'va': []}
    for ep in range(EPOCHS):
        tl, _  = run_epoch(model, train_loader, opt, criterion, train=True)
        vl, va = run_epoch(model, val_loader,   opt, criterion, train=False)
        hist['tl'].append(tl); hist['vl'].append(vl); hist['va'].append(va)
    hist_all[name] = hist
    print(f"  [{name:<22s}]  Val Acc: {hist['va'][-1]*100:.1f}%")
print()

# ─── Plot 3: Training curves ──────────────────────────────────────────────────
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Fusion Architecture Comparison — Training Curves',
             fontsize=13, fontweight='bold')

for (name, hist), color in zip(hist_all.items(), PALETTE):
    axes[0].plot(hist['vl'], label=name, color=color, linewidth=1.8)
    axes[1].plot([a * 100 for a in hist['va']], label=name,
                 color=color, linewidth=1.8)

for ax, title, ylabel in zip(axes,
                              ['Validation Loss', 'Validation Accuracy (%)'],
                              ['Loss', 'Accuracy (%)']):
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/03_fusion_training_curves.png', bbox_inches='tight')
plt.close()
print("✅  plots/03_fusion_training_curves.png")

# ─── Plot 4: Final accuracy bar chart ─────────────────────────────────────────
final_accs = {k: v['va'][-1] * 100 for k, v in hist_all.items()}

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(final_accs.keys(), final_accs.values(),
              color=PALETTE, alpha=0.85, edgecolor='white', linewidth=1.5)
ax.bar_label(bars, fmt='%.1f%%', fontsize=11, fontweight='bold', padding=4)
ax.axhline(100 / N_CLASSES, color='gray', linestyle='--', alpha=0.7,
           label=f'Random Baseline ({100/N_CLASSES:.1f}%)')
ax.set_title('Final Validation Accuracy — Single-Modal vs Fusion Models',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 110)
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(axis='x', rotation=10)
plt.tight_layout()
plt.savefig('plots/04_fusion_accuracy_comparison.png', bbox_inches='tight')
plt.close()
print("✅  plots/04_fusion_accuracy_comparison.png")

# ──────────────────────────────────────────────────────────────────────────────
# Section 3. CLIP-style Contrastive Pre-training
#            Modalities: FashionMNIST image  ↔  Sobel-edge outline (sketch)
# ──────────────────────────────────────────────────────────────────────────────
print("⏳  Downloading FashionMNIST …")
_ftfm = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])
f_train = torchvision.datasets.FashionMNIST('./data', train=True,  download=True, transform=_ftfm)
f_val   = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=_ftfm)

CLIP_BATCH = 64
f_train_loader = DataLoader(f_train, batch_size=CLIP_BATCH, shuffle=True, drop_last=True)
f_val_loader   = DataLoader(f_val,   batch_size=CLIP_BATCH, shuffle=False)
print()

FASHION_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# Sobel kernels (kept on CPU for pre-processing)
_Gx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float)
_Gy = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype=torch.float)


def sobel_outline(imgs: torch.Tensor) -> torch.Tensor:
    """Convert a batch of grayscale images [B,1,H,W] to Sobel-edge outlines."""
    b = imgs.clone()
    b[b > 0.25] = 1.0;  b[b <= 0.25] = 0.0
    ex = F.conv2d(b, _Gx.to(b.device), padding=1)
    ey = F.conv2d(b, _Gy.to(b.device), padding=1)
    e  = ex + ey;       e[e != 0] = 1.0
    return e


# ─── Plot 5: Dataset + cosine similarity before training ─────────────────────
# Collect 1 sample per class
sample_imgs, sample_lbls = [], []
for img, lbl in f_val:
    if lbl not in sample_lbls:
        sample_imgs.append(img);  sample_lbls.append(lbl)
    if len(sample_imgs) == 10:
        break

s_tensor  = torch.stack(sample_imgs)          # [10,1,28,28]
s_outlines= sobel_outline(s_tensor)

fig = plt.figure(figsize=(18, 8))
fig.suptitle('Contrastive Pre-training: FashionMNIST Images ↔ Sobel-Edge Outlines',
             fontsize=13, fontweight='bold')

gs = fig.add_gridspec(3, 10, hspace=0.4, wspace=0.3)
for i in range(10):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(s_tensor[i][0], cmap='gray');
    ax.set_title(FASHION_NAMES[sample_lbls[i]], fontsize=7);  ax.axis('off')
    ax = fig.add_subplot(gs[1, i])
    ax.imshow(s_outlines[i][0].detach(), cmap='gray');  ax.axis('off')

# Raw pixel cosine similarity matrix (before any training)
n_sim = 8
fi  = s_tensor[:n_sim].view(n_sim, -1).numpy()
fo  = s_outlines[:n_sim].detach().view(n_sim, -1).numpy()
nfi = np.linalg.norm(fi, axis=1, keepdims=True)
nfo = np.linalg.norm(fo, axis=1, keepdims=True)
sim_raw = (fi @ fo.T) / (nfi * nfo.T + 1e-8)

ax_sim = fig.add_subplot(gs[2, 1:9])
sns.heatmap(sim_raw, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=0, vmax=1, ax=ax_sim,
            xticklabels=[FASHION_NAMES[l] for l in sample_lbls[:n_sim]],
            yticklabels=[FASHION_NAMES[l] for l in sample_lbls[:n_sim]])
ax_sim.set_title('Cosine Similarity Matrix — Images vs Outlines (before training)',
                 fontweight='bold')
ax_sim.tick_params(axis='x', rotation=30)

plt.savefig('plots/05_contrastive_data.png', bbox_inches='tight')
plt.close()
print("✅  plots/05_contrastive_data.png")


# ── Contrastive model definition ─────────────────────────────────────────────
# FashionMNIST 28×28 → pool → 14×14 → pool → 7×7 → flatten 1568
_FFLAT = 32 * 7 * 7   # 1568


class EmbedNet(nn.Module):
    def __init__(self, in_ch=1, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(_FFLAT, emb_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)   # L2-normalised


class CLIPModel(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.img_enc = EmbedNet(1, emb_dim)
        self.out_enc = EmbedNet(1, emb_dim)
        self.temp    = nn.Parameter(torch.tensor(0.07))

    def forward(self, imgs, outlines):
        return self.img_enc(imgs), self.out_enc(outlines)

    def contrastive_loss(self, ie, oe):
        t   = self.temp.clamp(0.01, 0.5)
        log = (ie @ oe.T) / t
        lbl = torch.arange(len(ie), device=ie.device)
        return (F.cross_entropy(log, lbl) + F.cross_entropy(log.T, lbl)) / 2


clip_model = CLIPModel(emb_dim=64).to(DEVICE)
clip_opt    = Adam(clip_model.parameters(), lr=3e-4)
CLIP_EPOCHS = 12
clip_losses = []

print("⏳  Training CLIP-style contrastive model …")
for ep in range(CLIP_EPOCHS):
    clip_model.train()
    ep_loss = 0; n_b = 0
    for imgs, _ in f_train_loader:
        imgs = imgs.to(DEVICE)
        outs = sobel_outline(imgs)
        clip_opt.zero_grad()
        ie, oe = clip_model(imgs, outs)
        loss   = clip_model.contrastive_loss(ie, oe)
        loss.backward(); clip_opt.step()
        ep_loss += loss.item(); n_b += 1
    clip_losses.append(ep_loss / n_b)
    print(f"  Epoch [{ep+1:02d}/{CLIP_EPOCHS}]  Contrastive Loss: {ep_loss/n_b:.4f}")
print()

# ─── Plot 6: Contrastive results ──────────────────────────────────────────────
clip_model.eval()
with torch.no_grad():
    s_g = s_tensor[:n_sim].to(DEVICE)
    o_g = s_outlines[:n_sim].to(DEVICE)
    ie_t, oe_t = clip_model(s_g, o_g)
    sim_trained = (ie_t @ oe_t.T).cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Contrastive Pre-training Results', fontsize=13, fontweight='bold')

axes[0].plot(clip_losses, color='steelblue', marker='o', linewidth=2)
axes[0].set_title('Contrastive Loss Curve (NT-Xent)', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].grid(True, alpha=0.3)

sns.heatmap(sim_trained, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, ax=axes[1],
            xticklabels=[FASHION_NAMES[l] for l in sample_lbls[:n_sim]],
            yticklabels=[FASHION_NAMES[l] for l in sample_lbls[:n_sim]])
axes[1].set_title('Cosine Similarity Matrix — Images vs Outlines (after training)',
                  fontweight='bold')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('plots/06_contrastive_results.png', bbox_inches='tight')
plt.close()
print("✅  plots/06_contrastive_results.png")

# ─── Plot 7: t-SNE of learned embeddings ─────────────────────────────────────
print("⏳  Computing t-SNE visualisation …")
clip_model.eval()
all_embs, all_lbls = [], []
with torch.no_grad():
    for imgs, lbls in f_val_loader:
        all_embs.append(clip_model.img_enc(imgs.to(DEVICE)).cpu().numpy())
        all_lbls.append(lbls.numpy())
        if sum(len(e) for e in all_embs) >= 2000:
            break
all_embs = np.concatenate(all_embs)[:2000]
all_lbls = np.concatenate(all_lbls)[:2000]

embs_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(all_embs)

fig, ax = plt.subplots(figsize=(12, 8))
sc = ax.scatter(embs_2d[:, 0], embs_2d[:, 1],
                c=all_lbls, cmap='tab10', alpha=0.55, s=8, vmin=0, vmax=9)
handles = [plt.Line2D([0], [0], marker='o', color='w',
           markerfacecolor=plt.cm.tab10(i / 10),
           markersize=9, label=FASHION_NAMES[i])
           for i in range(10)]
ax.legend(handles=handles, loc='upper right', fontsize=8)
ax.set_title('t-SNE of Contrastive Embeddings — FashionMNIST (image encoder)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('plots/07_tsne_embeddings.png', bbox_inches='tight')
plt.close()
print("✅  plots/07_tsne_embeddings.png")

# ──────────────────────────────────────────────────────────────────────────────
# Section 4. Cross-Modal Projection
#            Goal: project LiDAR embeddings → RGB embedding space
#            (analogous to CLIP's vision–language projection used in LLaVA etc.)
# ──────────────────────────────────────────────────────────────────────────────
print("⏳  Training cross-modal projector …")


class CrossModalProjector(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(),
            nn.Linear(128, dim),
        )
    def forward(self, x): return self.proj(x)


rgb_enc_proj   = RGBEncoder(64).to(DEVICE)
lidar_enc_proj = LiDAREncoder(64).to(DEVICE)
projector      = CrossModalProjector(64).to(DEVICE)

proj_opt = Adam(
    list(rgb_enc_proj.parameters())
    + list(lidar_enc_proj.parameters())
    + list(projector.parameters()),
    lr=1e-3,
)
PROJ_EPOCHS = 20
proj_losses = []

for ep in range(PROJ_EPOCHS):
    rgb_enc_proj.train(); lidar_enc_proj.train(); projector.train()
    ep_loss = 0; n_b = 0
    for rgb, lidar, _ in train_loader:
        rgb, lidar = rgb.to(DEVICE), lidar.to(DEVICE)
        proj_opt.zero_grad()
        r_emb = F.normalize(rgb_enc_proj(rgb), dim=-1)
        l_emb = F.normalize(lidar_enc_proj(lidar), dim=-1)
        p_emb = F.normalize(projector(l_emb), dim=-1)
        loss  = F.mse_loss(p_emb, r_emb.detach())
        loss.backward(); proj_opt.step()
        ep_loss += loss.item(); n_b += 1
    proj_losses.append(ep_loss / n_b)
    print(f"  Epoch [{ep+1:02d}/{PROJ_EPOCHS}]  Projection MSE: {ep_loss/n_b:.6f}")
print()

# ─── Plot 8: Embedding space alignment (개선된 버전) ─────────────────────────
# t-SNE를 RGB+Projected 공동 계산 → 같은 좌표계에서 정렬 여부를 직접 비교
rgb_enc_proj.eval(); lidar_enc_proj.eval(); projector.eval()

with torch.no_grad():
    s_rgb   = rgb_va[:120].to(DEVICE)
    s_lidar = lidar_va[:120].to(DEVICE)
    s_lbl   = labels_va[:120].numpy()

    r_emb_np = rgb_enc_proj(s_rgb).cpu().numpy()
    l_emb_np = lidar_enc_proj(s_lidar).cpu().numpy()
    p_emb_np = projector(torch.tensor(l_emb_np).to(DEVICE)).cpu().numpy()

# ── 공동 t-SNE: [RGB, Projected]를 동일 공간에 투영 (LiDAR before는 별도)
tsne_joint = TSNE(n_components=2, random_state=42, perplexity=15)
rp_2d = tsne_joint.fit_transform(np.concatenate([r_emb_np, p_emb_np]))
r2d   = rp_2d[:120]      # RGB
p2d   = rp_2d[120:]      # Projected LiDAR

tsne_lidar = TSNE(n_components=2, random_state=42, perplexity=15)
l2d = tsne_lidar.fit_transform(l_emb_np)  # LiDAR before (독립 좌표계)

PAIR_COLORS = ['#E74C3C', '#27AE60', '#2980B9']
MARKERS     = ['o', 's', '^']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Cross-Modal Projection — Embedding Space Alignment',
             fontsize=14, fontweight='bold')

# ① RGB Embeddings
for ci, cn in enumerate(CLASS_NAMES):
    mask = s_lbl == ci
    axes[0].scatter(r2d[mask, 0], r2d[mask, 1],
                    label=cn, color=PAIR_COLORS[ci],
                    marker=MARKERS[ci], alpha=0.80, s=45, edgecolors='white', linewidths=0.4)
axes[0].set_title('① RGB Embeddings', fontweight='bold', fontsize=11)
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.2)

# ② LiDAR before projection (독립 t-SNE)
for ci, cn in enumerate(CLASS_NAMES):
    mask = s_lbl == ci
    axes[1].scatter(l2d[mask, 0], l2d[mask, 1],
                    label=cn, color=PAIR_COLORS[ci],
                    marker=MARKERS[ci], alpha=0.80, s=45, edgecolors='white', linewidths=0.4)
axes[1].set_title('② LiDAR Embeddings\n(before projection)', fontweight='bold', fontsize=11)
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.2)

# ③ 오버레이: RGB(투명) + Projected(불투명) — 같은 t-SNE 공간
for ci, cn in enumerate(CLASS_NAMES):
    mask = s_lbl == ci
    # RGB — 연한 배경으로
    axes[2].scatter(r2d[mask, 0], r2d[mask, 1],
                    color=PAIR_COLORS[ci], marker='o',
                    alpha=0.22, s=60, edgecolors='none')
    # Projected LiDAR — 진하게
    axes[2].scatter(p2d[mask, 0], p2d[mask, 1],
                    label=cn, color=PAIR_COLORS[ci], marker='D',
                    alpha=0.85, s=38, edgecolors='white', linewidths=0.5)

# 범례 설명 추가
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=9, alpha=0.3, label='RGB (reference)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
           markersize=8, label='LiDAR → Projected'),
] + [
    Line2D([0], [0], marker='D', color='w', markerfacecolor=PAIR_COLORS[ci],
           markersize=8, label=cn)
    for ci, cn in enumerate(CLASS_NAMES)
]
axes[2].legend(handles=legend_elems, fontsize=8, loc='upper right')
axes[2].set_title('③ LiDAR → Projected\n(overlaid with RGB reference)',
                  fontweight='bold', fontsize=11)
axes[2].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('plots/08_cross_modal_projection.png', bbox_inches='tight')
plt.close()
print("✅  plots/08_cross_modal_projection.png")

# ─── Plot 9: Projection loss curve ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(proj_losses, color='#C44E52', marker='o', linewidth=2)
ax.set_title('Cross-Modal Projector — Training Loss (MSE)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/09_projection_loss.png', bbox_inches='tight')
plt.close()
print("✅  plots/09_projection_loss.png")

# ──────────────────────────────────────────────────────────────────────────────
# Section 5. Final Summary Dashboard
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Multimodal AI — Final Results Summary', fontsize=14, fontweight='bold')

# Accuracy bar
names_sorted = list(final_accs.keys())
vals_sorted  = [final_accs[n] for n in names_sorted]
bars = axes[0].bar(names_sorted, vals_sorted, color=PALETTE,
                   alpha=0.85, edgecolor='white', linewidth=1.5)
axes[0].bar_label(bars, fmt='%.1f%%', fontsize=10, fontweight='bold', padding=3)
axes[0].axhline(100 / N_CLASSES, color='gray', linestyle='--', alpha=0.7,
                label=f'Random Baseline ({100/N_CLASSES:.1f}%)')
axes[0].set_title('Object Classification Accuracy by Model Type', fontweight='bold')
axes[0].set_ylabel('Val Accuracy (%)'); axes[0].set_ylim(0, 108)
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, axis='y')
axes[0].tick_params(axis='x', rotation=12)

# Dual loss curves
ax_l = axes[1]
ax_r = ax_l.twinx()
ax_l.plot(clip_losses, color='steelblue', marker='o',
          label='Contrastive Loss (CLIP-style)')
ax_r.plot(proj_losses, color='#C44E52',   marker='s', linestyle='--',
          label='Projection MSE Loss')
ax_l.set_xlabel('Epoch')
ax_l.set_ylabel('Contrastive Loss', color='steelblue')
ax_r.set_ylabel('Projection MSE',   color='#C44E52')
ax_l.set_title('Pre-training & Projection Loss Curves', fontweight='bold')
lh1, ll1 = ax_l.get_legend_handles_labels()
lh2, ll2 = ax_r.get_legend_handles_labels()
ax_l.legend(lh1 + lh2, ll1 + ll2, loc='upper right', fontsize=9)
ax_l.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/10_final_summary.png', bbox_inches='tight')
plt.close()
print("✅  plots/10_final_summary.png")

# ── Console summary ───────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  Multimodal AI — Experiment Results")
print("=" * 60)
for name, acc in final_accs.items():
    marker = "★" if acc == max(final_accs.values()) else " "
    print(f"  {marker} {name:<25s}: {acc:>5.1f}%")
print()
print(f"  Contrastive Loss  (final): {clip_losses[-1]:.4f}")
print(f"  Projection MSE   (final): {proj_losses[-1]:.6f}")
print("=" * 60)
print("\n  All plots saved → plots/ directory")
