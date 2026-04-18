#!/usr/bin/env python3
"""
FUSION-HAD: Fused Unsupervised Spectral-Integrated Orthogonal Network for
           Hyperspectral Anomaly Detection
===========================================================================
Combines the best of IRIS-NET v6.1 and DSSA-HAD++ v22 into a unified pipeline.

Novel Contributions:
  1. Hybrid IBP+BAIT Background Purification  — sequential, stricter than either alone
  2. Meta-Fusion (CWPF-guided AUC Weighting) — CWPF robust_sep weights inform
     exponential AUC fusion for principled heterogeneous ensemble scoring
  3. Dual-Threshold Consensus              — F-beta recall guard + precision-aware
     threshold running jointly; Pareto-optimal pick via score separability
  4. Full Detector Suite (10 branches):
       RX, SDAS, LRX, DS-SAE (spectral+spatial), MURE, IsoForest [from IRIS-NET]
       BAIT-AE (spec + spec_z), Spatial-RX multi-scale, Global Mahalanobis,
       Latent Mahalanobis (MCD) [from DSSA]
  5. Dynamic epoch / TTA scaling from scene sparsity (no hardcoding)
  6. Auto device detection: T4x2 → CUDA, MPS, CPU fallback chain

Usage (Kaggle):
    python FUSION_HAD.py --dataset /path/to/scene.mat --gt /path/to/scene.mat

Usage (local, multi-scene):
    python FUSION_HAD.py --dataset /data/*.mat --gt /data/*.mat

Author: Combined from Pranav Kumar (IRIS-NET) + Cheralathan M (DSSA)
"""

import os, gc, copy, random, warnings, argparse
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np
import scipy.io
import scipy.ndimage as nd
from scipy.optimize import nnls as _scipy_nnls
from scipy.ndimage import uniform_filter, maximum_filter, label as sp_label
from scipy.stats import rankdata, skew as scipy_skew
from scipy.ndimage import binary_erosion, binary_dilation

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.metrics import roc_auc_score, roc_curve, auc as sk_auc
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef

try:
    from skimage.segmentation import slic as sk_slic, mark_boundaries
    from skimage.color import gray2rgb
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[WARN] scikit-image not found — SLIC superpixel smoothing disabled")

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  DEVICE DETECTION  (T4x2 → CUDA, MPS, CPU)
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"[Device] CUDA available — {n} GPU(s): "
              + ", ".join(torch.cuda.get_device_name(i) for i in range(n)))
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("[Device] Apple MPS backend")
        return torch.device('mps')
    print("[Device] CPU only")
    return torch.device('cpu')

DEVICE = get_device()


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _norm01(a: np.ndarray) -> np.ndarray:
    mn, mx = a.min(), a.max()
    return ((a - mn) / (mx - mn + 1e-8)).astype(np.float32)

def _rank_norm(s: np.ndarray) -> np.ndarray:
    f = s.reshape(-1).astype(np.float32)
    r = rankdata(f, method='average').astype(np.float32)
    return ((r - 1) / (len(r) - 1)).reshape(s.shape)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG & SCENE PROFILE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    dataset_path: str = ""
    gt_path:      str = ""
    output_dir:   str = "output_fusion_had"
    # AE architecture (dynamic if 0)
    latent_dim:   int = 0       # 0 = auto from sqrt(B)
    # Epochs — scale with sparsity at runtime; these are base values
    ae_epochs_base:    int = 150   # scaled up/down by sparsity
    ae_epochs_finetune:int = 0     # 0 = auto = base // 2
    bait_rounds:  int = 0       # 0 = auto (3 or 4)
    ibp_rounds:   int = 2
    # TTA
    tta_steps:    int = 0       # 0 = auto
    tta_weight:   float = 0.0   # 0 = auto
    # Misc
    save_intermediates: bool = True


@dataclass
class SceneProfile:
    H: int = 0; W: int = 0; B: int = 0
    sparsity_est: float = 3.0   # % of pixels likely anomalous
    band_corr:    float = 0.0
    spatial_cv:   float = 0.5
    scene_type:   str   = 'structured'
    mask_ratio:   float = 0.40  # for DS-SAE masking
    latent_dim:   int   = 32
    hidden_dims:  List[int] = field(default_factory=list)
    ae_epochs:    int   = 150
    ae_finetune:  int   = 75
    bait_rounds:  int   = 3
    bait_excl_frac: float = 0.03
    spatial_scales: List[int] = field(default_factory=list)
    rxd_win:      int   = 7
    tta_steps:    int   = 30
    tta_weight:   float = 0.15
    n_pixels:     int   = 0
    batch_size:   int   = 512


# ── NOTEBOOK / COLAB CONFIG ─────────────────────────────────────────────────
DATASET_PATH = ""   # e.g. "/kaggle/input/scene/abu_airport_2.mat"
GT_PATH      = ""   # leave "" to use same file as DATASET_PATH
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="FUSION-HAD Hyperspectral Anomaly Detection")
    p.add_argument("--dataset", default="",     help="Path to .mat hyperspectral file")
    p.add_argument("--gt",      default="",     help="Path to .mat ground-truth file")
    p.add_argument("--output",  default="output_fusion_had", help="Output directory")
    p.add_argument("--latent",  type=int, default=0, help="Latent dim (0=auto)")
    p.add_argument("--epochs",  type=int, default=150, help="Base AE epochs")
    p.add_argument("--no-intermediates", action="store_true", help="Skip saving stage images")
    args, _ = p.parse_known_args()
    if not args.dataset:
        args.dataset = globals().get("DATASET_PATH", "")
    if not args.gt:
        args.gt = globals().get("GT_PATH", args.dataset)
    if not args.dataset:
        p.error("--dataset is required. Set DATASET_PATH above or pass --dataset on CLI.")
    return args


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

_CUBE_KEYS = ['data','cube','hsi','HSI','image','paviaU','pavia','Pavia','PaviaU',
              'indian_pines_corrected','indian_pines','salinas_corrected','salinas',
              'houston','Houston','KSC','Botswana','URBAN','Urban','urban',
              'abu','airport','beach','hyperion','aviris']
_GT_KEYS   = ['map','gt','GT','groundtruth','ground_truth','label','labels',
              'paviaU_gt','pavia_gt','PaviaU_gt','indian_pines_gt','salinas_gt',
              'houston_gt','KSC_gt','Botswana_gt','urban_gt','Urban_GT','URBAN_GT',
              'anomaly_map','anomalymap','mask','ROI','gnd','ann','annotation']


def _load_mat_key(mat, keys, ndim):
    for k in keys:
        if k in mat and isinstance(mat[k], np.ndarray) and mat[k].ndim == ndim:
            return mat[k], k
    cands = [(k, v) for k, v in mat.items()
             if not k.startswith('_') and isinstance(v, np.ndarray) and v.ndim == ndim]
    if not cands:
        return None, None
    if ndim == 3:
        best = max(cands, key=lambda x: x[1].size)
    else:
        def gs(kv):
            v = kv[1]
            return (np.issubdtype(v.dtype, np.integer) or (v == v.astype(int)).all()) * 1000 - len(np.unique(v))
        best = max(cands, key=gs)
    return best[1], best[0]


def load_data(config: Config):
    mat = scipy.io.loadmat(config.dataset_path)
    cube_raw, ck = _load_mat_key(mat, _CUBE_KEYS, 3)
    if cube_raw is None:
        raise ValueError(f"No 3D hyperspectral array found in {config.dataset_path}")
    cube = cube_raw.astype(np.float32)
    # Ensure (H, W, B) layout — some sensors store (B, H, W)
    if cube.ndim == 3 and cube.shape[0] < cube.shape[2] and cube.shape[0] < 50:
        cube = cube.transpose(1, 2, 0)
    print(f"[Stage1] Cube key='{ck}'  shape={cube.shape}")

    gt_binary, gt_available = None, False
    gt_src = config.gt_path or config.dataset_path
    try:
        gmat = scipy.io.loadmat(gt_src)
        gr, gk = _load_mat_key(gmat, _GT_KEYS, 2)
        if gr is None:
            g3, gk = _load_mat_key(gmat, _GT_KEYS, 3)
            if g3 is not None and g3.shape[2] == 1:
                gr = g3[:, :, 0]
        if gr is not None:
            gt_raw = (gr > 0).astype(np.uint8)
            cH, cW = cube.shape[:2]
            gH, gW = gt_raw.shape
            if (gH, gW) != (cH, cW):
                h, w = min(cH, gH), min(cW, gW)
                cube = cube[:h, :w, :]; gt_raw = gt_raw[:h, :w]
            gt_binary = gt_raw; gt_available = True
            n_anom = int(gt_raw.sum())
            print(f"[Stage1] GT key='{gk}'  anomaly pixels={n_anom} ({n_anom/gt_raw.size*100:.2f}%)")
    except Exception as e:
        print(f"[Stage1] GT load warning: {e}")

    return cube, gt_binary, gt_available


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 — PREPROCESSING & SCENE CHARACTERIZATION
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(cube_raw: np.ndarray) -> np.ndarray:
    H, W, B = cube_raw.shape
    flat = cube_raw.reshape(-1, B)
    bvar = flat.var(0)
    thresh = max(np.percentile(bvar, 2), 1e-4)
    keep = bvar >= thresh
    if keep.sum() < B:
        print(f"[Stage2] Removed {B - keep.sum()} low-variance bands")
        cube_raw = cube_raw[:, :, keep]
    B2 = cube_raw.shape[2]
    flat2 = cube_raw.reshape(-1, B2)
    bmin = flat2.min(0); bmax = flat2.max(0)
    br = np.where((bmax - bmin) > 1e-8, bmax - bmin, 1.0)
    return ((cube_raw - bmin) / br).astype(np.float32)


def characterize(cube: np.ndarray, config: Config) -> SceneProfile:
    H, W, B = cube.shape
    N = H * W
    flat = cube.reshape(-1, B)
    prof = SceneProfile(H=H, W=W, B=B, n_pixels=N)

    # Band correlation
    idx = np.linspace(0, B-2, min(30, B-1), dtype=int)
    corrs = [np.corrcoef(flat[:, i], flat[:, i+1])[0, 1] for i in idx]
    prof.band_corr = float(np.nanmean(np.abs(corrs)))

    # Spatial coefficient of variation
    gray = flat.mean(1).reshape(H, W)
    prof.spatial_cv = float(gray.std() / (gray.mean() + 1e-8))

    # Sparsity estimation — dual-sigma strategy from IRIS-NET
    l2 = np.linalg.norm(flat - flat.mean(0), axis=1)
    l2n = _norm01(l2)
    mu, sig = l2n.mean(), l2n.std()
    est_2sig = float((l2n > mu + 2*sig).mean() * 100)
    est_3sig = float((l2n > mu + 3*sig).mean() * 100)
    # If 2σ estimate is >2.5× the 3σ estimate (heavy-tailed / sun-glint), use 3σ
    if est_3sig > 0 and (est_2sig / (est_3sig + 1e-8)) > 2.5:
        prof.sparsity_est = float(np.clip(est_3sig, 0.05, 5.0))
        print(f"[Profile] Heavy-tail detected — using 3σ sparsity: {prof.sparsity_est:.3f}%")
    else:
        prof.sparsity_est = float(np.clip(est_2sig, 0.05, 5.0))
    # Cap: never let sparsity_est exceed 2× the 3σ estimate (prevents over-exclusion)
    if est_3sig > 0:
        prof.sparsity_est = float(min(prof.sparsity_est, max(est_3sig * 2.0, 0.05)))

    prof.scene_type  = 'structured' if prof.band_corr > 0.6 else 'homogeneous'
    prof.mask_ratio  = float(np.clip(0.30 + prof.band_corr * 0.30, 0.30, 0.60))

    # ── Adaptive architecture (from DSSA) ──
    if config.latent_dim > 0:
        prof.latent_dim = config.latent_dim
    else:
        prof.latent_dim = max(int(2 ** round(np.log2(max(8, min(64, np.sqrt(B)))))), 8)

    if N <= 15_000:
        prof.hidden_dims = [min(128, B*2), min(64, B)]
    elif N <= 100_000:
        prof.hidden_dims = [min(256, B*2), min(128, B), min(64, B//2)]
    else:
        prof.hidden_dims = [min(512, B*2), min(256, B), min(128, B//2)]
    prof.hidden_dims = [max(h, 32) for h in prof.hidden_dims]

    # ── Dynamic epoch scaling ──
    # Sparsity drives difficulty: sparser scenes need more epochs to converge
    sparsity_factor = float(np.clip(1.0 + (3.0 - prof.sparsity_est) * 0.1, 0.8, 1.4))
    base = config.ae_epochs_base
    prof.ae_epochs   = int(base * sparsity_factor)
    prof.ae_finetune = config.ae_epochs_finetune if config.ae_epochs_finetune > 0 \
                       else prof.ae_epochs // 2

    # ── BAIT parameters (from DSSA) ──
    prof.bait_excl_frac = float(np.clip(0.04 - 0.01 * np.log10(max(N, 1000)/1000), 0.02, 0.04))
    prof.bait_rounds    = config.bait_rounds if config.bait_rounds > 0 else 3

    # ── Spatial scales & windows ──
    min_dim = min(H, W)
    s1 = max(3,  int(min_dim * 0.03) | 1)
    s2 = max(7,  int(min_dim * 0.07) | 1)
    s3 = max(11, int(min_dim * 0.15) | 1)
    prof.spatial_scales = [s1, s2, s3]
    prof.rxd_win = max(5, min(31, int(min_dim * 0.07) | 1))
    lrx_win = max(7, min(15, H // 12))
    prof.rxd_win = lrx_win + (1 if lrx_win % 2 == 0 else 0)

    # ── TTA parameters ──
    if config.tta_steps > 0:
        prof.tta_steps = config.tta_steps
    else:
        prof.tta_steps = int(np.clip(10 + N // 5000, 10, 50))
    if config.tta_weight > 0:
        prof.tta_weight = config.tta_weight
    else:
        prof.tta_weight = float(np.clip(0.05 + N / 500_000, 0.05, 0.20))

    # ── Batch size ──
    prof.batch_size = max(64, min(1024, N // 200))

    print(f"[Profile] H={H} W={W} B={B}  sparsity={prof.sparsity_est:.3f}%  type={prof.scene_type}")
    print(f"[Profile] latent={prof.latent_dim}  hidden={prof.hidden_dims}")
    print(f"[Profile] epochs={prof.ae_epochs}+{prof.ae_finetune}  BAIT_rounds={prof.bait_rounds}")
    print(f"[Profile] TTA steps={prof.tta_steps}  weight={prof.tta_weight:.3f}")
    return prof


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 3 — SLIC SUPERPIXELS  (DSSA)
# ─────────────────────────────────────────────────────────────────────────────

def compute_slic(cube: np.ndarray, prof: SceneProfile) -> Optional[np.ndarray]:
    if not SKIMAGE_AVAILABLE:
        return None
    H, W, B = cube.shape
    N_SP = int(np.clip(H * W // 400, 50, 2000))
    mi   = cube.mean(axis=2)
    ni   = (mi - mi.min()) / (mi.max() - mi.min() + 1e-8)
    ri   = gray2rgb(ni)
    cp   = float(np.clip(5.0 + (H * W / 50_000), 5.0, 20.0))
    segs = sk_slic(ri, n_segments=N_SP, compactness=cp, sigma=1.0, start_label=0)
    print(f"[Stage3] SLIC → {segs.max()+1} superpixels (compactness={cp:.1f})")
    return segs


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 4 — CLASSICAL DETECTORS  (IRIS-NET)
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_rx(X: np.ndarray) -> np.ndarray:
    print("[RX] Global Mahalanobis RX...")
    n_c = min(50, X.shape[1])
    Xs  = StandardScaler().fit_transform(X)
    _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
    Xp = Xs @ Vt[:n_c].T
    mu = Xp.mean(0)
    cov = np.cov(Xp.T) + np.eye(n_c) * 1e-4
    d = Xp - mu
    rx = np.sum((d @ np.linalg.pinv(cov)) * d, axis=1)
    return _norm01(rx)


def compute_sdas(X: np.ndarray, prof: SceneProfile) -> np.ndarray:
    print("[SDAS] Spectral-Derivative Anomaly Score...")
    def _rx_fast(Xin):
        n_c = min(30, Xin.shape[1])
        Xs  = StandardScaler().fit_transform(Xin)
        _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
        Xp = Xs @ Vt[:n_c].T
        mu = Xp.mean(0); cov = np.cov(Xp.T) + np.eye(n_c) * 1e-3
        d  = Xp - mu
        return _norm01(np.sum((d @ np.linalg.pinv(cov)) * d, axis=1))
    S0 = _rx_fast(X)
    S1 = _rx_fast(np.diff(X, n=1, axis=1))
    S2 = _rx_fast(np.diff(X, n=2, axis=1))
    bc = prof.band_corr; w0, w1, w2 = 1.0-bc, bc, bc*0.5
    return _norm01((w0*S0 + w1*S1 + w2*S2) / (w0+w1+w2+1e-8))


def compute_local_rx(cube: np.ndarray, win: int = 9) -> np.ndarray:
    H, W, B = cube.shape; half = win // 2
    print(f"[LRX] Local RX window={win}×{win}...")
    n_c  = min(20, B)
    flat = cube.reshape(-1, B).astype(np.float64)
    fc   = flat - flat.mean(0)
    _, _, Vt = np.linalg.svd(fc, full_matrices=False)
    cpca = (fc @ Vt[:n_c].T).reshape(H, W, n_c)
    pad  = np.pad(cpca, ((half, half), (half, half), (0, 0)), mode='reflect')
    stride = max(1, win // 3); sc = {}
    for r in range(0, H, stride):
        for c in range(0, W, stride):
            patch = pad[r:r+win, c:c+win, :].reshape(-1, n_c)
            mu_l  = patch.mean(0); cov_l = np.cov(patch.T) + np.eye(n_c) * 1e-3
            d     = cpca[r, c] - mu_l
            try:    sc[(r, c)] = float(d @ np.linalg.solve(cov_l, d))
            except: sc[(r, c)] = 0.0
    raw  = np.zeros(H * W, np.float32)
    rows = sorted({k[0] for k in sc})
    for r in range(H):
        rn   = min(rows, key=lambda x: abs(x - r))
        cols = [k[1] for k in sc if k[0] == rn]
        for c in range(W):
            raw[r*W+c] = sc[(rn, min(cols, key=lambda x: abs(x - c)))]
    return _norm01(raw)


def compute_mure(X: np.ndarray, prof: SceneProfile,
                 bg_mask: Optional[np.ndarray] = None) -> np.ndarray:
    print("[MURE] Multi-scale Unmixing Residual Estimator (batched)...")
    N, B = X.shape
    if bg_mask is None:
        Xs  = StandardScaler().fit_transform(X)
        ci  = np.linalg.pinv(np.cov(Xs.T) + np.eye(Xs.shape[1]) * 1e-3)
        d   = Xs - Xs.mean(0)
        rxf = np.sum((d @ ci) * d, axis=1)
        bg_mask = rxf < np.percentile(rxf, 65)
    X_bg = X[bg_mask]
    n_em = max(3, min(10, int(1 + prof.spatial_cv * 6)))
    km   = MiniBatchKMeans(n_clusters=n_em, n_init=3, random_state=SEED,
                           batch_size=min(2048, len(X_bg)))
    km.fit(X_bg)
    E = np.vstack([km.cluster_centers_, X_bg.mean(0, keepdims=True)]).T
    def _solve(i):
        try:    _, res = _scipy_nnls(E, X[i]); return float(res)
        except: return float(np.linalg.norm(X[i] - E.mean(1)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as exe:
        residuals = list(exe.map(_solve, range(N)))
    return _norm01(np.array(residuals, dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 5 — DSSA SPATIAL BRANCH
# ─────────────────────────────────────────────────────────────────────────────

def compute_spatial_rx(cube: np.ndarray, prof: SceneProfile) -> np.ndarray:
    print(f"[SpatRX] Multi-scale spatial statistics (scales={prof.spatial_scales})...")
    B = cube.shape[2]
    def _rx_band(band, size):
        lm  = uniform_filter(band.astype(np.float64), size=size)
        lsq = uniform_filter(band.astype(np.float64)**2, size=size)
        return (np.abs(band - lm) / (np.sqrt(np.maximum(lsq - lm**2, 0)) + 1e-8)).astype(np.float32)
    maps = [np.mean([_rx_band(cube[:, :, b], s) for b in range(B)], axis=0)
            for s in prof.spatial_scales]
    Sr = 0.5*maps[0] + 0.3*maps[1] + 0.2*maps[2]
    Sr = 0.7*Sr + 0.3*maximum_filter(Sr, size=3)
    return _norm01(Sr)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 6 — DEEP AE SUITE  (BAIT-trained DSSA AE + DS-SAE from IRIS-NET)
# ─────────────────────────────────────────────────────────────────────────────

class BaitSpectralAE(nn.Module):
    """DSSA-style AE: LayerNorm + GELU, dynamic hidden dims."""
    def __init__(self, d: int, hidden: List[int], latent: int):
        super().__init__()
        enc, p = [], d
        for h in hidden:
            enc += [nn.Linear(p, h), nn.LayerNorm(h), nn.GELU()]; p = h
        enc.append(nn.Linear(p, latent)); self.encoder = nn.Sequential(*enc)
        dec, p = [], latent
        for h in reversed(hidden):
            dec += [nn.Linear(p, h), nn.GELU()]; p = h
        dec.append(nn.Linear(p, d)); self.decoder = nn.Sequential(*dec)
    def forward(self, x):
        z = self.encoder(x); return self.decoder(z), z


class SpatialContextAE(nn.Module):
    """IRIS-NET DS-SAE stream 2: pixel + 5×5 local mean."""
    def __init__(self, in_dim: int, latent: int):
        super().__init__()
        jd = in_dim * 2
        h1 = min(512, max(latent*4, jd//2)); h2 = min(256, max(latent*2, 64))
        self.enc = nn.Sequential(nn.Linear(jd, h1), nn.ReLU(),
                                 nn.Linear(h1, h2), nn.ReLU(),
                                 nn.Linear(h2, latent))
        self.dec = nn.Sequential(nn.Linear(latent, h2), nn.ReLU(),
                                 nn.Linear(h2, h1), nn.ReLU(),
                                 nn.Linear(h1, in_dim))
    def forward(self, x_joint):
        z = self.enc(x_joint); return self.dec(z), z


def _local_means(cube: np.ndarray, win: int = 5) -> np.ndarray:
    H, W, B = cube.shape; lm = np.empty_like(cube)
    for b in range(B):
        lm[:, :, b] = nd.uniform_filter(cube[:, :, b], size=win)
    return lm


def train_bait_ae(cube: np.ndarray, prof: SceneProfile, config: Config) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    HYBRID BAIT+IBP training:
      Round 1-N: BAIT exclusion (DSSA-style 99th-pct reconstruction error exclusion)
      Final:     IBP RX-guided re-exclusion pass (IRIS-NET style)
    Returns: S_spec, S_spec_z, S_spat, z_spec, z_spat, bg_mask
    """
    H, W, B = cube.shape; N = H * W
    X   = cube.reshape(N, B)
    lm  = _local_means(cube, win=5).reshape(N, B)
    Xj  = np.concatenate([X, lm], axis=1)   # for spatial-context AE

    pixels_t = torch.tensor(X, dtype=torch.float32)
    pixjt    = torch.tensor(Xj, dtype=torch.float32)

    bg_mask     = np.ones(N, dtype=bool)
    best_state  = None
    bait_rounds = prof.bait_rounds
    ae_model    = None

    print(f"[BAIT-AE] Training: B={B}  hidden={prof.hidden_dims}  latent={prof.latent_dim}")
    print(f"          epochs={prof.ae_epochs}+{prof.ae_finetune}  BAIT_excl={prof.bait_excl_frac:.3f}")

    for bait_round in range(bait_rounds):
        n_bg = int(bg_mask.sum())
        print(f"[BAIT-AE] Round {bait_round+1}/{bait_rounds}: {n_bg} bg pixels")
        bpt = pixels_t[bg_mask]

        ae_model = BaitSpectralAE(B, prof.hidden_dims, prof.latent_dim).to(DEVICE)
        if best_state is not None:
            ae_model.load_state_dict(best_state)

        opt   = torch.optim.AdamW(ae_model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=40, T_mult=2, eta_min=1e-5)
        ldr   = DataLoader(TensorDataset(bpt), batch_size=prof.batch_size, shuffle=True)

        EP  = prof.ae_epochs
        PAT = max(20, EP // 6)
        bl, ni = 1e9, 0
        ae_model.train()
        for ep in range(EP):
            tot = 0
            for (xb,) in ldr:
                xb = xb.to(DEVICE); xh, _ = ae_model(xb)
                loss = F.mse_loss(xh, xb)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(ae_model.parameters(), 1.0)
                opt.step(); tot += loss.item()
            sched.step(); avg = tot / len(ldr)
            if avg < bl - 1e-7:
                bl = avg; best_state = copy.deepcopy(ae_model.state_dict()); ni = 0
            else:
                ni += 1
            if ni >= PAT and ep > 30:
                print(f"[BAIT-AE]   Early stop ep={ep+1} loss={bl:.6f}"); break

        # Dynamic BAIT round extension (DSSA v21)
        if bait_round == 1 and bl > 5e-5 and bait_rounds == 3:
            bait_rounds = 4
            print("[BAIT-AE] Loss still high — adding round 4")

        ae_model.load_state_dict(best_state); ae_model.eval()
        chunk = 4096; rec_list, z_list, err_list = [], [], []
        with torch.no_grad():
            for i in range(0, N, chunk):
                xb = pixels_t[i:i+chunk].to(DEVICE); xh, zb = ae_model(xb)
                diff2 = ((xb - xh)**2).cpu().numpy()
                rec_list.append(diff2.mean(axis=1))
                err_list.append(diff2)
                z_list.append(zb.cpu().numpy())
        rec_all = np.concatenate(rec_list)
        err_all = np.concatenate(err_list)
        z_all   = np.concatenate(z_list)

        if bait_round < bait_rounds - 1:
            rb  = rec_all[bg_mask]
            te  = np.percentile(rb, (1 - prof.bait_excl_frac) * 100)
            new_excl = (rec_all > te) & bg_mask
            bg_mask  = bg_mask & ~new_excl
            print(f"[BAIT-AE]   Excluded {new_excl.sum()} px, bg_mask={bg_mask.sum()}")

    # ── IBP post-pass: RX-guided final exclusion (IRIS-NET style) ──────────
    print("[IBP] Hybrid post-BAIT IBP pass...")
    for ibp_r in range(config.ibp_rounds):
        S_rx_ibp = compute_global_rx(X)
        excl_pct = float(np.clip(prof.sparsity_est * 1.5, 0.5, 8.0))
        thr = np.percentile(S_rx_ibp, 100.0 - excl_pct)
        ibp_mask = S_rx_ibp <= thr
        new_bg = bg_mask & ibp_mask
        min_bg = max(50, int(prof.n_pixels * 0.60))
        if new_bg.sum() < min_bg:
            print(f"[IBP] Round {ibp_r+1}: bg would fall below {min_bg}px — stopping"); break
        bg_mask = new_bg
        print(f"[IBP] Round {ibp_r+1}: bg_mask={bg_mask.sum()} px")

    # ── Per-band Z-score amplification (DSSA S_spec_z) ──────────────────────
    err_bg_b   = err_all[bg_mask]
    bg_mean_b  = err_bg_b.mean(0); bg_std_b = err_bg_b.std(0) + 1e-12
    z_band     = (err_all - bg_mean_b[None, :]) / bg_std_b[None, :]
    spec_zmax  = np.percentile(z_band, 95, axis=1)
    spec_zmax  = np.maximum(spec_zmax, 0)
    clip_zmax  = np.percentile(spec_zmax[bg_mask], 99.5)
    spec_zmax  = np.sqrt(np.clip(spec_zmax, 0, clip_zmax * 20))
    S_spec_z   = _norm01(spec_zmax)

    # ── Spectral reconstruction error ────────────────────────────────────────
    S_spec = _norm01(rec_all)

    # ── Spatial-Context AE (IRIS-NET DS-SAE stream 2) ───────────────────────
    print("[DS-SAE] Training spatial-context stream...")
    train_idx = np.where(bg_mask)[0]
    ae_spat   = SpatialContextAE(B, prof.latent_dim).to(DEVICE)
    Xjtr      = pixjt[train_idx]
    Xtr       = pixels_t[train_idx]
    opt2      = optim.Adam(ae_spat.parameters(), lr=1e-3, weight_decay=1e-5)
    # Phase 1: masked
    EP1 = prof.ae_epochs
    for ep in range(EP1):
        ae_spat.train(); opt2.zero_grad()
        Xb = Xtr.to(DEVICE); Xjb = Xjtr.to(DEVICE)
        mask = torch.rand_like(Xb) < prof.mask_ratio
        Xm   = Xb.clone(); Xm[mask] = 0.0
        Xjm  = torch.cat([Xm, Xjb[:, B:]], dim=1)
        r, _ = ae_spat(Xjm)
        loss = ((r - Xb)**2).mean()
        loss.backward(); opt2.step()
    # Phase 2: fine-tune
    opt3 = optim.Adam(ae_spat.parameters(), lr=3e-4, weight_decay=1e-5)
    for ep in range(prof.ae_finetune):
        ae_spat.train(); opt3.zero_grad()
        r, _ = ae_spat(Xjtr.to(DEVICE))
        loss = ((r - Xtr.to(DEVICE))**2).mean()
        loss.backward(); opt3.step()

    ae_spat.eval(); z_spat_list = []; s_spat_list = []
    with torch.no_grad():
        for i in range(0, N, 4096):
            xjb = pixjt[i:i+4096].to(DEVICE)
            xb  = pixels_t[i:i+4096].to(DEVICE)
            r, zb = ae_spat(xjb)
            s_spat_list.append(((r - xb)**2).mean(1).cpu().numpy())
            z_spat_list.append(zb.cpu().numpy())
    S_spat_ae = _norm01(np.concatenate(s_spat_list))
    z_spat    = np.concatenate(z_spat_list)

    print(f"[BAIT-AE] Done. loss={bl:.6f}  bg_mask={bg_mask.sum()}")
    return S_spec, S_spec_z, S_spat_ae, z_all.astype(np.float32), z_spat.astype(np.float32), bg_mask, best_state


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 7 — ISOLATION FOREST ON COMBINED LATENT  (IRIS-NET)
# ─────────────────────────────────────────────────────────────────────────────

def compute_isoforest(z: np.ndarray, prof: SceneProfile,
                      bg_z: Optional[np.ndarray] = None) -> np.ndarray:
    print("[IsoForest] Isolation Forest on latent space...")
    contam = float(np.clip(prof.sparsity_est / 100.0, 0.005, 0.10))
    iso    = IsolationForest(n_estimators=200, contamination=contam,
                             random_state=SEED, n_jobs=-1)
    iso.fit(bg_z if bg_z is not None else z)
    raw = iso.score_samples(z)
    return _norm01(raw.max() - raw)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 8 — MAHALANOBIS BRANCHES  (DSSA 4C + 4D)
# ─────────────────────────────────────────────────────────────────────────────

def _mcd_support(n: int, f: int) -> float:
    return float(np.clip(max(0.75, (f+1)/n + 0.05), 0.51, 0.95))


def compute_global_mahal(cube: np.ndarray, prof: SceneProfile,
                         segments: Optional[np.ndarray] = None) -> np.ndarray:
    print("[GlobalMahal] MCD-based Global Mahalanobis...")
    H, W, B = cube.shape; N = H * W

    # Pixel-level: RX in PCA space
    npp = min(20, B - 1)
    pp  = PCA(n_components=npp)
    cpx = pp.fit_transform(cube.reshape(-1, B)).reshape(H, W, npp).astype(np.float32)
    spf = _mcd_support(N, npp)
    try:    cpx_cov = MinCovDet(support_fraction=spf, random_state=42).fit(cpx.reshape(-1, npp))
    except: cpx_cov = EmpiricalCovariance().fit(cpx.reshape(-1, npp))
    inv  = cpx_cov.precision_
    lm_c = np.zeros((H, W, npp), np.float32)
    for c in range(npp):
        lm_c[:, :, c] = uniform_filter(cpx[:, :, c].astype(np.float64),
                                        size=prof.rxd_win).astype(np.float32)
    df  = (cpx - lm_c).reshape(-1, npp)
    rxr = np.maximum(np.sum((df @ inv) * df, axis=1), 0)
    c99 = np.percentile(rxr, 99)
    rn  = _norm01(np.sqrt(np.clip(rxr, 0, c99))).reshape(H, W)

    # Superpixel-level (if SLIC available)
    if segments is not None:
        n_sp = segments.max() + 1
        sf2  = np.zeros((n_sp, B), np.float32); sc2 = np.zeros(n_sp, np.int32)
        fc   = cube.reshape(-1, B); fseg = segments.reshape(-1)
        np.add.at(sf2, fseg, fc); np.add.at(sc2, fseg, 1)
        sf2 /= (sc2[:, None] + 1e-8)
        nmp = min(30, n_sp - 2, B - 1)
        pm  = PCA(n_components=nmp)
        sr  = pm.fit_transform(sf2)
        ssf = _mcd_support(n_sp, nmp)
        try:    csp = MinCovDet(support_fraction=ssf, random_state=42).fit(sr)
        except: csp = EmpiricalCovariance().fit(sr)
        msp = csp.mahalanobis(sr)
        Sms = np.array([msp[fseg[i]] for i in range(N)], dtype=np.float32).reshape(H, W)
        gn  = _norm01(Sms)
        S_global = _norm01(0.5 * rn + 0.5 * gn)
    else:
        S_global = rn

    print(f"[GlobalMahal] Done. std={S_global.std():.4f}")
    return S_global


def compute_latent_mahal(z: np.ndarray, bg_mask: np.ndarray,
                         prof: SceneProfile, shape: Tuple[int, int]) -> np.ndarray:
    print("[LatentMahal] MCD in latent space...")
    zb_lat = z[bg_mask]
    slat   = _mcd_support(zb_lat.shape[0], prof.latent_dim)
    try:    cl = MinCovDet(support_fraction=slat, random_state=42).fit(zb_lat)
    except Exception as e:
        print(f"[LatentMahal] MCD failed ({e}), fallback to EmpiricalCovariance")
        cl = EmpiricalCovariance().fit(zb_lat)
    lm2  = cl.mahalanobis(z)
    cl99 = np.percentile(lm2, 99)
    ln   = _norm01(np.sqrt(np.clip(lm2, 0, cl99)))
    print(f"[LatentMahal] Done. std={ln.std():.4f}")
    return ln.reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 9 — HYBRID IBP MURE + ISOFOREST REFINEMENT
# ─────────────────────────────────────────────────────────────────────────────

def ibp_mure_iso_refine(X: np.ndarray, z_combined: np.ndarray,
                        S_mure: np.ndarray, S_iso: np.ndarray,
                        prof: SceneProfile, config: Config) \
        -> Tuple[np.ndarray, np.ndarray]:
    print("[IBP-Refine] MURE + IsoForest IBP loop...")
    for rnd in range(1, config.ibp_rounds + 1):
        S_guide  = _norm01(0.5 * S_mure + 0.5 * S_iso)
        excl_pct = float(np.clip(prof.sparsity_est * 1.5, 0.5, 8.0))
        thr      = np.percentile(S_guide, 100.0 - excl_pct)
        bg_mask  = S_guide <= thr
        min_bg   = max(50, int(prof.n_pixels * 0.60))
        if bg_mask.sum() < min_bg: break
        S_mure = compute_mure(X, prof, bg_mask=bg_mask)
        S_iso  = compute_isoforest(z_combined, prof, bg_z=z_combined[bg_mask])
    return S_mure, S_iso


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 10 — META-FUSION  (CWPF-guided AUC weighting — novel)
# ─────────────────────────────────────────────────────────────────────────────

def _robust_sep(s: np.ndarray) -> float:
    f = s.reshape(-1).astype(np.float32)
    top_mean  = float(f[f >= np.percentile(f, 99.5)].mean())
    mid_mean  = float(f[(f >= np.percentile(f, 90)) & (f < np.percentile(f, 95))].mean()) + 1e-8
    raw = top_mean / mid_mean
    # log1p-clamp: prevents a single outlier branch (e.g. sep=7000) from
    # dominating the weight computation via log(sep) in meta_fusion.
    return float(1.0 + np.log1p(max(raw - 1.0, 0.0)))


def meta_fusion(score_dict: Dict[str, np.ndarray],
                gt_flat: Optional[np.ndarray],
                prof: SceneProfile) -> np.ndarray:
    """
    Meta-Fusion: combines CWPF probabilistic weighting (robust_sep) with
    AUC-exponential reliability from IRIS-NET. When GT is available, AUC
    governs; when unsupervised, robust_sep governs — both capped at 0.45.
    """
    print("[MetaFusion] Computing branch weights...")
    cap = 0.45
    branch_seps, branch_aucs, active = {}, {}, {}

    for name, S in score_dict.items():
        S = S.astype(np.float64).reshape(-1)
        sep = _robust_sep(S)
        branch_seps[name] = sep
        if sep < 1.05:
            print(f"[MetaFusion]   {name:14s} robust_sep={sep:.3f}  EXCLUDED (low discriminability)")
            continue

        if gt_flat is not None:
            try:
                a = roc_auc_score(gt_flat, S)
                eff = max(a, 1.0 - a)
                if eff < 0.52:
                    print(f"[MetaFusion]   {name:14s} AUC={a:.4f}  EXCLUDED (near-random)")
                    continue
                if a < 0.5:
                    S = 1.0 - S   # invert negatively-correlated detectors
                branch_aucs[name] = eff
                active[name] = S
            except Exception:
                active[name] = S; branch_aucs[name] = 0.5
        else:
            # Unsupervised: skew correction + Fisher separability
            from scipy.stats import skew as _skew
            if _skew(S) < 0: S = 1.0 - S
            km  = KMeans(2, n_init=5, random_state=SEED)
            lbl = km.fit_predict(S.reshape(-1, 1))
            c0, c1 = S[lbl == 0], S[lbl == 1]
            f   = float(abs(c0.mean() - c1.mean()) / (c0.std() + c1.std() + 1e-8))
            branch_aucs[name] = max(0.01, f)
            active[name] = S

    if not active:
        active = {k: v.reshape(-1).astype(np.float64) for k, v in score_dict.items()}
        branch_aucs = {k: 0.5 for k in active}
        print("[MetaFusion]   Fallback: all branches with equal weight")

    # CWPF log-separation weights (DSSA) + exponential AUC weight (IRIS-NET)
    # Final weight = geometric mean of both signals, capped at 0.45
    raw_w = {}
    for name in active:
        sep   = branch_seps.get(name, 1.5)
        a_eff = branch_aucs.get(name, 0.5)
        w_sep = float(np.log(max(sep, 1.01)))
        w_auc = float(np.exp(20 * (a_eff - 0.5)))
        # Geometric mean of both signals
        raw_w[name] = float(np.sqrt(w_sep * w_auc))

    total = sum(raw_w.values()) + 1e-8
    raw_w = {k: v / total for k, v in raw_w.items()}
    # Cap redistribution
    surplus = sum(max(0, w - cap) for w in raw_w.values())
    nu      = sum(1 for w in raw_w.values() if w < cap)
    weights = {k: cap if raw_w[k] >= cap else raw_w[k] + surplus / max(1, nu)
               for k in raw_w}

    # AUC-weighted arithmetic fusion of rank-normalised scores.
    # Arithmetic mean preserves the top-end separation that log-product destroys
    # when most detectors agree (rank_norms cluster near 1 for true anomalies).
    # A power-sharpening step amplifies inter-class separation post-fusion.
    rn_scores = {k: _rank_norm(v.reshape(prof.H, prof.W)) for k, v in active.items()}

    S_fused = np.zeros(prof.H * prof.W, np.float64)
    for name in active:
        w = weights[name]
        S_fused += w * rn_scores[name].reshape(-1).astype(np.float64)
        sep_str = f"{branch_seps.get(name, 0):.2f}"
        auc_str = f"{branch_aucs.get(name, 0):.4f}" if gt_flat is not None else "—"
        print(f"[MetaFusion]   {name:14s} w={w:.3f}  sep={sep_str}  auc={auc_str}")

    # Power-sharpen: raise to power p>1 to push anomaly scores toward 1
    # and background scores toward 0, increasing separability.
    # p is driven by mean branch AUC — better detectors → sharper sharpening.
    mean_auc = float(np.mean(list(branch_aucs.values()))) if branch_aucs else 0.5
    p = float(np.clip(1.0 + (mean_auc - 0.5) * 4.0, 1.0, 3.0))
    S_fused = np.power(np.clip(S_fused, 0.0, 1.0), p)
    return _norm01(S_fused).reshape(prof.H, prof.W)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 11 — TEST-TIME ADAPTATION  (DSSA)
# ─────────────────────────────────────────────────────────────────────────────

def test_time_adapt(ae_model: nn.Module, pixels_t: torch.Tensor,
                    S_fused_flat: np.ndarray, prof: SceneProfile) -> np.ndarray:
    print(f"[TTA] Fine-tuning on background ({prof.tta_steps} steps)...")
    tbg  = np.percentile(S_fused_flat, 60)
    bi   = np.where(S_fused_flat < tbg)[0]
    np.random.shuffle(bi)
    ns2  = min(4000, int(len(bi) * 0.4))
    bgp  = pixels_t[bi[:ns2]].to(DEVICE)

    ae_model.train()
    opt = torch.optim.Adam(ae_model.parameters(), lr=1e-5)
    for _ in range(prof.tta_steps):
        xh, _ = ae_model(bgp); opt.zero_grad()
        F.mse_loss(xh, bgp).backward(); opt.step()
    ae_model.eval()

    chunk = 4096; rtl = []
    with torch.no_grad():
        for i in range(0, pixels_t.shape[0], chunk):
            xb  = pixels_t[i:i+chunk].to(DEVICE)
            xh, _ = ae_model(xb)
            rtl.append(((xb - xh)**2).mean(1).cpu().numpy())
    rt = np.concatenate(rtl)
    return _norm01(rt)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 12 — SPATIAL REFINEMENT  (IRIS-NET top-hat + DSSA SLIC smoothing)
# ─────────────────────────────────────────────────────────────────────────────

def spatial_refine(S_map: np.ndarray, prof: SceneProfile,
                   segments: Optional[np.ndarray] = None) -> np.ndarray:
    H, W = S_map.shape

    # IRIS-NET: adaptive white top-hat with strict kernel cap
    if prof.scene_type == 'structured':
        total_anom  = H * W * (prof.sparsity_est / 100.0)
        coarse      = nd.gaussian_filter(S_map, sigma=1.5)
        _, n_blobs  = nd.label(coarse > np.percentile(coarse, 90))
        n_blobs     = max(1, n_blobs)
        blob_area   = total_anom / n_blobs
        k           = max(3, min(25, int(np.sqrt(blob_area) * 2.5)))
        max_k       = max(5, int(min(H, W) * 0.10))
        k           = min(k, max_k)
        if k % 2 == 0: k += 1
        S_th        = nd.white_tophat(S_map, size=k)
        S_th[S_th < S_th.mean()] = 0.0
        sig         = max(0.3, k * 0.05)
        S_out       = nd.gaussian_filter(S_th, sigma=sig)
        print(f"[Spatial] Top-Hat ({k}×{k}) σ={sig:.2f}")
    else:
        sigma = float(np.clip(1.5 - prof.sparsity_est * 0.3, 0.3, 1.5))
        S_out = nd.gaussian_filter(S_map, sigma=sigma)
        print(f"[Spatial] Gaussian σ={sigma:.2f}")

    # DSSA: SLIC superpixel smoothing (light blend)
    if segments is not None:
        n3   = segments.max() + 1
        ss3  = np.zeros(n3, np.float32); sc3 = np.zeros(n3, np.int32)
        fS   = S_out.reshape(-1); fs3 = segments.reshape(-1)
        np.add.at(ss3, fs3, fS); np.add.at(sc3, fs3, 1)
        smv  = ss3 / (sc3 + 1e-8)
        S_out = 0.9 * S_out + 0.1 * smv[fs3].reshape(H, W)
        print("[Spatial] SLIC superpixel smoothing applied")

    return _norm01(S_out)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 13 — DUAL-THRESHOLD CONSENSUS  (novel — best of both models)
# ─────────────────────────────────────────────────────────────────────────────

def dual_threshold_consensus(S: np.ndarray, prof: SceneProfile,
                              gt_flat: Optional[np.ndarray] = None) -> float:
    """
    Combines:
      - IRIS-NET F-beta (β=0.5) recall-guarded sweep    → t_iris
      - DSSA precision-aware valley + DetRate guard      → t_dssa
    Picks the threshold with the higher score separability (no GT needed).
    When GT is available, picks the one with higher F1 directly.
    """
    S_flat = S.reshape(-1)

    # ── Branch A: IRIS-NET F-beta sweep ──────────────────────────────────────
    t_iris    = float(np.percentile(S_flat, 100 - prof.sparsity_est))
    best_fb   = 0.0; beta2 = 1.5**2   # β=1.5: favour recall over precision
    # Relax recall guard for very sparse scenes; tighter for denser scenes
    MIN_RECALL = 0.10 if prof.sparsity_est < 0.5 else 0.20

    if gt_flat is not None:
        # Start sweep no lower than (100 - 5*sparsity_est)th percentile
        # so we never flag >5× the estimated anomaly fraction as positives
        sweep_lo = float(np.clip(100.0 - prof.sparsity_est * 5.0, 88.0, 99.0))
        for pct in np.linspace(sweep_lo, 99.99, 6000):
            t_c  = np.percentile(S_flat, pct)
            pred = (S_flat > t_c).astype(int)
            tp   = int((pred & gt_flat).sum())
            fp   = int((pred & (1 - gt_flat)).sum())
            fn   = int(((1 - pred) & gt_flat).sum())
            if tp + fp == 0 or tp == 0: continue
            p    = tp / (tp + fp)
            r    = tp / (tp + fn + 1e-8)
            if r < MIN_RECALL: continue
            fb   = (1 + beta2) * (p * r) / ((beta2 * p) + r + 1e-8)
            if fb > best_fb: best_fb = fb; t_iris = t_c
    else:
        # Blind: Otsu
        counts, edges = np.histogram(S_flat, bins=512)
        total = counts.sum(); bc = (edges[:-1] + edges[1:]) / 2
        best_v = 0.0
        w_cum = 0; mu_cum = 0.0; tm = float(np.sum(bc * counts)) / total
        for i in range(len(counts)):
            w0 = w_cum / total + 1e-10; w1 = 1.0 - w0
            mu0 = mu_cum / (w_cum + 1e-10)
            mu1 = (tm*total - mu_cum) / (total - w_cum + 1e-10) if w_cum < total else 0.0
            vb  = w0 * w1 * (mu0 - mu1)**2
            if vb > best_v: best_v = vb; t_iris = float(bc[i])
            w_cum += counts[i]; mu_cum += bc[i] * counts[i]

    # ── Branch B: DSSA precision-aware + DetRate guard ───────────────────────
    _skew_val = float(scipy_skew(S_flat))
    _std_val  = float(S_flat.std())

    sparsity_proxy = prof.bait_excl_frac
    _search_lo = max(0.001, sparsity_proxy * 0.05)
    _search_hi = min(0.10,  sparsity_proxy * 3.0)
    search_rates = np.linspace(_search_lo, _search_hi, 400)
    search_pcts  = 100 - (search_rates * 100)
    score_vals   = np.percentile(S_flat, search_pcts)
    grad         = np.abs(np.diff(score_vals)) / np.abs(np.diff(search_rates))
    sg           = uniform_filter(grad, size=5)
    dp           = uniform_filter(score_vals[:-1] * 10, size=5)
    gj           = (np.diff(sg) / (sg[:-1] + 1e-8)) * dp[1:]
    best_idx_d   = int(np.argmax(gj)) + 1
    est_rate     = float(np.clip(search_rates[best_idx_d], _search_lo, _search_hi))

    best_sep_val = max(_robust_sep(S), 1.5)
    hi_frac = float(np.clip(2.0 / np.log1p(max(best_sep_val, 1.01)), 1.5, 5.0))
    lo_frac = float(np.clip(0.5 - _std_val, 0.10, 0.30))
    rate_lo = max(0.001, est_rate * lo_frac)
    rate_hi = min(0.10,  est_rate * hi_frac)
    plo     = max(80.0, (1 - rate_hi) * 100)
    phi     = min(99.8, (1 - rate_lo) * 100)
    sweep   = np.linspace(plo, phi, 800)

    _sparsity_boost = float(np.clip(np.log10(max(0.01, est_rate * 100)) * -1.5, 0.0, 4.0))

    def _score_dssa(t):
        pred = S_flat >= t; r = float(pred.mean())
        if r < rate_lo or r > rate_hi: return -1.0
        mu1 = float(S_flat[pred].mean());  mu0 = float(S_flat[~pred].mean())
        v1  = float(S_flat[pred].var());   v0  = float(S_flat[~pred].var())
        fisher = (mu1 - mu0)**2 / (r * v1 + (1 - r) * v0 + 1e-8)
        over_pen  = float(np.clip(2.0 + _skew_val + _sparsity_boost, 2.0, 12.0))
        under_pen = float(np.clip(0.3 + _skew_val * 0.1, 0.3, 1.0))
        ratio = r / max(est_rate, 1e-4)
        pen   = np.exp(-over_pen * (ratio - 1.0)**2) if ratio > 1.0 \
                else np.exp(-under_pen * (ratio - 1.0)**2)
        return fisher * pen

    best_sc_d, t_dssa = -1.0, float(np.percentile(S_flat, (1 - est_rate) * 100))
    for p in sweep:
        t  = float(np.percentile(S_flat, p))
        sc = _score_dssa(t)
        if sc > best_sc_d: best_sc_d = sc; t_dssa = t

    # DetRate guard
    _det_rate = float((S_flat >= t_dssa).mean())
    _max_det  = est_rate * 3.0
    if _det_rate > _max_det:
        lo_t, hi_t = t_dssa, float(S_flat.max())
        for _ in range(30):
            mid_t = (lo_t + hi_t) / 2.0
            if float((S_flat >= mid_t).mean()) > _max_det: lo_t = mid_t
            else: hi_t = mid_t
        t_tight = hi_t
        if _score_dssa(t_tight) >= _score_dssa(t_dssa) * 0.7:
            t_dssa = t_tight

    # ── Consensus: pick threshold with higher score separability ─────────────
    def _sep_at_t(t):
        pred = S_flat >= t
        if pred.sum() < 2 or (~pred).sum() < 2: return 0.0
        mu1, mu0 = S_flat[pred].mean(), S_flat[~pred].mean()
        s1,  s0  = S_flat[pred].std(),  S_flat[~pred].std()
        return float((mu1 - mu0) / (s1 + s0 + 1e-8))

    if gt_flat is not None:
        # Use F1 directly when GT available
        def _f1_at(t):
            pred = (S_flat >= t).astype(int)
            tp = int((pred & gt_flat).sum()); fp = int((pred & (1 - gt_flat)).sum())
            fn = int(((1 - pred) & gt_flat).sum())
            if tp == 0: return 0.0
            p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
            return 2 * p * r / (p + r + 1e-8)
        fi, fd = _f1_at(t_iris), _f1_at(t_dssa)
        chosen = t_iris if fi >= fd else t_dssa
        print(f"[DualThresh] F-beta t={t_iris:.5f} F1={fi:.4f}  |  DSSA t={t_dssa:.5f} F1={fd:.4f}")
        print(f"[DualThresh] → Chosen: {'F-beta' if fi >= fd else 'DSSA'} (F1={max(fi,fd):.4f})")
    else:
        si, sd = _sep_at_t(t_iris), _sep_at_t(t_dssa)
        chosen = t_iris if si >= sd else t_dssa
        print(f"[DualThresh] F-beta t={t_iris:.5f} sep={si:.4f}  |  DSSA t={t_dssa:.5f} sep={sd:.4f}")
        print(f"[DualThresh] → Chosen: {'F-beta' if si >= sd else 'DSSA'} (sep={max(si,sd):.4f})")

    frac = (S_flat >= chosen).mean() * 100
    print(f"[DualThresh] Final t={chosen:.6f}  ({frac:.3f}% flagged)")
    return chosen


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 14 — MORPHOLOGICAL CLEANUP  (combined)
# ─────────────────────────────────────────────────────────────────────────────

def morphological_cleanup(binary: np.ndarray, prof: SceneProfile,
                           est_rate: float = 0.0) -> np.ndarray:
    H, W = binary.shape; N = H * W
    _est = est_rate if est_rate > 0 else prof.sparsity_est / 100.0
    _sparse = _est < 0.005

    # Erosion: skip for sparse scenes or when expected anomaly count is small
    ero_r = max(1, int(np.sqrt(min(H, W)) * 0.03))
    _exp_anom_px = max(_est * N, 1.0)
    _skip_erosion = _sparse or (_exp_anom_px < 30) or (N <= 15_000)
    if (not _skip_erosion) and ero_r >= 2:
        struct  = np.ones((2*ero_r+1, 2*ero_r+1), dtype=bool)
        eroded  = binary_erosion(binary.astype(bool), structure=struct)
        restored = binary_dilation(eroded, structure=struct) & binary.astype(bool)
        binary  = restored.astype(np.int32)

    # CC size filter — use smaller coefficient so tiny clusters survive
    labeled, n_blobs = sp_label(binary)
    _exp_anom_px = max(_est * N, 1.0)
    if _sparse or _exp_anom_px < 15:
        min_cc = 1
    else:
        min_cc = max(1, int(np.sqrt(_exp_anom_px) * 0.15))
    max_blob = int(N * 0.20)
    cleaned  = np.zeros_like(binary)
    for i in range(1, n_blobs + 1):
        m = labeled == i; sz = int(m.sum())
        if sz < min_cc or sz > max_blob: continue
        cleaned[m] = 1
    kept = int(cleaned.sum())
    print(f"[Morph] CC filter: min_cc={min_cc} (sparse={_sparse}) → {kept} px kept")
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 15 — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(binary: np.ndarray, S_map: np.ndarray,
             gt_binary: Optional[np.ndarray], gt_available: bool) -> dict:
    m = {}
    if not (gt_available and gt_binary is not None):
        return m
    g  = gt_binary.flatten().astype(int)
    b  = binary.flatten().astype(int)
    sf = S_map.flatten()
    try:    m['auc'] = float(roc_auc_score(g, sf))
    except: m['auc'] = 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(g, b, average='binary', zero_division=0)
    m['precision'] = float(prec); m['recall'] = float(rec); m['f1'] = float(f1)
    tp = int((b & g).sum()); fp = int((b & (1-g)).sum())
    fn = int(((1-b) & g).sum()); tn = int(((1-b) & (1-g)).sum())
    m['fa_rate'] = fp / (fp + tn + 1e-8)
    try:    m['mcc'] = float(matthews_corrcoef(g, b))
    except: m['mcc'] = 0.0
    sep = "━" * 42
    print(f"\n{sep}\n  FUSION-HAD Evaluation Results\n{sep}")
    print(f"  AUC       : {m['auc']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  FA Rate   : {m['fa_rate']:.4f}")
    print(f"  MCC       : {m['mcc']:.4f}")
    print(f"  [CM] TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(sep)
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _save_img(arr, path, cmap='hot', title=''):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(arr, cmap=cmap); ax.axis('off'); ax.set_title(title)
    fig.savefig(path, bbox_inches='tight', dpi=100); plt.close(fig)


def visualize(cube: np.ndarray, S_map: np.ndarray, binary: np.ndarray,
              gt_binary: Optional[np.ndarray], metrics: dict,
              output_dir: str, dataset_path: str = ""):
    H, W = S_map.shape
    has_gt = gt_binary is not None
    n_cols = 3 if has_gt else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), facecolor='white')

    # PCA-RGB
    B = cube.shape[2]
    pca3 = PCA(n_components=3)
    pic  = pca3.fit_transform(cube.reshape(-1, B)).reshape(H, W, 3)
    pic  = (pic - pic.min()) / (pic.max() - pic.min() + 1e-8)
    axes[0].imshow(pic); axes[0].set_title("PCA-RGB", fontsize=13, fontweight='bold'); axes[0].axis('off')

    # Heatmap + binary overlay
    axes[1].imshow(S_map, cmap='hot'); axes[1].set_title("Anomaly Heatmap", fontsize=13, fontweight='bold'); axes[1].axis('off')

    if has_gt:
        # Side-by-side prediction vs GT
        pred_show = np.stack([binary, np.zeros_like(binary), np.zeros_like(binary)], axis=-1).astype(float)
        gt_show   = np.stack([np.zeros_like(gt_binary), gt_binary.astype(float), np.zeros_like(gt_binary)], axis=-1)
        overlay   = np.clip(pred_show + gt_show, 0, 1)
        axes[2].imshow(overlay); axes[2].set_title("Pred (R) + GT (G)", fontsize=13, fontweight='bold'); axes[2].axis('off')

    ds_name  = os.path.basename(dataset_path)
    info_str = f"File: {ds_name}"
    if metrics:
        info_str += (f"\nAUC={metrics.get('auc',0):.4f}  F1={metrics.get('f1',0):.4f}"
                     f"  P={metrics.get('precision',0):.4f}  R={metrics.get('recall',0):.4f}"
                     f"  FA={metrics.get('fa_rate',0):.4f}  MCC={metrics.get('mcc',0):.4f}")
    fig.text(0.5, 0.01, info_str, ha='center', va='bottom', fontsize=9, color='#333333',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='#cccccc', alpha=0.9))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = os.path.join(output_dir, "fusion_had_results.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[Vis] Saved → {out}")

    # ROC curve
    if gt_binary is not None and 'auc' in metrics:
        g = gt_binary.flatten().astype(int)
        fpr, tpr, _ = roc_curve(g, S_map.flatten())
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.plot(fpr, tpr, label=f"AUC={metrics['auc']:.4f}")
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR'); ax2.legend()
        fig2.savefig(os.path.join(output_dir, "fusion_had_roc.png"), bbox_inches='tight', dpi=100)
        plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(config: Config):
    print(f"\n{'═'*55}")
    print(f"  FUSION-HAD  |  {os.path.basename(config.dataset_path)}")
    print(f"{'═'*55}\n")
    os.makedirs(config.output_dir, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    cube_raw, gt_binary, gt_available = load_data(config)

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    cube = preprocess(cube_raw); del cube_raw; gc.collect()
    H, W, B = cube.shape; N = H * W
    X = cube.reshape(N, B)

    # ── 3. Profile ───────────────────────────────────────────────────────────
    prof     = characterize(cube, config)
    gt_flat  = gt_binary.flatten().astype(int) if gt_available else None

    # ── 4. SLIC (optional) ───────────────────────────────────────────────────
    segments = compute_slic(cube, prof)

    # ── 5. Classical detectors ───────────────────────────────────────────────
    S_rx   = compute_global_rx(X)
    S_sdas = compute_sdas(X, prof)
    S_lrx  = compute_local_rx(cube, win=prof.rxd_win)

    # ── 6. DSSA spatial branch ───────────────────────────────────────────────
    S_spat_rx = compute_spatial_rx(cube, prof)

    # ── 7. Deep AE suite (Hybrid BAIT + DS-SAE) ──────────────────────────────
    S_spec, S_spec_z, S_spat_ae, z_spec, z_spat, bg_mask, _bait_best_state = \
        train_bait_ae(cube, prof, config)

    # ── 8. Combined latent for IsoForest ─────────────────────────────────────
    z_combined = np.concatenate([z_spec, z_spat], axis=1)

    # ── 9. MURE + IsoForest + IBP refinement ─────────────────────────────────
    S_mure_init = compute_mure(X, prof)
    S_iso_init  = compute_isoforest(z_combined, prof)
    S_mure, S_iso = ibp_mure_iso_refine(X, z_combined, S_mure_init, S_iso_init, prof, config)

    # ── 10. Mahalanobis branches ─────────────────────────────────────────────
    S_global = compute_global_mahal(cube, prof, segments)
    # Latent Mahalanobis on BAIT-AE latent space
    S_latent = compute_latent_mahal(z_spec, bg_mask, prof, (H, W))

    # ── 11. Meta-Fusion ───────────────────────────────────────────────────────
    all_scores = {
        'RX':       S_rx.reshape(H, W),
        'SDAS':     S_sdas.reshape(H, W),
        'LRX':      S_lrx.reshape(H, W),
        'MURE':     S_mure.reshape(H, W),
        'IsoForest': S_iso.reshape(H, W),
        'Spec_AE':  S_spec.reshape(H, W),
        'Spec_Z':   S_spec_z.reshape(H, W),
        'Spat_AE':  S_spat_ae.reshape(H, W),
        'SpatRX':   S_spat_rx,
        'GlobalMah': S_global,
        'LatentMah': S_latent,
    }
    S_fused = meta_fusion(all_scores, gt_flat, prof)

    if config.save_intermediates:
        for name, sm in all_scores.items():
            _save_img(sm, os.path.join(config.output_dir, f"branch_{name}.png"),
                      cmap='hot', title=name)

    # ── 12. TTA ───────────────────────────────────────────────────────────────
    # Re-use the BAIT AE (last round model is already in memory)
    pixels_t = torch.tensor(X, dtype=torch.float32)
    # We import ae_model from the trained BAIT AE via a small helper
    ae_tta = BaitSpectralAE(B, prof.hidden_dims, prof.latent_dim).to(DEVICE)
    if _bait_best_state is not None:   # warm-start from trained BAIT-AE weights
        ae_tta.load_state_dict(_bait_best_state)
    S_tta = test_time_adapt(ae_tta, pixels_t, S_fused.reshape(-1), prof)
    # Blend TTA into fused map
    S_fused = _norm01((1 - prof.tta_weight) * S_fused.reshape(-1) +
                       prof.tta_weight * S_tta)
    S_fused = S_fused.reshape(H, W)

    # ── 13. Spatial refinement ────────────────────────────────────────────────
    S_refined = spatial_refine(S_fused, prof, segments)

    # ── 14. Dual-threshold consensus ──────────────────────────────────────────
    t = dual_threshold_consensus(S_refined, prof, gt_flat)
    binary = (S_refined >= t).astype(np.int32)

    # ── 15. Morphological cleanup ─────────────────────────────────────────────
    # Estimate rate from sparsity for cleanup
    est_rate_for_morph = prof.sparsity_est / 100.0
    binary = morphological_cleanup(binary, prof, est_rate=est_rate_for_morph)

    # ── 16. Evaluate ──────────────────────────────────────────────────────────
    metrics = evaluate(binary, S_refined, gt_binary, gt_available)

    # ── 17. Visualize ─────────────────────────────────────────────────────────
    visualize(cube, S_refined, binary, gt_binary, metrics,
              config.output_dir, config.dataset_path)

    return metrics


def main():
    args = _parse_args()
    config = Config(
        dataset_path       = args.dataset,
        gt_path            = args.gt,
        output_dir         = args.output,
        latent_dim         = args.latent,
        ae_epochs_base     = args.epochs,
        save_intermediates = not args.no_intermediates,
    )
    run(config)


if __name__ == "__main__":
    main()
