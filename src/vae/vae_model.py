"""
Lesion-Conditioned VAE Model and Training
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# VAE MODEL
# ============================================================================

class LesionConditionedVAE(nn.Module):
    """Dual-pathway VAE: microstructure + lesion context"""

    def __init__(self, seq_len=100, micro_ch=13, lesion_ch=3, latent=10):
        super().__init__()
        self.sequence_length = seq_len
        self.micro_ch = micro_ch
        self.lesion_ch = lesion_ch
        self.latent = latent

        # Microstructure encoder
        self.micro_c1 = nn.Conv1d(micro_ch, 64, 5, padding=2)
        self.micro_b1 = nn.BatchNorm1d(64)
        self.micro_p1 = nn.AvgPool1d(2)

        self.micro_c2 = nn.Conv1d(64, 128, 5, padding=2)
        self.micro_b2 = nn.BatchNorm1d(128)
        self.micro_p2 = nn.AvgPool1d(2)

        self.micro_c3 = nn.Conv1d(128, 128, 5, padding=2)
        self.micro_b3 = nn.BatchNorm1d(128)
        self.micro_p3 = nn.AvgPool1d(2)

        micro_out = 128 * (seq_len // 8)

        # Lesion context encoder
        self.lesion_c1 = nn.Conv1d(lesion_ch, 32, 5, padding=2)
        self.lesion_b1 = nn.BatchNorm1d(32)
        self.lesion_p1 = nn.AvgPool1d(2)

        self.lesion_c2 = nn.Conv1d(32, 64, 5, padding=2)
        self.lesion_b2 = nn.BatchNorm1d(64)
        self.lesion_p2 = nn.AvgPool1d(2)

        lesion_out = 64 * (seq_len // 4)

        # Combined latent space
        combined_dim = micro_out + lesion_out
        self.fc_mu = nn.Linear(combined_dim, latent)
        self.fc_logv = nn.Linear(combined_dim, latent)

        # Decoder
        self.fc_dec = nn.Linear(latent + lesion_out, micro_out)

        self.dec_t1 = nn.ConvTranspose1d(128, 64, 5, padding=2)
        self.dec_b1 = nn.BatchNorm1d(64)
        self.dec_u1 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

        self.dec_t2 = nn.ConvTranspose1d(64, 64, 5, padding=2)
        self.dec_b2 = nn.BatchNorm1d(64)
        self.dec_u2 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

        self.dec_t3 = nn.ConvTranspose1d(64, micro_ch, 5, padding=2)
        self.dec_u3 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

    def encode_microstructure(self, x_micro):
        x = x_micro.transpose(1, 2)
        h = self.micro_p1(F.relu(self.micro_b1(self.micro_c1(x))))
        h = self.micro_p2(F.relu(self.micro_b2(self.micro_c2(h))))
        h = self.micro_p3(F.relu(self.micro_b3(self.micro_c3(h))))
        h = h.reshape(h.size(0), -1)
        return h

    def encode_lesion(self, x_lesion):
        x = x_lesion.transpose(1, 2)
        h = self.lesion_p1(F.relu(self.lesion_b1(self.lesion_c1(x))))
        h = self.lesion_p2(F.relu(self.lesion_b2(self.lesion_c2(h))))
        h = h.reshape(h.size(0), -1)
        return h

    def encode(self, x_micro, x_lesion):
        h_micro = self.encode_microstructure(x_micro)
        h_lesion = self.encode_lesion(x_lesion)

        h = torch.cat([h_micro, h_lesion], dim=1)

        mu = self.fc_mu(h)
        logv = self.fc_logv(h)

        return mu, logv, h_lesion

    def reparam(self, mu, logv):
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, h_lesion):
        h = torch.cat([z, h_lesion], dim=1)
        h = self.fc_dec(h)
        h = h.view(h.size(0), 128, -1)

        h = self.dec_u1(F.relu(self.dec_b1(self.dec_t1(h))))
        h = self.dec_u2(F.relu(self.dec_b2(self.dec_t2(h))))
        h = self.dec_u3(self.dec_t3(h))

        if h.size(2) != self.sequence_length:
            h = F.interpolate(h, size=self.sequence_length, mode="linear", align_corners=False)

        return h.transpose(1, 2)

    def forward(self, x_micro, x_lesion):
        mu, logv, h_lesion = self.encode(x_micro, x_lesion)
        z = self.reparam(mu, logv)
        xh = self.decode(z, h_lesion)
        return xh, mu, logv


def elbo(xh, x, mu, logv, beta=1.0):
    """Evidence Lower Bound loss"""
    recon = F.mse_loss(xh, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
    return recon + beta * kld, recon, kld


# ============================================================================
# TRAINING
# ============================================================================

def beta_schedule(epoch, total):
    """Annealing schedule for KLD weight"""
    return 0.1 + 1.9 * (epoch / (total - 1)) if total > 1 else 1.0


def train_lesion_vae(X_micro, X_lesion, latent_dim=10, epochs=40,
                     batch_size=64, lr=2e-4, device=None):
    """
    Train lesion-conditioned VAE

    Args:
        X_micro: (n_samples, seq_len, n_micro_features) normalized microstructure
        X_lesion: (n_samples, seq_len, n_lesion_features) lesion data
        latent_dim: Latent space dimensionality
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: torch device

    Returns:
        model: Trained VAE model
        hist: Training history DataFrame
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LesionConditionedVAE(
        seq_len=X_micro.shape[1],
        micro_ch=X_micro.shape[2],
        lesion_ch=X_lesion.shape[2],
        latent=latent_dim
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    ds = TensorDataset(
        torch.tensor(X_micro, dtype=torch.float32),
        torch.tensor(X_lesion, dtype=torch.float32)
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        tot = recon = kld = 0.0
        b = beta_schedule(ep - 1, epochs)
        seen = 0

        for xb_micro, xb_lesion in dl:
            xb_micro = torch.nan_to_num(xb_micro, nan=0.0).to(device)
            xb_lesion = torch.nan_to_num(xb_lesion, nan=0.0).to(device)

            opt.zero_grad()
            xh, mu, logv = model(xb_micro, xb_lesion)
            xh = torch.nan_to_num(xh, nan=0.0)
            mu = torch.nan_to_num(mu, nan=0.0)
            logv = torch.nan_to_num(logv, nan=0.0)

            loss, r, k = elbo(xh, xb_micro, mu, logv, beta=b)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            n = len(xb_micro)
            tot += loss.item() * n
            recon += r.item() * n
            kld += k.item() * n
            seen += n

        if seen > 0:
            avg_loss = tot / seen
            avg_recon = recon / seen
            avg_kld = kld / seen
        else:
            avg_loss = avg_recon = avg_kld = float("nan")

        if ep in {1, 10, 20, 30, 40}:
            print(f"[{ep:02d}/{epochs}] loss={avg_loss:.3f} | " +
                  f"recon={avg_recon:.3f} | kld={avg_kld:.3f} | beta={b:.2f}")

        hist.append((avg_loss, avg_recon, avg_kld, b))

    model.eval()
    return model, pd.DataFrame(hist, columns=["loss", "recon", "kld", "beta"])


# ============================================================================
# Z-SCORE RESIDUAL COMPUTATION
# ============================================================================

def compute_normative_statistics(model, X_micro_sham, X_lesion_sham, batch=256, device=None):
    """
    Compute normative reconstruction statistics from Sham subjects

    Args:
        model: Trained VAE model
        X_micro_sham: Sham microstructure data
        X_lesion_sham: Sham lesion data
        batch: Batch size for inference
        device: torch device

    Returns:
        mean_recon: (100, n_features) - mean reconstruction across Sham
        std_recon: (100, n_features) - std of reconstruction across Sham
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Computing normative statistics from Sham subjects...")

    reconstructions = []

    with torch.no_grad():
        for i in range(0, len(X_micro_sham), batch):
            xb_micro = torch.tensor(X_micro_sham[i:i+batch], dtype=torch.float32, device=device)
            xb_lesion = torch.tensor(X_lesion_sham[i:i+batch], dtype=torch.float32, device=device)

            xb_micro = torch.nan_to_num(xb_micro, nan=0.0)
            xb_lesion = torch.nan_to_num(xb_lesion, nan=0.0)

            xh, _, _ = model(xb_micro, xb_lesion)
            xh = torch.nan_to_num(xh, nan=0.0)

            reconstructions.append(xh.cpu().numpy())

    reconstructions = np.vstack(reconstructions)  # (n_sham_streamlines, 100, n_features)

    # Compute statistics across streamlines for each position and feature
    mean_recon = reconstructions.mean(axis=0)  # (100, n_features)
    std_recon = reconstructions.std(axis=0)    # (100, n_features)
    std_recon = np.maximum(std_recon, 1e-6)    # Avoid division by zero

    print(f"  Mean reconstruction shape: {mean_recon.shape}")
    print(f"  Std reconstruction shape: {std_recon.shape}")

    return mean_recon, std_recon


def compute_zscore_residuals(model, X_micro, X_lesion, mean_recon, std_recon, batch=256, device=None):
    """
    Compute z-score residuals for all subjects

    Z-score residual = (observed - reconstructed - normative_mean) / normative_std

    Args:
        model: Trained VAE model
        X_micro: Microstructure data
        X_lesion: Lesion data
        mean_recon: Normative mean reconstruction
        std_recon: Normative std of reconstruction
        batch: Batch size for inference
        device: torch device

    Returns:
        Z_residuals: (n_streamlines, 100, n_features) - z-scored residuals
        Z_magnitude: (n_streamlines,) - magnitude of deviation per streamline
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Z_residuals_list = []
    Z_magnitude_list = []

    with torch.no_grad():
        for i in range(0, len(X_micro), batch):
            xb_micro = torch.tensor(X_micro[i:i+batch], dtype=torch.float32, device=device)
            xb_lesion = torch.tensor(X_lesion[i:i+batch], dtype=torch.float32, device=device)

            xb_micro = torch.nan_to_num(xb_micro, nan=0.0)
            xb_lesion = torch.nan_to_num(xb_lesion, nan=0.0)

            # Get reconstruction
            xh, _, _ = model(xb_micro, xb_lesion)
            xh = torch.nan_to_num(xh, nan=0.0)

            # Convert to numpy
            observed = xb_micro.cpu().numpy()
            reconstructed = xh.cpu().numpy()

            # Compute residuals
            residuals = observed - reconstructed

            # Z-score relative to normative Sham
            z_residuals = (residuals - mean_recon[None, :, :]) / std_recon[None, :, :]
            z_residuals = np.nan_to_num(z_residuals, nan=0.0, posinf=10.0, neginf=-10.0)

            # Magnitude: RMS of z-scores across positions and features
            z_magnitude = np.sqrt(np.mean(z_residuals**2, axis=(1, 2)))

            Z_residuals_list.append(z_residuals)
            Z_magnitude_list.append(z_magnitude)

    Z_residuals = np.vstack(Z_residuals_list)
    Z_magnitude = np.concatenate(Z_magnitude_list)

    return Z_residuals, Z_magnitude
