"""Compact 1D-CNN autoencoder for 1 s, 3-axis vibration windows (about 150k parameters).

Loss = MSE(x, x_hat) + 0.5 * L1(log|STFT x|, log|STFT x_hat|). Trained on healthy windows only.
Inputs are scaled per axis with statistics from the training files of each fold.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _block(cin, cout):
    return nn.Sequential(nn.Conv1d(cin, cout, 9, stride=2, padding=4), nn.BatchNorm1d(cout), nn.GELU())


def _up(cin, cout, last=False):
    layers = [nn.ConvTranspose1d(cin, cout, 9, stride=2, padding=4, output_padding=1)]
    if not last:
        layers += [nn.BatchNorm1d(cout), nn.GELU()]
    return nn.Sequential(*layers)


class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 64), nn.Conv1d(64, 32, 9, stride=2, padding=4))
        self.dec = nn.Sequential(_up(32, 64), _up(64, 64), _up(64, 32), _up(32, 3, last=True))

    def forward(self, x):  # x: (B, 3, 2000)
        return self.dec(self.enc(x))


def stft_logmag(x: torch.Tensor) -> torch.Tensor:
    b, c, t = x.shape
    win = torch.hann_window(256, device=x.device)
    s = torch.stft(x.reshape(b * c, t), n_fft=256, hop_length=64, window=win, return_complex=True)
    return torch.log(s.abs() + 1e-4)


def window_loss(x: torch.Tensor, xh: torch.Tensor, spec_weight: float = 0.5) -> torch.Tensor:
    """Per-window loss (B,) used both for training (mean) and as the anomaly score."""
    mse = ((x - xh) ** 2).mean(dim=(1, 2))
    if spec_weight == 0:
        return mse
    b = x.shape[0]
    sd = (stft_logmag(x) - stft_logmag(xh)).abs().reshape(b, -1).mean(dim=1)
    return mse + spec_weight * sd


class AEScorer:
    name = "AE_conv1d"

    def __init__(self, seed: int = 0, epochs: int = 12, batch: int = 256, lr: float = 1e-3, spec_weight: float = 0.5):
        self.seed, self.epochs, self.batch, self.lr, self.spec_weight = seed, epochs, batch, lr, spec_weight

    def fit(self, train_windows: np.ndarray, val_windows: np.ndarray | None = None, log=print):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = device()
        self.mu = train_windows.mean(axis=(0, 1))
        self.sd = train_windows.std(axis=(0, 1)) + 1e-6
        xt = torch.from_numpy(self._norm(train_windows)).permute(0, 2, 1).contiguous()
        self.model = ConvAE().to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state = float("inf"), None
        for ep in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(xt), generator=g)
            tot = 0.0
            for i in range(0, len(xt), self.batch):
                xb = xt[perm[i:i + self.batch]].to(dev)
                loss = window_loss(xb, self.model(xb), self.spec_weight).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            sched.step()
            msg = f"epoch {ep + 1}/{self.epochs} train {tot / len(xt):.4f}"
            if val_windows is not None and len(val_windows):
                v = float(np.mean(self.score(val_windows)))
                msg += f" val {v:.4f}"
                if v < best:
                    best, best_state = v, {k: t.detach().clone() for k, t in self.model.state_dict().items()}
            log(msg)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _norm(self, w: np.ndarray) -> np.ndarray:
        return ((w - self.mu) / self.sd).astype(np.float32)

    @torch.no_grad()
    def score(self, windows: np.ndarray, batch: int = 2048) -> np.ndarray:
        self.model.eval()
        dev = next(self.model.parameters()).device
        out = []
        for i in range(0, len(windows), batch):
            xb = torch.from_numpy(self._norm(windows[i:i + batch])).permute(0, 2, 1).to(dev)
            out.append(window_loss(xb, self.model(xb), self.spec_weight).float().cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
