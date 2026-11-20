"""NumPy scoring for the deployed autoencoder (ONNX Runtime forward pass + loss computed here).

Matches driftguard.models.autoencoder.window_loss: MSE + 0.5 * mean |log|STFT x| - log|STFT x_hat||, with
torch.stft defaults (periodic Hann 256, hop 64, center=True with reflect padding, one-sided).
"""
from __future__ import annotations

import numpy as np

N_FFT, HOP = 256, 64
_HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)  # periodic


def stft_logmag(x: np.ndarray) -> np.ndarray:
    """x: (B, C, T) -> log|STFT| of shape (B, C, F, frames)."""
    pad = N_FFT // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
    n_frames = 1 + (xp.shape[-1] - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = xp[..., idx] * _HANN  # (B, C, frames, N_FFT)
    spec = np.fft.rfft(frames, axis=-1)
    return np.log(np.abs(spec) + 1e-4).transpose(0, 1, 3, 2)


def window_loss(x: np.ndarray, xh: np.ndarray, spec_weight: float = 0.5) -> np.ndarray:
    mse = ((x - xh) ** 2).mean(axis=(1, 2))
    sd = np.abs(stft_logmag(x) - stft_logmag(xh)).reshape(len(x), -1).mean(axis=1)
    return mse + spec_weight * sd


class OnnxScorer:
    def __init__(self, onnx_path: str, mu, sd, spec_weight: float = 0.5, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.mu = np.asarray(mu, np.float32)
        self.sd = np.asarray(sd, np.float32)
        self.spec_weight = spec_weight

    def score(self, windows: np.ndarray) -> np.ndarray:
        x = ((windows - self.mu) / self.sd).astype(np.float32).transpose(0, 2, 1)
        xh = self.sess.run(None, {self.inp: x})[0]
        return window_loss(x, xh, self.spec_weight)
