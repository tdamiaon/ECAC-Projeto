"""Audio: PANNs CNN14, trained to tag the 527 sound classes of AudioSet.

What this extractor does to a recording (the facts needed for Section 10.4):
  1. decodes the file and averages its channels into one (mono);
  2. resamples it to 32 000 samples per second, the rate CNN14 was trained at;
  3. cuts it into windows of 10 s, the length of the AudioSet clips (a 30 s excerpt gives 3;
     if the length is not a multiple of 10 s, the last window is aligned with the end, and
     a recording shorter than 10 s is one window);
  4. passes each window through CNN14 (log-mel spectrogram with 64 bands between 50 Hz
     and 14 kHz, then the network) and keeps the 2048 activations of its penultimate layer;
  5. summarises the windows of the song by their mean (and, on request, their standard deviation).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ._common import RANDOM_WEIGHTS, download, models_dir, resolve_device, sha256

SAMPLE_RATE = 32000
WINDOW_SECONDS = 10.0
DIMENSION = 2048
CHECKPOINT_NAME = "Cnn14_mAP=0.431.pth"
CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
CHECKPOINT_SHA256 = None    # fill in with the value recorded in models_used.json after the first verified download

CARD = {
    "model": "PANNs CNN14 (32 kHz)",
    "weights": CHECKPOINT_NAME,
    "source": "https://zenodo.org/record/3987831 (code: https://github.com/qiuqiangkong/audioset_tagging_cnn)",
    "reference": "Kong et al. (2020), PANNs: Large-scale pretrained audio neural networks for audio pattern recognition",
    "trained_on": "AudioSet: about 2 million 10-second clips from YouTube, each tagged with some of 527 sound classes",
    "output": "2048 activations of the penultimate layer (fc1), per 10-second window",
}


def load_audio(path, sample_rate: int = SAMPLE_RATE):
    """Decoded, mono, resampled waveform (float32), and what the file was before."""
    import soundfile as sf
    import soxr
    try:
        data, file_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:                       # formats libsndfile cannot read: librosa falls back on audioread/ffmpeg
        import librosa
        y, file_rate = librosa.load(str(path), sr=None, mono=False)
        data = np.atleast_2d(y).T.astype(np.float32)
    channels = data.shape[1]
    y = data.mean(axis=1)
    if file_rate != sample_rate:
        y = soxr.resample(y, file_rate, sample_rate, quality="HQ")
    info = {"file_sample_rate": int(file_rate), "file_channels": int(channels),
            "seconds": round(len(y) / sample_rate, 3)}
    return np.ascontiguousarray(y, dtype=np.float32), info


def split_windows(y: np.ndarray, sample_rate: int = SAMPLE_RATE, seconds: float = WINDOW_SECONDS) -> np.ndarray:
    n = int(round(seconds * sample_rate))
    if len(y) <= n:
        return y[None, :]
    starts = list(range(0, len(y) - n + 1, n))
    if starts[-1] + n < len(y):
        starts.append(len(y) - n)
    return np.stack([y[s:s + n] for s in starts])


def audioset_labels() -> list[str]:
    with open(Path(__file__).with_name("audioset_labels.csv"), newline="", encoding="utf-8") as f:
        return [row["display_name"] for row in csv.DictReader(f)]


class AudioEmbedder:
    """embedder = AudioEmbedder(); vector = embedder("song.mp3")  ->  numpy array of 2048 numbers."""

    dimension = DIMENSION

    def __init__(self, device: str = "auto", models_folder=None, window_seconds: float = WINDOW_SECONDS):
        import torch
        from .cnn14 import Cnn14
        self.device = resolve_device(device)
        self.window_seconds = window_seconds
        self.model = Cnn14()
        self.checkpoint = None
        if not RANDOM_WEIGHTS:
            path = download(CHECKPOINT_URL, models_dir(models_folder) / CHECKPOINT_NAME, CHECKPOINT_SHA256)
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:               # the published file also holds training bookkeeping
                state = torch.load(path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state["model"] if "model" in state else state)
            self.checkpoint = {"file": str(path), "sha256": sha256(path)}
        self.model.eval().to(self.device)
        self.labels = audioset_labels()

    def windows(self, waveform: np.ndarray):
        """Embeddings (n_windows, 2048) and tag probabilities (n_windows, 527) of a 32 kHz mono waveform."""
        import torch
        x = torch.from_numpy(split_windows(waveform, SAMPLE_RATE, self.window_seconds)).to(self.device)
        with torch.inference_mode():
            embedding, tags = self.model(x)
        return embedding.cpu().numpy(), tags.cpu().numpy()

    def embed_file(self, path, std: bool = False, tags: bool = False) -> dict:
        waveform, info = load_audio(path)
        emb, tag = self.windows(waveform)
        out = {"mean": emb.mean(axis=0), "info": dict(info, windows=len(emb))}
        if std:
            out["std"] = emb.std(axis=0)
        if tags:
            out["tags"] = tag.mean(axis=0)
        return out

    def __call__(self, path) -> np.ndarray:
        return self.embed_file(path)["mean"]

    def top_tags(self, path_or_waveform, k: int = 5) -> list[tuple[str, float]]:
        """The k AudioSet classes the network finds most likely: a sanity check, and part of the answer to 10.3."""
        if isinstance(path_or_waveform, np.ndarray):
            _, tag = self.windows(path_or_waveform)
        else:
            _, tag = self.windows(load_audio(path_or_waveform)[0])
        p = tag.mean(axis=0)
        return [(self.labels[i], float(p[i])) for i in np.argsort(-p)[:k]]

    def card(self) -> dict:
        return dict(CARD, checkpoint=self.checkpoint, sample_rate=SAMPLE_RATE,
                    window_seconds=self.window_seconds, mel_bands="64, from 50 Hz to 14 kHz",
                    summary="mean over the windows of the song")
