"""The CNN14 network of PANNs, for inference only.

Copied from the official repository of PANNs (Kong et al., 2020),
https://github.com/qiuqiangkong/audioset_tagging_cnn, file pytorch/models.py,
classes ConvBlock and Cnn14, MIT License, Copyright (c) 2018-2020 Qiuqiang Kong
(full text in PANNS_LICENSE.txt). Only the parts used for training were removed:
SpecAugment and mixup have no parameters, so the module names, and therefore
the keys of the published checkpoint, are unchanged.

Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., Plumbley, M. D. (2020).
PANNs: Large-scale pretrained audio neural networks for audio pattern
recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing,
28, 2880-2894.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14(nn.Module):
    """CNN14, 32 kHz version: log-mel spectrogram, six convolutional blocks, a 2048-unit layer, 527 AudioSet outputs."""

    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                 fmin=50, fmax=14000, classes_num=527):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size, win_length=window_size, window="hann",
            center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax,
            ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

    def forward(self, waveform):
        """waveform: (batch, samples) at 32 kHz. Returns the 2048-number embedding and the 527 tag probabilities."""
        x = self.spectrogram_extractor(waveform)      # (batch, 1, time, freq)
        x = self.logmel_extractor(x)                  # (batch, 1, time, 64 mel bands)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        for block in (self.conv_block1, self.conv_block2, self.conv_block3,
                      self.conv_block4, self.conv_block5):
            x = block(x, pool_size=(2, 2))
        x = self.conv_block6(x, pool_size=(1, 1))
        x = torch.mean(x, dim=3)                      # average over frequency
        x1, _ = torch.max(x, dim=2)                   # max over time
        x2 = torch.mean(x, dim=2)                     # mean over time
        x = x1 + x2
        embedding = F.relu_(self.fc1(x))              # the penultimate layer: 2048 numbers
        tags = torch.sigmoid(self.fc_audioset(embedding))
        return embedding, tags
