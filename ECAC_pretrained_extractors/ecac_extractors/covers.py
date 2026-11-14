"""Covers: ResNet-50 with the ImageNet weights IMAGENET1K_V2 of torchvision, without its classification head.

What this extractor does to a cover (the facts needed for Section 10.4):
  1. opens the image and converts it to RGB (grey or palette images get three equal or
     looked-up channels; an alpha channel is dropped);
  2. applies weights.transforms(), the preprocessing the network was trained with: resize so that
     the shorter side is 232 pixels, cut the central 224 x 224 square, scale to [0, 1] and
     standardise each channel with the ImageNet means and standard deviations;
  3. passes it through ResNet-50 with the last layer (fc, 1000 ImageNet classes) replaced by the
     identity, and keeps the 2048 numbers of the global average pooling.
The exact sizes are printed by `print(ResNet50_Weights.IMAGENET1K_V2.transforms())`.
"""
from __future__ import annotations

import warnings

import numpy as np

from ._common import RANDOM_WEIGHTS, models_dir, resolve_device

DIMENSION = 2048

CARD = {
    "model": "ResNet-50, torchvision weights IMAGENET1K_V2",
    "source": "https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html",
    "trained_on": "ImageNet-1k: about 1.28 million photographs, each labelled with one of 1000 object categories",
    "output": "2048 numbers of the global average pooling, before the 1000-way head",
}


def load_image(path):
    from PIL import Image
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")             # palette images with transparency warn on conversion
        image = Image.open(path)
        image.load()
        info = {"mode": image.mode, "width": image.width, "height": image.height}
        return image.convert("RGB"), info


class CoverEmbedder:
    """embedder = CoverEmbedder(); vector = embedder("cover.jpg")  ->  numpy array of 2048 numbers."""

    dimension = DIMENSION

    def __init__(self, device: str = "auto", models_folder=None):
        import torch
        from torchvision.models import ResNet50_Weights, resnet50
        self.device = resolve_device(device)
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        net = resnet50(weights=None)
        if not RANDOM_WEIGHTS:
            state = self.weights.get_state_dict(progress=True, check_hash=True,
                                                model_dir=str(models_dir(models_folder) / "torch"))
            net.load_state_dict(state)
        self._head = net.fc                         # kept only for the sanity check of top_classes
        net.fc = torch.nn.Identity()
        self.net = net.eval().to(self.device)
        self._head = self._head.eval().to(self.device)
        self.preprocess = self.weights.transforms()

    def embed_images(self, images) -> np.ndarray:
        import torch
        x = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with torch.inference_mode():
            return self.net(x).cpu().numpy()

    def embed_file(self, path) -> dict:
        image, info = load_image(path)
        return {"vector": self.embed_images([image])[0], "info": info}

    def __call__(self, path_or_image) -> np.ndarray:
        if hasattr(path_or_image, "convert"):       # already a PIL image
            return self.embed_images([path_or_image.convert("RGB")])[0]
        return self.embed_file(path_or_image)["vector"]

    def top_classes(self, path, k: int = 5) -> list[tuple[str, float]]:
        """The k ImageNet categories the full network would predict: a sanity check, and part of the answer to 10.3."""
        import torch
        image, _ = load_image(path)
        with torch.inference_mode():
            z = self.net(self.preprocess(image)[None].to(self.device))
            p = torch.softmax(self._head(z), dim=1)[0].cpu().numpy()
        names = self.weights.meta["categories"]
        return [(names[i], float(p[i])) for i in np.argsort(-p)[:k]]

    def card(self) -> dict:
        return dict(CARD, weights_url=self.weights.url, preprocessing=str(self.preprocess).replace("\n", " "),
                    dimension=DIMENSION)
