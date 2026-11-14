"""Pre-trained feature extractors for the ECAC project (Section 10.2 of the brief).

    from ecac_extractors import AudioEmbedder, LyricsEmbedder, CoverEmbedder
    audio = AudioEmbedder()          # PANNs CNN14            -> 2048 numbers per song
    lyrics = LyricsEmbedder()        # all-MiniLM-L6-v2       ->  384 numbers per song
    covers = CoverEmbedder()         # ResNet-50 IMAGENET1K_V2 -> 2048 numbers per cover
    v = audio("song.mp3"); w = lyrics(text); z = covers("cover.jpg")

Each model is used frozen: nothing is fitted on the project's songs, so the vectors of every
song, test songs included, can be computed once. Every step fitted after them (scaling, weights
of the blocks, PCA, LDA, selection) belongs inside the training folds.
"""
from ._common import PACKAGE_VERSION as __version__
from .audio import AudioEmbedder
from .covers import CoverEmbedder
from .lyrics import LyricsEmbedder

__all__ = ["AudioEmbedder", "LyricsEmbedder", "CoverEmbedder", "__version__"]
