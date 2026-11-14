"""Lyrics: the sentence-transformers model all-MiniLM-L6-v2.

What this extractor does to a lyric (the facts needed for Section 10.4):
  1. reads the file as UTF-8 and drops empty lines;
  2. the model reads at most 256 word pieces (sub-word units of its tokeniser) and ignores the rest,
     so, by default ("chunks"), the lyric is split at line breaks into consecutive pieces of at most
     256 word pieces, each piece is encoded, and the song vector is the mean of the piece vectors,
     weighted by their number of word pieces and scaled back to length 1;
     with mode "truncate" the whole lyric is passed once and the model keeps its first 256 word pieces;
  3. each piece goes through the model: tokenisation, six transformer layers, the mean of the token
     vectors, and a normalisation to length 1; 384 numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ._common import models_dir, resolve_device

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = None             # None = the latest; fill in with the commit recorded in models_used.json to pin it

CARD = {
    "model": "all-MiniLM-L6-v2 (sentence-transformers)",
    "source": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    "trained_on": ("MiniLM, a small BERT-like language model, fine-tuned by contrastive learning on about "
                   "1 billion pairs of related texts (questions and answers, titles and abstracts, paraphrases...) "
                   "so that the cosine of two text vectors measures how similar their meanings are; see the model card"),
    "output": "384 numbers per text, of length 1",
}


def read_lyrics(path) -> str:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


class LyricsEmbedder:
    """embedder = LyricsEmbedder(); vector = embedder(text)  ->  numpy array of 384 numbers."""

    def __init__(self, model: str = MODEL_NAME, revision: str | None = REVISION, device: str = "auto",
                 models_folder=None, mode: str = "chunks"):
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging
        logging.set_verbosity_error()               # silences the length warnings of the tokeniser
        if mode not in ("chunks", "truncate"):
            raise ValueError("mode is 'chunks' or 'truncate'")
        self.mode = mode
        self.model_name = model
        if Path(model).exists():                    # a model folder on disk
            path, self.commit = Path(model), None
        else:                                       # a name on the Hugging Face hub
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(model, revision=revision,
                                          cache_dir=str(models_dir(models_folder) / "huggingface")))
            self.commit = path.name                 # the folder of a snapshot is named after its commit
        self.model = SentenceTransformer(str(path), device=resolve_device(device))
        self.tokenizer = self.model.tokenizer
        self.max_tokens = int(self.model.max_seq_length)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dimension = int(get_dim())

    def _n_tokens(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text))

    def pieces(self, text: str) -> tuple[list[str], list[int]]:
        """Consecutive pieces of whole lines, each of at most max_tokens word pieces (special tokens included)."""
        budget = self.max_tokens - 2                # [CLS] and [SEP]
        pieces, sizes, current, size = [], [], [], 0
        for line in text.splitlines():
            n = self._n_tokens(line)
            if n > budget:                          # a single line too long for the model: cut it by words
                words, line_parts = line.split(), []
                part, part_size = [], 0
                for w in words:
                    m = self._n_tokens(w)
                    if part and part_size + m > budget:
                        line_parts.append((" ".join(part), part_size))
                        part, part_size = [], 0
                    part.append(w)
                    part_size += m
                if part:
                    line_parts.append((" ".join(part), part_size))
            else:
                line_parts = [(line, n)]
            for segment, m in line_parts:
                if current and size + m > budget:
                    pieces.append("\n".join(current))
                    sizes.append(size)
                    current, size = [], 0
                current.append(segment)
                size += m
        if current:
            pieces.append("\n".join(current))
            sizes.append(size)
        return pieces, sizes

    def embed_text(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("empty text")
        if self.mode == "truncate":
            n = self._n_tokens(text)
            vector = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
            return {"vector": vector, "info": {"word_pieces": n, "pieces": 1,
                                               "word_pieces_ignored": max(0, n - (self.max_tokens - 2))}}
        pieces, sizes = self.pieces(text)
        E = self.model.encode(pieces, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
        w = np.asarray(sizes, dtype=np.float64)
        v = (E * w[:, None]).sum(axis=0) / w.sum()
        v = v / np.linalg.norm(v)
        return {"vector": v.astype(np.float32), "info": {"word_pieces": int(w.sum()), "pieces": len(pieces),
                                                          "word_pieces_ignored": 0}}

    def embed_file(self, path) -> dict:
        return self.embed_text(read_lyrics(path))

    def __call__(self, text_or_path) -> np.ndarray:
        """A str is the lyric itself; a pathlib.Path is a file to read."""
        if isinstance(text_or_path, Path):
            return self.embed_file(text_or_path)["vector"]
        return self.embed_text(text_or_path)["vector"]

    def card(self) -> dict:
        return dict(CARD, model_id=self.model_name, commit=self.commit, max_word_pieces=self.max_tokens,
                    dimension=self.dimension, mode=self.mode,
                    summary=("mean of the piece vectors weighted by word pieces, scaled to length 1"
                             if self.mode == "chunks" else "first max_word_pieces word pieces only"))
