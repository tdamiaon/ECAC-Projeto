# Pre-trained feature extractors (ECAC project, Section 10.2)

Three pre-trained models, one per domain, used **frozen**: nothing in them is trained on the
project's songs. They turn every song into a row of numbers, the learned representations of
Section 10 of the brief.

| domain | model | numbers per song | output file | id column |
|---|---|---|---|---|
| audio | PANNs CNN14 (32 kHz) | 2048 | `audio_panns.csv` | `Audio_Song` |
| lyrics | all-MiniLM-L6-v2 | 384 | `lyrics_sbert.csv` | `Lyric_Song` |
| covers | ResNet-50, weights IMAGENET1K_V2 | 2048 | `covers_resnet50.csv` | `Audio_Song` |

The id of a row is the file name without its extension. The name of the folder that holds
the file (`Q1` to `Q4`) is never written: it is the quadrant, the target.

These are a starting point, not an obligation (Section 10.2): other models may be used instead.

## Install

Use a virtual environment, then install the requirements from this folder:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Linux without an NVIDIA GPU, run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
before the line above; otherwise pip downloads the GPU build of PyTorch, several gigabytes.

Then, from this folder, check that everything works:

```bash
python -m ecac_extractors check
```

The first run downloads the weights, about 330 MB for CNN14, 90 MB for MiniLM and 100 MB for
ResNet-50, into `~/.cache/ecac_extractors` (or the folder in `--models-dir`, or in the
environment variable `ECAC_MODELS_DIR`). It then runs each model on a known input and prints what
it sees: the AudioSet classes of a synthetic chord, the cosine similarity of two happy sentences
against an unrelated one, and the ImageNet classes of a photograph of a cat. It ends with
`everything works`, or with `CHECK` lines that say what went wrong.

## Extract the features of the dataset

`--media` is the folder that holds `audio/`, `lyrics/` and `album_covers/`:

```bash
python -m ecac_extractors extract --media path/to/MERGE_ECAC --out features
```

- One CSV per domain in `features/`, one row per file, and `models_used.json`.
- `--domains audio lyrics covers` chooses the domains; `--limit 20` tries the first 20 files only.
- The audio takes longest: in our tests about half a second per song on a CPU, so the whole
  dataset takes a few tens of minutes. The run can be interrupted and started again with the same
  command: files already in the CSV are skipped.
- Files that cannot be read are listed, with the error, under `failed` in `models_used.json`;
  they get no row. Songs without a cover get no row in `covers_resnet50.csv`.

`models_used.json` records, for each domain, the model, its exact version (the checkpoint's
SHA-256, the commit of the text model, the URL of the image weights), every parameter, the
library versions, and what the files were before they were transformed (sample rates, channels,
image modes, the number of lyrics longer than the text model reads). Keep it with the report:
it is the "name, version and source" Section 10.2 asks for, and the raw material for 10.4.

### Options

| option | effect |
|---|---|
| `--audio-std` | also the standard deviation of the 2048 numbers over the windows of the song (2048 more columns) |
| `--audio-tags` | also the probability of each of the 527 AudioSet classes, averaged over the windows, in `audio_panns_tags.csv` |
| `--lyrics-mode truncate` | encode only the beginning of each lyric, as the model does by itself (default `chunks`: the whole lyric) |
| `--lyrics-model NAME` | any other sentence-transformers model, by name or by folder |
| `--device cpu` | `auto` (default) uses an NVIDIA GPU when there is one; `mps` uses an Apple GPU |
| `--models-dir DIR` | where the weights are, or will be downloaded |

## The three models

What each extractor does to its input is listed here because Section 10.4 asks for it. What was
lost in the process, and how far each training task is from emotion (10.3), is for you to discuss.

### Audio: PANNs CNN14

- **Source**: Kong et al. (2020), *PANNs: Large-scale pretrained audio neural networks for audio
  pattern recognition*. Weights `Cnn14_mAP=0.431.pth`, https://zenodo.org/record/3987831;
  code https://github.com/qiuqiangkong/audioset_tagging_cnn (the network is copied in
  `ecac_extractors/cnn14.py`, MIT licence).
- **Trained on**: AudioSet, about 2 million clips of 10 s from YouTube, each tagged with some of
  527 sound classes. The list is in `ecac_extractors/audioset_labels.csv`.
- **What the extractor does**:
  - it decodes the file and averages the channels into one;
  - it resamples to 32 000 samples per second;
  - it cuts the recording into windows of 10 s (a 30 s excerpt gives 3; the last window is aligned with the end);
  - it computes a log-mel spectrogram of each window, with 64 bands between 50 Hz and 14 kHz, as the network was trained;
  - it keeps the 2048 activations of the penultimate layer of each window;
  - it averages them over the windows of the song.
- **Output**: `panns_0000` to `panns_2047`.

### Lyrics: all-MiniLM-L6-v2

- **Source**: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 (sentence-transformers).
- **Trained on**: a small BERT-like model (six transformer layers) fine-tuned by contrastive
  learning on about one billion pairs of related texts, so that the cosine of two text vectors
  measures how close their meanings are. See its model card.
- **What the extractor does**:
  - it reads the file as UTF-8 and drops empty lines;
  - the model reads at most 256 word pieces (the sub-word units of its tokeniser) and ignores the rest, so the extractor splits each lyric at line breaks into consecutive pieces of at most 256 word pieces;
  - each piece is tokenised, passed through the model, averaged over its tokens and scaled to length 1;
  - the song is the mean of its pieces, weighted by their size, scaled to length 1.
- **Output**: `sbert_000` to `sbert_383`.

### Covers: ResNet-50

- **Source**: torchvision, `ResNet50_Weights.IMAGENET1K_V2`
  (https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).
- **Trained on**: ImageNet-1k, about 1.28 million photographs, each labelled with one of 1000
  object categories.
- **What the extractor does**:
  - it converts the image to RGB (an alpha channel is dropped);
  - it applies the preprocessing the weights were trained with, `weights.transforms()`: resize the shorter side to 232 pixels, cut the central 224 × 224 square, scale to [0, 1], and standardise each channel with the ImageNet means and standard deviations;
  - it replaces the last layer (1000 ImageNet classes) by the identity and keeps the 2048 numbers before it.
- **Output**: `resnet_0000` to `resnet_2047`.

## Using them in Python, and in the application of Section 16

```python
from pathlib import Path
from ecac_extractors import AudioEmbedder, LyricsEmbedder, CoverEmbedder

audio, lyrics, covers = AudioEmbedder(), LyricsEmbedder(), CoverEmbedder()   # load once
a = audio("song.mp3")                                   # numpy array, 2048 numbers
l = lyrics(Path("song.txt").read_text(encoding="utf-8"))   # 384 numbers (a str is the lyric itself)
c = covers("cover.jpg")                                 # 2048 numbers

audio.top_tags("song.mp3")        # the AudioSet classes CNN14 hears in the song
covers.top_classes("cover.jpg")   # the ImageNet classes ResNet-50 sees in the cover
```

The application must compute the vectors of a new song with the same models and the same
preprocessing; these functions are exactly what the extraction used.

## The protocol (Section 10.6)

A frozen model fits nothing on the songs: the same file always gives the same vector. Its vectors
can therefore be computed once, for every song, test songs included. Everything fitted after
them is fitted inside the training folds only:

- the scaling of the columns;
- the weights given to each block when blocks are joined;
- PCA, LDA and the selection of features.

The CSV files are raw: the join with `metadata.csv`, and what to do with duplicated recordings
and missing covers, are part of the work (Sections 2.2, 4.1 and 4.5).

## When something fails

- **A download is blocked** (some networks block Zenodo or Hugging Face): run `check` on another
  network and copy the folder `~/.cache/ecac_extractors` across, or point `--models-dir` at a
  copy of it.
- **An MP3 cannot be decoded**: check that `soundfile` is 0.12 or later
  (`python -c "import soundfile; print(soundfile.__libsndfile_version__)"`, at least 1.1); for
  other formats, installing `ffmpeg` lets librosa decode them.
- **Out of memory on the covers**: lower `--batch-size`.

## Credits

The CNN14 code is from the PANNs repository (MIT licence, `ecac_extractors/PANNS_LICENSE.txt`).
`check_image.jpg` is the photograph *chelsea* distributed with scikit-image (CC0, Stefan van der
Walt). The weights of each model come with the licence of their source; read it there.
