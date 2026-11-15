"""Command line: python -m ecac_extractors {check, download, extract}. Run with --help for the options."""
from __future__ import annotations

import argparse
import collections
import re
import sys
import time
from pathlib import Path

import numpy as np

from ._common import RowWriter, find_files, record_run

AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
LYRICS_EXT = {".txt"}
COVER_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _duplicates(files):
    counts = collections.Counter(p.stem for p in files)
    return sorted(stem for stem, n in counts.items() if n > 1)


def _relative(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


# The id of a row is the file name without extension (Audio_Song or Lyric_Song in DATA_DICTIONARY.md).
# The name of the folder holding the file is never written: it is the song's quadrant, the target.

def extract_audio(args, folder: Path, out: Path):
    from tqdm.auto import tqdm
    from .audio import AudioEmbedder, audioset_labels
    files = find_files(folder, AUDIO_EXT)[: args.limit]
    emb = AudioEmbedder(device=args.device, models_folder=args.models_dir)
    cols = [f"panns_{i:04d}" for i in range(emb.dimension)]
    if args.audio_std:
        cols += [f"panns_std_{i:04d}" for i in range(emb.dimension)]
    writer = RowWriter(out / "audio_panns.csv", "Audio_Song", cols)
    tag_writer = None
    if args.audio_tags:
        slugs = [re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_") for name in audioset_labels()]
        tag_writer = RowWriter(out / "audio_panns_tags.csv", "Audio_Song", [f"audioset_{s}" for s in slugs])
    done = writer.done & (tag_writer.done if tag_writer else writer.done)
    todo = [p for p in files if p.stem not in done]
    failed, rates, channels, seconds = [], collections.Counter(), collections.Counter(), []
    start = time.time()
    for i, p in enumerate(tqdm(todo, desc="audio", unit="song")):
        try:
            r = emb.embed_file(p, std=args.audio_std, tags=args.audio_tags)
        except Exception as e:
            failed.append({"file": _relative(p, folder), "error": repr(e)[:300]})
            continue
        values = np.concatenate([r["mean"], r["std"]]) if args.audio_std else r["mean"]
        if p.stem not in writer.done:
            writer.write(p.stem, values)
        if tag_writer and p.stem not in tag_writer.done:
            tag_writer.write(p.stem, r["tags"])
        rates[r["info"]["file_sample_rate"]] += 1
        channels[r["info"]["file_channels"]] += 1
        seconds.append(r["info"]["seconds"])
        if i % 20 == 0:
            writer.flush()
            tag_writer and tag_writer.flush()
    writer.close()
    tag_writer and tag_writer.close()
    record_run(out, "audio", {
        "model": emb.card(), "folder": str(folder), "files_found": len(files), "already_done": len(files) - len(todo),
        "embedded_now": len(todo) - len(failed), "failed": failed, "duplicate_ids": _duplicates(files),
        "outputs": ["audio_panns.csv"] + (["audio_panns_tags.csv"] if tag_writer else []),
        "columns": "mean over windows" + (" then standard deviation" if args.audio_std else ""),
        "files_seen_this_run": {"sample_rates": dict(rates), "channels": dict(channels),
                                "seconds_min": min(seconds, default=None), "seconds_max": max(seconds, default=None)},
        "elapsed_seconds": round(time.time() - start, 1)})


def extract_lyrics(args, folder: Path, out: Path):
    from tqdm.auto import tqdm
    from .lyrics import LyricsEmbedder
    files = find_files(folder, LYRICS_EXT)[: args.limit]
    emb = LyricsEmbedder(model=args.lyrics_model, device=args.device, models_folder=args.models_dir,
                         mode=args.lyrics_mode)
    writer = RowWriter(out / "lyrics_sbert.csv", "Lyric_Song", [f"sbert_{i:03d}" for i in range(emb.dimension)])
    todo = [p for p in files if p.stem not in writer.done]
    failed, pieces, word_pieces = [], collections.Counter(), []
    start = time.time()
    for i, p in enumerate(tqdm(todo, desc="lyrics", unit="song")):
        try:
            r = emb.embed_file(p)
        except Exception as e:
            failed.append({"file": _relative(p, folder), "error": repr(e)[:300]})
            continue
        writer.write(p.stem, r["vector"])
        pieces[r["info"]["pieces"]] += 1
        word_pieces.append(r["info"]["word_pieces"])
        if i % 50 == 0:
            writer.flush()
    writer.close()
    longer = int(sum(n > emb.max_tokens - 2 for n in word_pieces))
    record_run(out, "lyrics", {
        "model": emb.card(), "folder": str(folder), "files_found": len(files), "already_done": len(files) - len(todo),
        "embedded_now": len(todo) - len(failed), "failed": failed, "duplicate_ids": _duplicates(files),
        "outputs": ["lyrics_sbert.csv"],
        "files_seen_this_run": {"pieces_per_lyric": dict(sorted(pieces.items())),
                                "word_pieces_median": float(np.median(word_pieces)) if word_pieces else None,
                                "lyrics_longer_than_the_model_reads": longer},
        "elapsed_seconds": round(time.time() - start, 1)})


def extract_covers(args, folder: Path, out: Path):
    from tqdm.auto import tqdm
    from .covers import CoverEmbedder, load_image
    files = find_files(folder, COVER_EXT)[: args.limit]
    emb = CoverEmbedder(device=args.device, models_folder=args.models_dir)
    writer = RowWriter(out / "covers_resnet50.csv", "Audio_Song", [f"resnet_{i:04d}" for i in range(emb.dimension)])
    todo = [p for p in files if p.stem not in writer.done]
    failed, modes, sizes = [], collections.Counter(), collections.Counter()
    start = time.time()
    bar = tqdm(total=len(todo), desc="covers", unit="cover")
    for b in range(0, len(todo), args.batch_size):
        batch, images = [], []
        for p in todo[b:b + args.batch_size]:
            try:
                image, info = load_image(p)
            except Exception as e:
                failed.append({"file": _relative(p, folder), "error": repr(e)[:300]})
                continue
            batch.append(p)
            images.append(image)
            modes[info["mode"]] += 1
            sizes[f'{info["width"]}x{info["height"]}'] += 1
        if images:
            for p, v in zip(batch, emb.embed_images(images)):
                writer.write(p.stem, v)
            writer.flush()
        bar.update(len(todo[b:b + args.batch_size]))
    bar.close()
    writer.close()
    record_run(out, "covers", {
        "model": emb.card(), "folder": str(folder), "files_found": len(files), "already_done": len(files) - len(todo),
        "embedded_now": len(todo) - len(failed), "failed": failed, "duplicate_ids": _duplicates(files),
        "outputs": ["covers_resnet50.csv"],
        "files_seen_this_run": {"image_modes": dict(modes), "most_common_sizes": dict(sizes.most_common(5))},
        "elapsed_seconds": round(time.time() - start, 1)})


def cmd_extract(args):
    media = Path(args.media) if args.media else None
    out = Path(args.out)
    folders = {"audio": args.audio_dir or (media / "audio" if media else None),
               "lyrics": args.lyrics_dir or (media / "lyrics" if media else None),
               "covers": args.covers_dir or (media / "album_covers" if media else None)}
    run = {"audio": extract_audio, "lyrics": extract_lyrics, "covers": extract_covers}
    for domain in args.domains:
        if folders[domain] is None:
            raise SystemExit(f"give --media, or --{domain}-dir")
        print(f"== {domain}: {folders[domain]}", file=sys.stderr)
        run[domain](args, Path(folders[domain]), out)
    print(f"done; what was used is in {out / 'models_used.json'}", file=sys.stderr)


def cmd_download(args):
    from .audio import AudioEmbedder
    from .covers import CoverEmbedder
    from .lyrics import LyricsEmbedder
    for name, make in (("audio", lambda: AudioEmbedder(device="cpu", models_folder=args.models_dir)),
                       ("lyrics", lambda: LyricsEmbedder(model=args.lyrics_model, device="cpu",
                                                         models_folder=args.models_dir)),
                       ("covers", lambda: CoverEmbedder(device="cpu", models_folder=args.models_dir))):
        card = make().card()
        print(f"{name}: {card['model']} ready", file=sys.stderr)


def cmd_check(args):
    """Downloads the three models and runs each on a known input; prints what a working installation gives."""
    from .audio import SAMPLE_RATE, AudioEmbedder
    from .covers import CoverEmbedder
    from .lyrics import LyricsEmbedder
    ok = True

    print("\n== audio: PANNs CNN14")
    t = time.time()
    audio = AudioEmbedder(device=args.device, models_folder=args.models_dir)
    s = np.arange(30 * SAMPLE_RATE) / SAMPLE_RATE
    chord = sum(np.sin(2 * np.pi * f * s) / (1 + k) for k in range(4) for f in (220 * (k + 1), 277.18 * (k + 1), 329.63 * (k + 1)))
    chord = (0.3 * chord / np.abs(chord).max() * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * s) ** 2)).astype(np.float32)
    emb, _ = audio.windows(chord)
    tags = audio.top_tags(chord)
    print(f"30 s of a synthetic A major chord: {emb.shape[0]} windows x {emb.shape[1]} numbers, {time.time() - t:.1f} s")
    print("top AudioSet classes:", ", ".join(f"{n} {p:.2f}" for n, p in tags))
    if not any("music" in n.lower() or "sine" in n.lower() or "organ" in n.lower() or "synth" in n.lower()
               or "tone" in n.lower() or "hum" in n.lower() for n, _ in tags):
        ok = False
        print("  CHECK: expected a musical or tonal class among them")
    if audio.checkpoint:
        print("checkpoint sha256:", audio.checkpoint["sha256"])

    print("\n== lyrics: all-MiniLM-L6-v2")
    t = time.time()
    lyr = LyricsEmbedder(model=args.lyrics_model, device=args.device, models_folder=args.models_dir)
    a = lyr("I feel so happy and alive tonight")
    b = lyr("Joy fills my heart and I can't stop smiling")
    c = lyr("The train to the city leaves at noon")
    print(f"{len(a)} numbers per text, length {np.linalg.norm(a):.3f}, {time.time() - t:.1f} s")
    print(f"cosine(happy, joy) = {a @ b:.3f}   cosine(happy, train) = {a @ c:.3f}")
    if not a @ b > a @ c:
        ok = False
        print("  CHECK: the two happy sentences should be the closer pair")
    print(f"reads at most {lyr.max_tokens} word pieces; model commit {lyr.commit}")

    print("\n== covers: ResNet-50 IMAGENET1K_V2")
    t = time.time()
    cov = CoverEmbedder(device=args.device, models_folder=args.models_dir)
    image = Path(__file__).with_name("check_image.jpg")
    v = cov(image)
    top = cov.top_classes(image)
    print(f"{len(v)} numbers per image, {time.time() - t:.1f} s")
    print("a photograph of a cat; top ImageNet classes:", ", ".join(f"{n} {p:.2f}" for n, p in top))
    if not any("cat" in n.lower() or "tabby" in n.lower() for n, _ in top[:3]):
        ok = False
        print("  CHECK: expected a cat among the first three")
    print("\n" + ("everything works" if ok else "something is wrong: see the CHECK lines above"))
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ecac_extractors",
                                description="Pre-trained feature extractors for the ECAC project: audio, lyrics, covers.")
    sub = p.add_subparsers(dest="command", required=True)

    def common(q):
        q.add_argument("--device", default="auto", help="auto (GPU if there is one), cpu, cuda or mps")
        q.add_argument("--models-dir", default=None,
                       help="where the weights are kept (default $ECAC_MODELS_DIR or ~/.cache/ecac_extractors)")
        q.add_argument("--lyrics-model", default="sentence-transformers/all-MiniLM-L6-v2",
                       help="any sentence-transformers model, by name or folder")

    q = sub.add_parser("check", help="download the models and test them on known inputs")
    common(q)
    q = sub.add_parser("download", help="only download the models")
    common(q)
    q = sub.add_parser("extract", help="one CSV per domain, one row per file")
    common(q)
    q.add_argument("--media", help="folder holding audio/, lyrics/ and album_covers/")
    q.add_argument("--out", required=True, help="folder for the CSV files and models_used.json")
    q.add_argument("--domains", nargs="+", default=["audio", "lyrics", "covers"], choices=["audio", "lyrics", "covers"])
    q.add_argument("--audio-dir", help="instead of MEDIA/audio")
    q.add_argument("--lyrics-dir", help="instead of MEDIA/lyrics")
    q.add_argument("--covers-dir", help="instead of MEDIA/album_covers")
    q.add_argument("--limit", type=int, default=None, help="only the first N files of each domain (to try it out)")
    q.add_argument("--audio-std", action="store_true", help="also the standard deviation over the windows (2048 more columns)")
    q.add_argument("--audio-tags", action="store_true",
                   help="also the 527 AudioSet class probabilities, in audio_panns_tags.csv")
    q.add_argument("--lyrics-mode", default="chunks", choices=["chunks", "truncate"],
                   help="chunks: the whole lyric, in pieces the model can read; truncate: its beginning only")
    q.add_argument("--batch-size", type=int, default=16, help="covers per batch")
    args = p.parse_args(argv)
    if args.command == "check":
        return cmd_check(args)
    if args.command == "download":
        return cmd_download(args)
    return cmd_extract(args)


if __name__ == "__main__":
    sys.exit(main())
