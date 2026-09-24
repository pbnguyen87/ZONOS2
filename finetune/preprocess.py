"""Preprocess audio+text pairs into training tensors for Zonos2 fine-tuning.

For every utterance this computes and caches:
  * DAC codes        (T, 9) int16 — descript-audio-codec 44 kHz, the same codec
                     the vocoder decodes (tokenizer/vocoder.py)
  * speaker embedding (2048,) fp32 — Qwen3 voice encoder, same as the server
                     (models/speaker_cloning.py), taken from the utterance itself
                     (self-cloning, the standard recipe for speaker fine-tuning)
  * normalized text  — NeMo forward TN, same normalizer as the server; falls back
                     to raw text if unavailable (matching server behavior)

Input is a JSONL manifest with rows {"audio": path, "text": str, "language": "en_us"}
or a directory of audio files with same-stem .txt transcripts.

Usage:
  python finetune/preprocess.py \
      --model-path Zyphra/ZONOS2 \
      --manifest data/train.jsonl \
      --output-dir data/preprocessed \
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zonos2_compat import (
    load_config,
    prompt_module,
    speaker_cloning_module,
    textnorm_module,
)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".webm"}
DAC_SAMPLE_RATE = 44_100


def _load_audio(path: str) -> tuple[torch.Tensor, int]:
    import torchaudio

    wav, sr = torchaudio.load(path)
    return wav, sr


def _to_dac_input(wav: torch.Tensor, sr: int) -> torch.Tensor:
    import torchaudio

    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != DAC_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, DAC_SAMPLE_RATE)
    return wav


class Preprocessor:
    def __init__(self, model_path: str, device: str, normalize: bool):
        self.config = load_config(model_path)
        self.device = device
        self.n_codebooks = int(self.config.n_codebooks)
        self.speaker_dim = (
            int(self.config.speaker_embedding_dim)
            if getattr(self.config, "speaker_enabled", False)
            else 0
        )

        import dac  # descript-audio-codec, already a repo dependency

        self.dac = dac.DAC.load(dac.utils.download(model_type="44khz")).eval().to(device)

        self.speaker_encoder = None
        if self.speaker_dim > 0:
            self.speaker_encoder = speaker_cloning_module().Qwen3SpeakerEmbedding(
                device=device
            )

        self.normalizer = None
        if normalize:
            try:
                self.normalizer = textnorm_module().TTSTextNormalizer()
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] text normalizer unavailable ({exc}); using raw text")

    @torch.inference_mode()
    def encode_codes(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        x = _to_dac_input(wav, sr).to(self.device)
        x = self.dac.preprocess(x.unsqueeze(0), DAC_SAMPLE_RATE)  # (1, 1, T)
        _, codes, *_ = self.dac.encode(x)  # codes: (1, n_codebooks, T)
        return codes[0, : self.n_codebooks].T.to(torch.int16).cpu()  # (T, C)

    @torch.inference_mode()
    def embed_speaker(self, wav: torch.Tensor, sr: int) -> torch.Tensor | None:
        if self.speaker_encoder is None:
            return None
        out = self.speaker_encoder(wav, sr)
        candidates = (
            [t.squeeze(0) for t in out] if isinstance(out, tuple) else [out.squeeze(0)]
        )
        for cand in candidates:
            if cand.numel() == self.speaker_dim:
                return cand.reshape(-1).float().cpu().contiguous()
        raise ValueError(
            f"Speaker encoder produced {[c.numel() for c in candidates]}, "
            f"model expects {self.speaker_dim}"
        )

    def normalize_text(self, text: str, language: str) -> str:
        if self.normalizer is None:
            return text
        return self.normalizer.normalize(text, language)

    def process_one(self, audio_path: str, text: str, language: str) -> dict:
        wav, sr = _load_audio(audio_path)
        codes = self.encode_codes(wav, sr)
        speaker = self.embed_speaker(wav, sr)
        norm_text = self.normalize_text(text, language)
        return {
            "codes": codes,
            "speaker_embedding": speaker,
            "text": norm_text,
            "raw_text": text,
            "language": language,
            "audio_path": str(audio_path),
        }


def collect_rows(args) -> list[dict]:
    rows: list[dict] = []
    if args.manifest:
        with open(args.manifest) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif args.audio_dir:
        for path in sorted(Path(args.audio_dir).rglob("*")):
            if path.suffix.lower() in AUDIO_EXTS:
                txt = path.with_suffix(".txt")
                if txt.is_file():
                    rows.append(
                        {"audio": str(path), "text": txt.read_text().strip()}
                    )
                else:
                    print(f"[warn] no transcript for {path}, skipping")
    else:
        raise SystemExit("Provide --manifest or --audio-dir")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, help="Checkpoint dir or HF repo id")
    ap.add_argument("--manifest", help="JSONL: {audio, text, language?} per line")
    ap.add_argument("--audio-dir", help="Directory of audio files with .txt sidecars")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--language", default="en_us", help="Default when a row has none")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rows = collect_rows(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = Preprocessor(args.model_path, args.device, normalize=not args.no_normalize)
    frame_rate = DAC_SAMPLE_RATE / 512  # DAC hop length

    index = []
    for i, row in enumerate(rows):
        out_path = out_dir / f"{i:06d}_{Path(row['audio']).stem}.pt"
        if out_path.exists() and not args.overwrite:
            index.append(out_path.name)
            continue
        try:
            item = pre.process_one(
                row["audio"], row["text"], row.get("language", args.language)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed on {row['audio']}: {exc}")
            continue
        seconds = item["codes"].shape[0] / frame_rate
        if seconds > args.max_seconds:
            print(f"[warn] {row['audio']} is {seconds:.1f}s > {args.max_seconds}s, skipping")
            continue
        torch.save(item, out_path)
        index.append(out_path.name)
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"[{i + 1}/{len(rows)}] processed")

    with open(out_dir / "index.json", "w") as f:
        json.dump({"files": index, "model_path": args.model_path}, f, indent=2)
    print(f"Done: {len(index)} samples in {out_dir}")

    # Sanity: prompt building must work with this config (fails fast on bad text_vocab).
    prompt_module()
    _ = load_config(args.model_path).text_vocab


if __name__ == "__main__":
    main()
