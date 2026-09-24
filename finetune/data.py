"""Dataset and frame construction for Zonos2 fine-tuning.

Each training sequence mirrors the inference-time prompt layout exactly
(scheduler.py:372 _with_speaker_frames + tokenizer prompt building):

    [speaker slot]                      1 frame, all audio_pad + text padding
    [background marker]                 if the checkpoint defines it
    [accurate-mode marker]              if defined and accurate_mode=True
    [speaking-rate row]                 optional
    [quality rows]                      optional (server default: trailing_silence_s)
    [BOS, utf-8 bytes ..., EOS]         text rows, audio columns = audio_pad
    [sheared audio stream]              silence(17) + DAC codes + EOA + tail

The audio stream is built unsheared as ``silence ++ codes ++ EOA-row ++ (C-1)
pad rows`` and sheared once; because shear(row t) only references rows <= t, its
first 17 rows equal the inference prompt's sheared silence, and the delayed EOA
pattern in the tail matches what the engine's delayed-EOS countdown expects
(core.py:167).

Loss labels are next-frame audio codes: positions predicting the prompt or the
given silence are ignored, as are shear-fill audio_pad cells; EOA cells are kept
so the model learns to stop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from zonos2_compat import prompt_module

IGNORE_INDEX = -100
SILENCE_FRAMES = 17  # 0.2 s prefix, matching _SILENCE_TOKENS_0_2S

# Server-side default quality conditioning (api_server DEFAULT_QUALITY_BUCKETS).
DEFAULT_QUALITY = {"trailing_silence_s": 3}


class PromptSpec:
    """Conditioning-token layout resolved from the checkpoint config."""

    def __init__(
        self,
        config,
        *,
        quality: Optional[Dict[str, int]] = DEFAULT_QUALITY,
        clean_background: bool = False,
        accurate_mode: bool = True,
        speaking_rate_bucket: Optional[int] = None,
    ):
        pm = prompt_module()
        self.pm = pm
        self.n_codebooks = int(config.n_codebooks)
        self.audio_pad_id = int(config.audio_pad_id)
        self.eoa_id = int(config.eoa_id)
        if config.text_vocab is None:
            raise ValueError("Checkpoint config has no text_vocab; cannot build prompts.")
        self.text_vocab = int(config.text_vocab)

        self.rate_num_buckets = int(getattr(config, "speaking_rate_num_buckets", 0) or 0)
        features = list(getattr(config, "quality_features", None) or [])
        buckets = dict(getattr(config, "quality_buckets", None) or {})
        self.quality_features = features
        self.quality_bucket_counts = tuple(len(buckets.get(f, [])) for f in features)
        bg_enabled = bool(getattr(config, "speaker_background_token_enabled", False))
        acc_enabled = bool(getattr(config, "accurate_mode_token_enabled", False))
        self.background_num_buckets = 2 if bg_enabled else 0
        # Accurate-mode markers only exist alongside background markers (hf.py rule).
        self.accurate_num_buckets = 1 if (acc_enabled and bg_enabled) else 0

        self.clean_background = bool(clean_background)
        self.accurate_mode = bool(accurate_mode)
        self.speaking_rate_bucket = speaking_rate_bucket

        # Per-feature quality bucket selection aligned with quality_features order.
        self.quality_selection: Optional[List[Optional[int]]] = None
        if quality:
            selection: List[Optional[int]] = [None] * len(features)
            for name, bucket in quality.items():
                if name not in features:
                    continue
                idx = features.index(name)
                count = self.quality_bucket_counts[idx]
                if count > 0:
                    selection[idx] = min(int(bucket), count - 1)
            if any(v is not None for v in selection):
                self.quality_selection = selection

    # -- prompt rows -------------------------------------------------------

    def _marker_row(self, text_token: int) -> List[int]:
        return [self.audio_pad_id] * self.n_codebooks + [int(text_token)]

    def build_prompt_rows(self, text: str, with_speaker: bool) -> List[List[int]]:
        pm = self.pm
        rows: List[List[int]] = []
        if with_speaker:
            rows.append(self._marker_row(self.text_vocab))  # reserved speaker slot
            if self.background_num_buckets:
                rows.append(
                    self._marker_row(
                        pm.speaker_background_token_id(
                            self.text_vocab,
                            self.rate_num_buckets,
                            self.quality_bucket_counts,
                            self.clean_background,
                            self.background_num_buckets,
                            self.accurate_num_buckets,
                        )
                    )
                )
                if self.accurate_num_buckets and self.accurate_mode:
                    rows.append(
                        self._marker_row(
                            pm.accurate_mode_token_id(
                                self.text_vocab,
                                self.rate_num_buckets,
                                self.quality_bucket_counts,
                                self.background_num_buckets,
                                self.accurate_num_buckets,
                            )
                        )
                    )
        rows.extend(
            pm.tokens_to_prompt_tokens(
                pm.text_to_byte_ids(text),
                n_codebooks=self.n_codebooks,
                audio_pad_id=self.audio_pad_id,
                text_vocab=self.text_vocab,
                speaking_rate_num_buckets=self.rate_num_buckets,
                speaking_rate_bucket=self.speaking_rate_bucket,
                quality_bucket_counts=self.quality_bucket_counts,
                quality_buckets=self.quality_selection,
                speaker_background_num_buckets=self.background_num_buckets,
                accurate_mode_num_buckets=self.accurate_num_buckets,
            )
        )
        return rows

    # -- full training sequence -------------------------------------------

    def build_example(
        self, text: str, codes: torch.Tensor, with_speaker: bool = True
    ) -> Dict[str, torch.Tensor]:
        """codes: (T, n_codebooks) int; returns frames (S, C+1) and labels (S-1, C)."""
        pm = self.pm
        C = self.n_codebooks
        pad = self.audio_pad_id

        prompt = torch.tensor(self.build_prompt_rows(text, with_speaker), dtype=torch.long)
        prompt_len = prompt.shape[0]

        silence = torch.tensor(pm._SILENCE_TOKENS_0_2S, dtype=torch.long)[:, :C]
        eoa_row = torch.full((1, C), self.eoa_id, dtype=torch.long)
        tail = torch.full((C - 1, C), pad, dtype=torch.long)
        stream = torch.cat([silence, codes.long(), eoa_row, tail], dim=0)
        sheared = pm.shear(stream, pad)
        text_col = torch.full((sheared.shape[0], 1), self.text_vocab, dtype=torch.long)
        audio_frames = torch.cat([sheared, text_col], dim=1)

        frames = torch.cat([prompt, audio_frames], dim=0)  # (S, C+1)

        labels = frames[1:, :C].clone()  # target for position t is frame t+1
        # No loss while the target is still prompt or the given silence prefix.
        first_target = prompt_len + SILENCE_FRAMES  # index into frames
        labels[: first_target - 1] = IGNORE_INDEX
        labels[labels == pad] = IGNORE_INDEX  # shear fill / tail padding

        return {"frames": frames, "labels": labels}

    def pad_frame(self) -> torch.Tensor:
        return torch.tensor(self._marker_row(self.text_vocab), dtype=torch.long)


class Zonos2FinetuneDataset(Dataset):
    """Reads .pt files produced by finetune/preprocess.py."""

    def __init__(
        self,
        data_dir: str | Path,
        spec: PromptSpec,
        max_frames: Optional[int] = None,
        require_speaker: bool = True,
    ):
        self.spec = spec
        self.max_frames = max_frames
        self.require_speaker = require_speaker
        self.files = sorted(Path(data_dir).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No preprocessed .pt files in {data_dir}")

        self._skipped = 0
        if max_frames is not None:
            kept = []
            for f in self.files:
                item = torch.load(f, map_location="cpu", weights_only=False)
                approx = item["codes"].shape[0] + SILENCE_FRAMES + len(item["text"]) + 32
                if approx <= max_frames:
                    kept.append(f)
                else:
                    self._skipped += 1
            self.files = kept
            if not self.files:
                raise ValueError(f"All samples exceed max_frames={max_frames}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        speaker = item.get("speaker_embedding")
        with_speaker = speaker is not None
        if self.require_speaker and not with_speaker:
            raise ValueError(f"{self.files[idx]} has no speaker_embedding")
        example = self.spec.build_example(
            item["text"], item["codes"], with_speaker=with_speaker
        )
        if with_speaker:
            example["speaker_embedding"] = speaker.float()
        return example


def collate(batch: List[Dict[str, torch.Tensor]], spec: PromptSpec) -> Dict[str, torch.Tensor]:
    max_len = max(ex["frames"].shape[0] for ex in batch)
    pad_frame = spec.pad_frame()

    frames, labels, speakers = [], [], []
    for ex in batch:
        f, l = ex["frames"], ex["labels"]
        pad_rows = max_len - f.shape[0]
        if pad_rows:
            f = torch.cat([f, pad_frame.unsqueeze(0).expand(pad_rows, -1)], dim=0)
            l = torch.cat(
                [l, torch.full((pad_rows, l.shape[1]), IGNORE_INDEX, dtype=torch.long)], dim=0
            )
        frames.append(f)
        labels.append(l)
        if "speaker_embedding" in ex:
            speakers.append(ex["speaker_embedding"])

    out = {
        "frames": torch.stack(frames),           # (B, S, C+1)
        "labels": torch.stack(labels),           # (B, S-1, C)
    }
    if speakers:
        if len(speakers) != len(batch):
            raise ValueError("Mixed speaker/no-speaker batches are not supported")
        out["speaker_embedding"] = torch.stack(speakers)  # (B, D)
        out["speaker_positions"] = torch.zeros(len(batch), dtype=torch.long)
    return out


def load_manifest(path: str | Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
