"""Generate Vietnamese speech with ZONOS2 through the offline Python API.

Why a dedicated script: ZONOS2's built-in NeMo text normalization covers 10 languages
(en_us, en_gb, fr_fr, de, es, it, pt_br, ja, cmn, ko) and rejects other codes. For
Vietnamese this script normalizes numbers, dates, percentages and common abbreviations
into words itself, then calls the engine with ``text_normalization=False`` so the
UTF-8 bytes go in verbatim. Everything else (speaker cloning, accurate/expressive mode,
speaking rate, quality buckets, sampling) is the same as the server.

Requires the repo environment on Linux + NVIDIA GPU (``uv sync``). Run from the repo root:

    uv run python scripts/generate_vi.py \\
        --speaker-wav default_voices/AmericanFemale.mp3 \\
        --text "Hôm nay là 24/9/2026, nhiệt độ 31,5 độ, giá 1.250.000 đ." \\
        --out out/vi.wav

    # one line per utterance, batched in a single call
    uv run python scripts/generate_vi.py --speaker-wav ref.wav --text-file lines.txt --out out/vi_%03d.wav

    # fine-tuned checkpoint directory (model.pth + params.json) instead of the HF id
    uv run python scripts/generate_vi.py --model-path runs/my_voice --speaker-wav ref.wav --text "..." --out out.wav

    # only print the normalized text, no model loading
    uv run python scripts/generate_vi.py --text "Số 2 và 105" --print-normalized
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- Vietnamese text normalization
_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

ABBREVIATIONS = {
    r"\bTP\.?\s*HCM\b": "thành phố Hồ Chí Minh",
    r"\bTP\.": "thành phố",
    r"\bHN\b": "Hà Nội",
    r"\bUBND\b": "ủy ban nhân dân",
    r"\bVN\b": "Việt Nam",
    r"\bTS\.": "tiến sĩ",
    r"\bThS\.": "thạc sĩ",
    r"\bGS\.": "giáo sư",
    r"\bPGS\.": "phó giáo sư",
    r"\bBS\.": "bác sĩ",
    r"\bv\.v\.": "vân vân",
    r"%": " phần trăm",
    r"\bkm\b": "ki lô mét",
    r"\bkg\b": "ki lô gam",
    r"\bcm\b": "xen ti mét",
    r"\bmm\b": "mi li mét",
    r"\bUSD\b": "đô la Mỹ",
    r"\bVNĐ\b|\bVND\b|\bđ\b": "đồng",
}


def _read_three(n: int, full: bool) -> str:
    """Read a 0..999 group; ``full`` forces leading zeros ("không trăm ...")."""
    hundreds, rest = divmod(n, 100)
    tens, units = divmod(rest, 10)
    words = []
    if hundreds or full:
        words += [_DIGITS[hundreds], "trăm"]
    if tens == 0:
        if units:
            if hundreds or full:
                words.append("lẻ")
            words.append(_DIGITS[units])
    elif tens == 1:
        words.append("mười")
        if units == 5:
            words.append("lăm")
        elif units:
            words.append(_DIGITS[units])
    else:
        words += [_DIGITS[tens], "mươi"]
        if units == 1:
            words.append("mốt")
        elif units == 4:
            words.append("tư")
        elif units == 5:
            words.append("lăm")
        elif units:
            words.append(_DIGITS[units])
    return " ".join(words)


def number_to_vietnamese(n: int) -> str:
    if n == 0:
        return "không"
    if n < 0:
        return "âm " + number_to_vietnamese(-n)
    groups = []
    while n > 0:
        groups.append(n % 1000)
        n //= 1000
    scales = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ"]
    parts = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        text = _read_three(g, full=(i != len(groups) - 1))
        if scales[i]:
            text += " " + scales[i]
        parts.append(text)
    return " ".join(parts)


def _replace_number(match: re.Match) -> str:
    raw = match.group(0)
    s = raw.replace(".", "") if re.fullmatch(r"\d{1,3}(\.\d{3})+", raw) else raw
    if "," in s:  # Vietnamese decimal comma
        int_part, frac = s.split(",", 1)
        return number_to_vietnamese(int(int_part)) + " phẩy " + " ".join(_DIGITS[int(c)] for c in frac if c.isdigit())
    return number_to_vietnamese(int(s))


def normalize_vietnamese(text: str) -> str:
    """Lightweight Vietnamese written-to-spoken normalization; uses vinorm first if installed."""
    try:
        from vinorm import TTSnorm  # type: ignore

        text = TTSnorm(text, punc=False, unknown=False, lower=False, rule=False)
    except Exception:
        pass
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    for pat, rep in ABBREVIATIONS.items():
        text = re.sub(pat, rep, text)
    text = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", lambda m: f"ngày {m.group(1)} tháng {m.group(2)} năm {m.group(3)}", text)
    text = re.sub(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+,\d+|\d+", _replace_number, text)
    text = text.replace('"', "").replace("“", "").replace("”", "")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Vietnamese TTS with ZONOS2 (offline API)", formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="text to synthesize")
    src.add_argument("--text-file", help="UTF-8 file, one utterance per non-empty line (batched)")
    ap.add_argument("--out", default="out/vi.wav", help="output wav; with --text-file use a %%d pattern, e.g. out/vi_%%03d.wav")
    ap.add_argument("--model-path", default="Zyphra/ZONOS2", help="HF id or local checkpoint dir (e.g. a fine-tuned runs/<name>)")
    ap.add_argument("--speaker-wav", default=None, help="reference clip for voice cloning (3-15 s, one speaker); omit for the model's default voice")
    ap.add_argument("--speaker-npy", default=None, help="precomputed speaker embedding (.npy) instead of --speaker-wav")
    ap.add_argument("--clean-background", action="store_true", help="mark the reference as clean-background (default: noisy, like the server)")
    ap.add_argument("--expressive", action="store_true", help="expressive mode (accurate_mode=False); default is accurate mode")
    ap.add_argument("--speed", type=float, default=None, help="speaking-rate multiplier, 1.0 = neutral")
    ap.add_argument("--trailing-silence", type=float, default=None, help="quality value trailing_silence_s (server default 3)")
    ap.add_argument("--max-tokens", type=int, default=None, help="max audio frames (~86 per second); default = model limit")
    ap.add_argument("--no-normalize", action="store_true", help="send the text as-is")
    ap.add_argument("--print-normalized", action="store_true", help="print normalized text and exit without loading the model")

    g = ap.add_argument_group("sampling (server defaults)")
    g.add_argument("--temperature", type=float, default=1.15)
    g.add_argument("--topk", type=int, default=106)
    g.add_argument("--top-p", type=float, default=0.0)
    g.add_argument("--min-p", type=float, default=0.18)
    g.add_argument("--repetition-penalty", type=float, default=1.2)
    g.add_argument("--repetition-window", type=int, default=50)
    g.add_argument("--seed", type=int, default=None)
    return ap.parse_args()


def load_texts(a: argparse.Namespace) -> list[str]:
    if a.text is not None:
        return [a.text.strip()]
    lines = Path(a.text_file).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def main() -> None:
    a = parse_args()
    raw = load_texts(a)
    texts = raw if a.no_normalize else [normalize_vietnamese(t) for t in raw]
    for r, t in zip(raw, texts):
        print(f"[generate_vi] text      : {r[:160]}")
        print(f"[generate_vi] normalized: {t[:160]}")
    if a.print_normalized:
        return
    if len(texts) > 1 and "%" not in a.out:
        sys.exit("--text-file has several lines: give --out a pattern such as out/vi_%03d.wav")

    import torch

    from zonos2.message import TTSSamplingParams
    from zonos2.tts import TTSLLM

    t0 = time.time()
    tts = TTSLLM(model_path=a.model_path)
    print(f"[generate_vi] model loaded from {a.model_path} in {time.time() - t0:.1f}s")

    speaker = None
    if a.speaker_npy:
        import numpy as np

        speaker = torch.from_numpy(np.load(a.speaker_npy)).float()
        print(f"[generate_vi] speaker embedding from {a.speaker_npy}: dim {speaker.numel()}")
    elif a.speaker_wav:
        speaker = tts.embed_speaker_file(a.speaker_wav)
        print(f"[generate_vi] speaker embedding from {a.speaker_wav}: dim {speaker.numel()}")

    sp = TTSSamplingParams(
        temperature=a.temperature, topk=a.topk, top_p=a.top_p, min_p=a.min_p,
        repetition_penalty=a.repetition_penalty, repetition_window=a.repetition_window,
        seed=a.seed,
    )
    quality_values = {"trailing_silence_s": a.trailing_silence} if a.trailing_silence is not None else None

    t1 = time.time()
    results = tts.generate(
        texts, sp,
        language="en_us",            # ignored: normalization is off, Vietnamese is handled above
        text_normalization=False,
        speaker_embedding=speaker,
        clean_speaker_background=a.clean_background,
        accurate_mode=not a.expressive,
        speed=a.speed,
        quality_values=quality_values,
        max_tokens=a.max_tokens,
    )
    elapsed = time.time() - t1

    total_s = 0.0
    for i, res in enumerate(results):
        out = Path(a.out % i) if "%" in a.out else Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tts.save_audio(res["audio"], str(out))
        secs = len(res["audio"]) / 4 / 44100  # float32 PCM at 44.1 kHz
        total_s += secs
        print(f"[generate_vi] saved {out}: {secs:.2f}s, {len(res['audio_tokens'])} frames, eos_frame={res['eos_frame']}")
    print(f"[generate_vi] {len(results)} utterance(s), {total_s:.1f}s audio in {elapsed:.1f}s (RTF {elapsed / max(total_s, 1e-6):.2f})")


if __name__ == "__main__":
    main()
