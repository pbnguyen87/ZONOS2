"""Build ZONOS2 fine-tuning data (Vietnamese) from an audio-pipeline ``s7_loudnorm`` folder.

Input : ``<workdir>/s7_loudnorm/manifest.jsonl`` from speech_dataset/audio-pipeline
        (one record per segment: id, audio_path, speaker_id, multi_speaker, duration,
        text_normalized, cer, snr_db, dnsmos, clipping, ...).
Output: ``<out-dir>/train.jsonl`` and ``<out-dir>/val.jsonl`` in the format
        ``finetune/preprocess.py`` reads::

            {"audio": "/abs/path.wav", "text": "<normalized Vietnamese>", "language": "vi"}

        and, with ``--run-preprocess``, the ``.pt`` tensors under ``<out-dir>/train`` and
        ``<out-dir>/val`` ready for ``finetune/train.py --data <out-dir>/train``.

Steps
    1. read the manifest, resolve audio paths, drop rows whose audio is missing
    2. filter with the pipeline's tier-A rule (defaults from audio-pipeline/config/default.yaml)
       plus ZONOS2's own length limit (--max-seconds, default 20 s so a full sequence
       stays under train.py's --max-frames 2048 at 86 frames/s)
    3. normalize Vietnamese text (numbers, dates, currency, abbreviations) with the same
       normalizer as scripts/generate_vi.py; ZONOS2's NeMo normalizer has no Vietnamese
    4. split train / val inside every speaker (the goal is to learn these voices)
    5. optionally run preprocess.py's Preprocessor with text normalization disabled

Example::

    uv run python finetune/prepare_vi_from_s7.py \\
        --s7-dir ../speech_dataset/audio-pipeline/work_5min/s7_loudnorm \\
        --out-dir data/zonos2_vi --dry-run

    uv run python finetune/prepare_vi_from_s7.py --s7-dir ... --out-dir data/zonos2_vi \\
        --run-preprocess --model-path Zyphra/ZONOS2 --device cuda

    uv run python finetune/train.py --model-path Zyphra/ZONOS2 --data data/zonos2_vi/train \\
        --output-dir runs/vi_podcast --batch-size 2 --grad-accum 8 --epochs 3 --grad-checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "finetune"))

from generate_vi import normalize_vietnamese  # noqa: E402  (scripts/generate_vi.py)

DAC_FRAME_RATE = 44_100 / 512  # 86.13 frames per second
log = logging.getLogger("prepare_vi_from_s7")


# --------------------------------------------------------------------------- args
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s7-dir", required=True, help="path to <workdir>/s7_loudnorm")
    ap.add_argument("--out-dir", required=True, help="output folder, e.g. data/zonos2_vi")
    ap.add_argument("--pipeline-root", default=None,
                    help="folder that manifest audio_path is relative to (default: two levels above --s7-dir)")
    ap.add_argument("--language", default="vi", help="language tag written to the JSONL")
    ap.add_argument("--text-field", default="text_normalized", help="manifest field used as text")

    g = ap.add_argument_group("quality filter (pipeline tier A defaults)")
    g.add_argument("--max-cer", type=float, default=0.02)
    g.add_argument("--min-snr-db", type=float, default=20.0)
    g.add_argument("--min-dnsmos", type=float, default=3.0, help="used instead of SNR when dnsmos exists")
    g.add_argument("--max-clipping", type=float, default=0.001)
    g.add_argument("--min-seconds", type=float, default=2.0)
    g.add_argument("--max-seconds", type=float, default=20.0,
                   help="ZONOS2 limit: ~23 s fits train.py --max-frames 2048 including the prompt rows")
    g.add_argument("--allow-multi-speaker", action="store_true")
    g.add_argument("--min-utts-per-speaker", type=int, default=5)

    s = ap.add_argument_group("split")
    s.add_argument("--val-ratio", type=float, default=0.05)
    s.add_argument("--seed", type=int, default=42)

    p = ap.add_argument_group("preprocess (optional)")
    p.add_argument("--run-preprocess", action="store_true",
                   help="also run finetune/preprocess.py's Preprocessor into <out-dir>/train and <out-dir>/val")
    p.add_argument("--model-path", default="Zyphra/ZONOS2", help="checkpoint dir or HF id, needed for --run-preprocess")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument("--overwrite", action="store_true", help="recompute .pt files that already exist")

    ap.add_argument("--no-normalize", action="store_true", help="keep the manifest text as-is")
    ap.add_argument("--dry-run", action="store_true", help="filter and report only; write nothing")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


# --------------------------------------------------------------------------- helpers
def read_manifest(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("skip malformed line %d: %s", ln, e)
    return rows


def resolve_audio(row: dict, s7_dir: Path, pipeline_root: Path) -> Optional[Path]:
    p = row.get("audio_path")
    cands = []
    if p:
        pp = Path(p)
        cands += [pp] if pp.is_absolute() else [pipeline_root / pp, s7_dir.parent / pp]
    cands.append(s7_dir / "audio" / f"{row.get('id')}.wav")
    for c in cands:
        if c.is_file():
            return c.resolve()
    return None


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def reject_reason(row: dict, a: argparse.Namespace, text: str) -> Optional[str]:
    if not text:
        return "empty_text"
    cer = row.get("cer")
    if cer is None:
        return "cer_missing"
    if cer > a.max_cer:
        return "cer"
    dur = row.get("duration")
    if dur is None:
        return "duration_missing"
    if dur < a.min_seconds:
        return "too_short"
    if dur > a.max_seconds:
        return "too_long"
    clip = row.get("clipping")
    if clip is not None and clip > a.max_clipping:
        return "clipping"
    if row.get("multi_speaker") and not a.allow_multi_speaker:
        return "multi_speaker"
    dnsmos = row.get("dnsmos")
    if dnsmos is not None:
        if dnsmos < a.min_dnsmos:
            return "dnsmos"
    else:
        snr = row.get("snr_db")
        if snr is None or snr < a.min_snr_db:
            return "snr"
    return None


def split_per_speaker(rows: List[dict], val_ratio: float, seed: int) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    by_spk: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_spk[r["speaker_id"]].append(r)
    out = {"train": [], "val": []}
    for spk in sorted(by_spk):
        utts = sorted(by_spk[spk], key=lambda r: r["id"])
        rng.shuffle(utts)
        n = len(utts)
        n_val = max(1, int(round(n * val_ratio))) if n >= 10 else (1 if n >= 4 else 0)
        out["val"] += utts[:n_val]
        out["train"] += utts[n_val:]
    for k in out:
        out[k].sort(key=lambda r: r["id"])
    return out


def write_jsonl(path: Path, rows: List[dict], language: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "audio": str(r["wav_abs"]),
                "text": r["norm_text"],
                "language": language,
                "id": r["id"],
                "speaker_id": r["speaker_id"],
                "duration": r["duration"],
            }, ensure_ascii=False) + "\n")


def hours(rows: List[dict]) -> float:
    return round(sum(r["duration"] for r in rows) / 3600, 3)


# --------------------------------------------------------------------------- optional preprocess
def run_preprocess(rows: List[dict], out_dir: Path, a: argparse.Namespace, pre) -> int:
    """Mirror finetune/preprocess.py's main loop with normalization off (text is already spoken form)."""
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    index, n_new, n_skip = [], 0, 0
    for i, r in enumerate(rows):
        out_path = out_dir / f"{i:06d}_{r['id']}.pt"
        if out_path.exists() and not a.overwrite:
            index.append(out_path.name)
            continue
        try:
            item = pre.process_one(str(r["wav_abs"]), r["norm_text"], a.language)
        except Exception as exc:  # noqa: BLE001
            log.warning("failed on %s: %s", r["id"], exc)
            n_skip += 1
            continue
        frames = item["codes"].shape[0]
        if frames > 2000:  # leave headroom for the prompt rows under --max-frames 2048
            log.warning("%s: %d DAC frames (%.1fs), too long for max-frames 2048, skipped", r["id"], frames, frames / DAC_FRAME_RATE)
            n_skip += 1
            continue
        torch.save(item, out_path)
        index.append(out_path.name)
        n_new += 1
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            log.info("  %s: %d/%d", out_dir.name, i + 1, len(rows))
    (out_dir / "index.json").write_text(json.dumps({"files": index, "model_path": a.model_path}, indent=2), encoding="utf-8")
    log.info("%s: %d new, %d reused, %d skipped", out_dir, n_new, len(index) - n_new, n_skip)
    return len(index)


# --------------------------------------------------------------------------- main
def main() -> None:
    a = parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    s7_dir = Path(a.s7_dir).expanduser().resolve()
    manifest = s7_dir / "manifest.jsonl"
    if not manifest.is_file():
        sys.exit(f"manifest not found: {manifest}")
    pipeline_root = Path(a.pipeline_root).resolve() if a.pipeline_root else s7_dir.parent.parent
    out_dir = Path(a.out_dir).expanduser()
    out_dir = out_dir.resolve() if out_dir.is_absolute() else (REPO_ROOT / out_dir).resolve()

    # ---- 1-3: read, filter, normalize
    rows = read_manifest(manifest)
    log.info("manifest %s: %d records", manifest, len(rows))
    reasons: Counter = Counter()
    accepted: List[dict] = []
    for r in rows:
        wav = resolve_audio(r, s7_dir, pipeline_root)
        if wav is None:
            reasons["audio_missing"] += 1
            continue
        text = clean_text(r.get(a.text_field) or r.get("text"))
        why = reject_reason(r, a, text)
        if why:
            reasons[why] += 1
            continue
        if not r.get("speaker_id"):
            reasons["speaker_missing"] += 1
            continue
        rr = dict(r)
        rr["wav_abs"] = wav
        rr["norm_text"] = text if a.no_normalize else normalize_vietnamese(text)
        if not rr["norm_text"]:
            reasons["empty_after_normalize"] += 1
            continue
        accepted.append(rr)
    log.info("tier filter: %d accepted, rejected: %s", len(accepted), dict(reasons))

    by_spk = Counter(r["speaker_id"] for r in accepted)
    small = {s for s, n in by_spk.items() if n < a.min_utts_per_speaker}
    if small:
        n_drop = sum(by_spk[s] for s in small)
        reasons["speaker_too_few_utts"] += n_drop
        accepted = [r for r in accepted if r["speaker_id"] not in small]
        log.info("dropped %d speakers with <%d utts (%d utterances)", len(small), a.min_utts_per_speaker, n_drop)

    splits = split_per_speaker(accepted, a.val_ratio, a.seed)
    summary = {
        "source": str(manifest),
        "records": len(rows),
        "accepted": len(accepted),
        "rejected": dict(reasons),
        "speakers": len({r["speaker_id"] for r in accepted}),
        "hours_total": hours(accepted),
        "train": {"utts": len(splits["train"]), "hours": hours(splits["train"])},
        "val": {"utts": len(splits["val"]), "hours": hours(splits["val"])},
        "utts_per_speaker": dict(sorted(Counter(r["speaker_id"] for r in accepted).items())),
        "thresholds": {k: getattr(a, k) for k in
                       ("max_cer", "min_snr_db", "min_dnsmos", "max_clipping", "min_seconds", "max_seconds",
                        "allow_multi_speaker", "min_utts_per_speaker")},
        "normalized": not a.no_normalize,
    }
    if accepted:
        ex = accepted[0]
        log.info("example: %r -> %r", clean_text(ex.get(a.text_field))[:90], ex["norm_text"][:90])
    if a.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    if not accepted:
        sys.exit("nothing accepted; relax the thresholds or check the manifest")

    # ---- 4: manifests
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", splits["train"], a.language)
    write_jsonl(out_dir / "val.jsonl", splits["val"], a.language)
    log.info("wrote %s (%d) and %s (%d)", out_dir / "train.jsonl", len(splits["train"]), out_dir / "val.jsonl", len(splits["val"]))

    # ---- 5: optional preprocess into .pt
    if a.run_preprocess:
        import torch

        from preprocess import Preprocessor  # finetune/preprocess.py

        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading DAC + speaker encoder for %s on %s", a.model_path, device)
        pre = Preprocessor(a.model_path, device, normalize=False)
        summary["preprocessed"] = {
            "train": run_preprocess(splits["train"], out_dir / "train", a, pre),
            "val": run_preprocess(splits["val"], out_dir / "val", a, pre),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
