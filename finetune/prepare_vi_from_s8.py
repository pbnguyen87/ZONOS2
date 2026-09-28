"""Build ZONOS2 fine-tuning data (Vietnamese) from an audio-pipeline ``s8_package`` folder.

``s8`` is the packaging stage of speech_dataset/audio-pipeline: every segment already
carries a quality tier (A / B / C) and a speaker-disjoint split, and the kept tiers are
exported as a self-contained dataset::

    s8_package/
    ├── dataset/
    │   ├── metadata.csv     file_name, text, text_normalized, speaker_id, duration, tier,
    │   │                    snr_db, dnsmos, cer, clipping, bandwidth_hz, multi_speaker,
    │   │                    source_path, split
    │   └── wav/{train,val,test}/*.wav
    └── manifest.jsonl       every segment incl. tier C (audio still under s7_loudnorm)

This script reads ``dataset/metadata.csv`` (works on a machine that only has the packaged
dataset), keeps the requested tiers, normalizes Vietnamese text, splits train/val, writes
the JSONL manifests ``finetune/preprocess.py`` reads, and optionally runs the preprocessing
into ``.pt`` tensors. Shared helpers come from ``prepare_vi_from_s7.py``.

Differences from the s7 script
    * No per-metric thresholds: the tier already encodes them. Select with ``--tiers``.
    * ``--use-pipeline-split`` keeps s8's speaker-disjoint split (val/test -> val);
      the default re-splits inside each speaker, which is what a voice fine-tune wants.
    * ``--from-manifest`` reads ``manifest.jsonl`` instead (needs the s7 audio present),
      useful to pull tier B rows that s8 did not export.

Examples::

    uv run python finetune/prepare_vi_from_s8.py \\
        --s8-dir ../speech_dataset/audio-pipeline/work_5min/s8_package \\
        --out-dir data/zonos2_vi --dry-run

    uv run python finetune/prepare_vi_from_s8.py --s8-dir ... --out-dir data/zonos2_vi \\
        --tiers A,B --run-preprocess --model-path Zyphra/ZONOS2 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_vi_from_s7 import (  # noqa: E402
    REPO_ROOT,
    clean_text,
    hours,
    normalize_vietnamese,
    read_manifest,
    run_preprocess,
    split_per_speaker,
    write_jsonl,
)

log = logging.getLogger("prepare_vi_from_s8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s8-dir", required=True, help="path to <workdir>/s8_package")
    ap.add_argument("--out-dir", required=True, help="output folder, e.g. data/zonos2_vi")
    ap.add_argument("--tiers", default="A", help="comma-separated tiers to keep, e.g. A or A,B (default A)")
    ap.add_argument("--language", default="vi")
    ap.add_argument("--text-field", default="text_normalized", help="metadata column used as text")
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, default=20.0,
                    help="ZONOS2 limit: ~23 s fits train.py --max-frames 2048 including the prompt rows")
    ap.add_argument("--min-utts-per-speaker", type=int, default=5)
    ap.add_argument("--use-pipeline-split", action="store_true",
                    help="keep s8's split (train->train, val/test->val) instead of re-splitting per speaker")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from-manifest", action="store_true",
                    help="read s8_package/manifest.jsonl (needs s7 audio) instead of dataset/metadata.csv")
    ap.add_argument("--pipeline-root", default=None,
                    help="with --from-manifest: folder audio_path is relative to (default: two levels above --s8-dir)")

    p = ap.add_argument_group("preprocess (optional)")
    p.add_argument("--run-preprocess", action="store_true",
                   help="also run finetune/preprocess.py's Preprocessor into <out-dir>/train and <out-dir>/val")
    p.add_argument("--model-path", default="Zyphra/ZONOS2")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument("--overwrite", action="store_true")

    ap.add_argument("--no-normalize", action="store_true", help="keep the text as-is")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


# --------------------------------------------------------------------------- loaders
def _to_float(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    v = str(v).strip()
    if v == "" or v.lower() in ("nan", "none", "null"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _to_bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")


def load_metadata_csv(s8_dir: Path, text_field: str) -> List[dict]:
    ds = s8_dir / "dataset"
    csv_path = ds / "metadata.csv"
    if not csv_path.is_file():
        sys.exit(f"metadata.csv not found: {csv_path}")
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            wav = ds / rec.get("file_name", "")
            rows.append({
                "id": Path(rec.get("file_name", "")).stem,
                "wav_abs": wav.resolve(),
                "wav_exists": wav.is_file(),
                "speaker_id": rec.get("speaker_id") or None,
                "duration": _to_float(rec.get("duration")),
                "tier": (rec.get("tier") or "").strip().upper(),
                "split": (rec.get("split") or "train").strip().lower(),
                "cer": _to_float(rec.get("cer")),
                "text_raw": clean_text(rec.get(text_field) or rec.get("text")),
            })
    return rows


def load_manifest(s8_dir: Path, pipeline_root: Path, text_field: str) -> List[dict]:
    path = s8_dir / "manifest.jsonl"
    if not path.is_file():
        sys.exit(f"manifest not found: {path}")
    rows = []
    for r in read_manifest(path):
        p = Path(r.get("audio_path", ""))
        cands = [p] if p.is_absolute() else [pipeline_root / p, s8_dir.parent / p]
        cands.append(s8_dir.parent / "s7_loudnorm" / "audio" / f"{r.get('id')}.wav")
        wav = next((c for c in cands if c.is_file()), None)
        rows.append({
            "id": r.get("id"),
            "wav_abs": wav.resolve() if wav else None,
            "wav_exists": wav is not None,
            "speaker_id": r.get("speaker_id"),
            "duration": _to_float(r.get("duration")),
            "tier": (r.get("tier") or "").upper(),
            "split": (r.get("split") or "train").lower(),
            "cer": _to_float(r.get("cer")),
            "text_raw": clean_text(r.get(text_field) or r.get("text")),
        })
    return rows


def split_from_field(rows: List[dict]) -> Dict[str, List[dict]]:
    out = {"train": [], "val": []}
    for r in rows:
        (out["train"] if r.get("split", "train") == "train" else out["val"]).append(r)
    for k in out:
        out[k].sort(key=lambda r: r["id"])
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    a = parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    s8_dir = Path(a.s8_dir).expanduser().resolve()
    out_dir = Path(a.out_dir).expanduser()
    out_dir = out_dir.resolve() if out_dir.is_absolute() else (REPO_ROOT / out_dir).resolve()
    keep = {t.strip().upper() for t in a.tiers.split(",") if t.strip()}
    if "C" in keep:
        log.warning("tier C is the pipeline's reject tier; including it is not recommended")

    if a.from_manifest:
        pipeline_root = Path(a.pipeline_root).resolve() if a.pipeline_root else s8_dir.parent.parent
        rows = load_manifest(s8_dir, pipeline_root, a.text_field)
        source = str(s8_dir / "manifest.jsonl")
    else:
        rows = load_metadata_csv(s8_dir, a.text_field)
        source = str(s8_dir / "dataset" / "metadata.csv")
    log.info("%s: %d records, tiers %s", source, len(rows), dict(Counter(r["tier"] for r in rows)))

    reasons: Counter = Counter()
    accepted: List[dict] = []
    for r in rows:
        if r["tier"] not in keep:
            reasons[f"tier_{r['tier'] or 'missing'}"] += 1
            continue
        if not r["wav_exists"]:
            reasons["audio_missing"] += 1
            continue
        if not r["text_raw"]:
            reasons["empty_text"] += 1
            continue
        if not r["speaker_id"]:
            reasons["speaker_missing"] += 1
            continue
        dur = r["duration"]
        if dur is None:
            reasons["duration_missing"] += 1
            continue
        if dur < a.min_seconds:
            reasons["too_short"] += 1
            continue
        if dur > a.max_seconds:
            reasons["too_long"] += 1
            continue
        r["norm_text"] = r["text_raw"] if a.no_normalize else normalize_vietnamese(r["text_raw"])
        if not r["norm_text"]:
            reasons["empty_after_normalize"] += 1
            continue
        accepted.append(r)
    log.info("tier filter (%s): %d accepted, rejected: %s", ",".join(sorted(keep)), len(accepted), dict(reasons))

    by_spk = Counter(r["speaker_id"] for r in accepted)
    small = {s for s, n in by_spk.items() if n < a.min_utts_per_speaker}
    if small:
        n_drop = sum(by_spk[s] for s in small)
        reasons["speaker_too_few_utts"] += n_drop
        accepted = [r for r in accepted if r["speaker_id"] not in small]
        log.info("dropped %d speakers with <%d utts (%d utterances)", len(small), a.min_utts_per_speaker, n_drop)

    splits = split_from_field(accepted) if a.use_pipeline_split else split_per_speaker(accepted, a.val_ratio, a.seed)
    summary = {
        "source": source,
        "records": len(rows),
        "tiers_kept": sorted(keep),
        "accepted": len(accepted),
        "rejected": dict(reasons),
        "speakers": len({r["speaker_id"] for r in accepted}),
        "hours_total": hours(accepted),
        "train": {"utts": len(splits["train"]), "hours": hours(splits["train"])},
        "val": {"utts": len(splits["val"]), "hours": hours(splits["val"])},
        "utts_per_speaker": dict(sorted(Counter(r["speaker_id"] for r in accepted).items())),
        "split_mode": "pipeline_split_field" if a.use_pipeline_split else "per_speaker",
        "max_seconds": a.max_seconds,
        "normalized": not a.no_normalize,
    }
    if accepted:
        ex = accepted[0]
        log.info("example: %r -> %r", ex["text_raw"][:90], ex["norm_text"][:90])
    if a.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    if not accepted:
        sys.exit("nothing accepted; add tiers with --tiers A,B or check the input")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", splits["train"], a.language)
    write_jsonl(out_dir / "val.jsonl", splits["val"], a.language)
    log.info("wrote %s (%d) and %s (%d)", out_dir / "train.jsonl", len(splits["train"]), out_dir / "val.jsonl", len(splits["val"]))

    if a.run_preprocess:
        import torch

        from preprocess import Preprocessor

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
