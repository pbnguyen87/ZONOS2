"""Fine-tune Zonos2 on preprocessed data (see finetune/preprocess.py).

LoRA (default) or full fine-tuning of the trainable mirror model; checkpoints are
saved merged, in the exact release format, so the output directory is directly
servable:

  python -m zonos2 --model-path <output-dir>

Examples:
  # LoRA fine-tune on a speaker dataset
  python finetune/train.py --model-path Zyphra/ZONOS2 \
      --data data/preprocessed --output-dir runs/my_voice \
      --batch-size 2 --grad-accum 8 --epochs 3

  # Full fine-tune (needs much more VRAM)
  python finetune/train.py --model-path Zyphra/ZONOS2 \
      --data data/preprocessed --output-dir runs/full --full --lr 1e-5

  # CPU smoke test with a tiny random model + synthetic data (no checkpoint needed)
  python finetune/train.py --smoke
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import DEFAULT_QUALITY, IGNORE_INDEX, PromptSpec, Zonos2FinetuneDataset, collate
from lora import DEFAULT_TARGETS, apply_lora, export_merged_state_dict, lora_state_dict
from model import Zonos2Trainable, tts_cross_entropy
from zonos2_compat import load_config, save_servable_checkpoint

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", help="Checkpoint dir or HF repo id (e.g. Zyphra/ZONOS2)")
    ap.add_argument("--data", help="Directory of preprocessed .pt files")
    ap.add_argument("--output-dir", default="runs/finetune")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = use --epochs")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-4 LoRA, 1e-5 full")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--max-frames", type=int, default=2048,
                    help="Skip samples whose sequence would exceed this many frames")
    ap.add_argument("--save-every", type=int, default=0, help="Steps between merged saves (0 = end only)")
    ap.add_argument("--log-every", type=int, default=10)

    # LoRA
    ap.add_argument("--full", action="store_true", help="Full fine-tune instead of LoRA")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", nargs="+", default=list(DEFAULT_TARGETS))

    # Conditioning used to build training prompts (mirrors server defaults)
    ap.add_argument("--clean-background", action="store_true",
                    help="Use the 'clean' background marker (server default is noisy)")
    ap.add_argument("--expressive", action="store_true",
                    help="Omit the accurate-mode marker (server default includes it)")
    ap.add_argument("--no-quality", action="store_true",
                    help="Omit the default trailing_silence_s quality row")

    ap.add_argument("--smoke", action="store_true", help="Tiny random model + data, one short run")
    args = ap.parse_args()
    if not args.smoke and (not args.model_path or not args.data):
        ap.error("--model-path and --data are required (or use --smoke)")
    return args


def lr_lambda_factory(warmup: int, total: int):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return fn


def train_loop(model, loader, args, save_fn) -> None:
    device = torch.device(args.device)
    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    steps_per_epoch = max(len(loader) // args.grad_accum, 1)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(args.warmup_steps, total_steps)
    )

    step = 0
    micro = 0
    running_loss = 0.0
    t0 = time.time()
    done = False
    for epoch in range(args.epochs if not args.max_steps else 10**9):
        if done:
            break
        for batch in loader:
            frames = batch["frames"].to(device)
            labels = batch["labels"].to(device)
            speaker = batch.get("speaker_embedding")
            spk_pos = batch.get("speaker_positions")
            if speaker is not None:
                speaker = speaker.to(device)
                spk_pos = spk_pos.to(device)

            logits = model(frames, speaker_embedding=speaker, speaker_positions=spk_pos)
            loss = tts_cross_entropy(logits, labels, IGNORE_INDEX)
            (loss / args.grad_accum).backward()
            running_loss += loss.item()
            micro += 1

            if micro % args.grad_accum == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0 or step == total_steps:
                    avg = running_loss / (args.log_every * args.grad_accum)
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"epoch {epoch} step {step}/{total_steps} "
                        f"loss {avg:.4f} lr {lr_now:.2e} "
                        f"({time.time() - t0:.0f}s)"
                    )
                    running_loss = 0.0
                if args.save_every and step % args.save_every == 0:
                    save_fn(model, tag=f"step{step}")
                if step >= total_steps:
                    done = True
                    break

    save_fn(model, tag="final")


def run_real(args) -> None:
    torch.manual_seed(args.seed)
    dtype = DTYPES[args.dtype]

    config = load_config(args.model_path)
    print(f"Loading {args.model_path} "
          f"(layers={config.n_layers}, dim={config.dim}, experts={config.moe_n_experts})")
    model = Zonos2Trainable.from_pretrained(args.model_path, dtype=dtype)
    model.gradient_checkpointing = args.grad_checkpoint

    if args.full:
        model.requires_grad_(True)
        args.lr = args.lr or 1e-5
    else:
        adapted = apply_lora(
            model, r=args.lora_r, alpha=args.lora_alpha,
            dropout=args.lora_dropout, targets=args.lora_targets,
        )
        args.lr = args.lr or 1e-4
        print(f"LoRA r={args.lora_r} on {len(adapted)} modules "
              f"({args.lora_targets})")
    trainable = model.num_parameters(trainable_only=True)
    print(f"Trainable parameters: {trainable / 1e6:.1f}M / {model.num_parameters() / 1e6:.1f}M")

    spec = PromptSpec(
        config,
        quality=None if args.no_quality else DEFAULT_QUALITY,
        clean_background=args.clean_background,
        accurate_mode=not args.expressive,
    )
    dataset = Zonos2FinetuneDataset(args.data, spec, max_frames=args.max_frames)
    print(f"Dataset: {len(dataset)} samples ({dataset._skipped} skipped over max-frames)")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate(b, spec),
        drop_last=len(dataset) > args.batch_size,
    )

    out_dir = Path(args.output_dir)

    def save_fn(m, tag: str) -> None:
        target = out_dir if tag == "final" else out_dir / tag
        sd = export_merged_state_dict(m)
        save_servable_checkpoint(sd, args.model_path, target)
        if not args.full:
            torch.save(lora_state_dict(m), Path(target) / "lora_adapter.pt")
        print(f"Saved servable checkpoint to {target}")

    train_loop(model, loader, args, save_fn)
    print(f"\nDone. Serve it with:\n  python -m zonos2 --model-path {out_dir}")


# ---------------------------------------------------------------------------
# Smoke test: tiny random model + synthetic data, verifies the whole train path
# ---------------------------------------------------------------------------


def run_smoke(args) -> None:
    from types import SimpleNamespace

    torch.manual_seed(0)
    config = SimpleNamespace(
        n_layers=4, dim=64, head_dim=16, n_heads=4, n_kv_heads=2,
        ffn_dim_multiplier=2.0, multiple_of=32, norm_eps=1e-5,
        rope_theta=10000.0, max_seqlen=512,
        n_codebooks=9, codebook_size=1024, eoa_id=1024, audio_pad_id=1025,
        text_vocab=469, loss_softcap=15.0,
        speaker_enabled=True, speaker_embedding_dim=32, speaker_lda_dim=16,
        speaker_background_token_enabled=True, accurate_mode_token_enabled=True,
        speaking_rate_num_buckets=8,
        quality_features=["trailing_silence_s"],
        quality_buckets={"trailing_silence_s": [str(i) for i in range(8)]},
        moe_n_experts=4, moe_router_topk=2, moe_router_dim=32,
        moe_start_from_layer=1, moe_end_from_layer=1,
        moe_balancing_strategy="legacy", special_topk_layers=None,
    )
    # text_vocab consistency: 448 bytes + 8 rate + 8 quality + 2 bg + 1 accurate = 467
    config.text_vocab = 448 + 8 + 8 + 2 + 1

    model = Zonos2Trainable(config).float()
    for p in model.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02)
    adapted = apply_lora(model, r=4, alpha=8.0)
    print(f"smoke: LoRA on {len(adapted)} modules, "
          f"{model.num_parameters(trainable_only=True)} trainable params")

    spec = PromptSpec(config, clean_background=True, accurate_mode=True)
    codes = torch.randint(0, 1024, (40, 9))
    ex1 = spec.build_example("Hello world.", codes)
    ex1["speaker_embedding"] = torch.randn(32)
    ex2 = spec.build_example("Xin chào!", torch.randint(0, 1024, (25, 9)))
    ex2["speaker_embedding"] = torch.randn(32)
    batch = collate([ex1, ex2], spec)
    print(f"smoke: frames {tuple(batch['frames'].shape)}, labels {tuple(batch['labels'].shape)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    first = last = None
    for i in range(8):
        logits = model(
            batch["frames"],
            speaker_embedding=batch["speaker_embedding"],
            speaker_positions=batch["speaker_positions"],
        )
        loss = tts_cross_entropy(logits, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        first = first if first is not None else loss.item()
        last = loss.item()
        print(f"smoke step {i}: loss {loss.item():.4f}")
    assert last < first, "loss did not decrease"

    sd = export_merged_state_dict(model)
    expected = {
        "multi_embedder.embedders.0.weight",
        "layers.0.attention.wq.weight",
        "layers.0.attention.wkv.weight",
        "layers.0.attention.temp",
        "layers.0.attention.gater.weight",
        "layers.0.attention_norm.weight",
        "layers.0.feed_forward.w_in.weight",
        "layers.0.feed_forward.w_out.weight",
        "layers.1.feed_forward.router.down_proj.weight",
        "layers.1.feed_forward.router.router_mlp.0.weight",
        "layers.1.feed_forward.router.router_mlp.4.weight",
        "layers.1.feed_forward.router.rmsnorm_eda.weight",
        "layers.1.feed_forward.router.balancing_biases",
        "layers.1.feed_forward.experts.gate_up_proj",
        "layers.1.feed_forward.experts.down_proj",
        "layers.2.feed_forward.router.router_states_scale",
        "speaker_lda_projection.weight",
        "speaker_projection.bias",
        "out_norm.weight",
        "multi_output.weight",
    }
    missing = expected - set(sd)
    assert not missing, f"exported state dict missing keys: {missing}"
    assert not any(".lora_" in k or ".base." in k for k in sd), "unmerged LoRA keys leaked"

    # Round-trip: merged dict loads into a fresh model strictly.
    fresh = Zonos2Trainable(config).float()
    fresh.load_state_dict(sd, strict=True)
    print(f"smoke: exported {len(sd)} checkpoint keys, round-trip load OK")
    print("SMOKE TEST PASSED")


def main() -> None:
    args = parse_args()
    if args.smoke:
        run_smoke(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
