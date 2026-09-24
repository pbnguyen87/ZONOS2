# Zonos2 fine-tuning

Fine-tune a ZONOS2 checkpoint (LoRA or full) and save it back in the exact release
format, so the output directory is directly servable by the existing inference
stack — no changes to `python/zonos2/` are needed.

## Why a separate model implementation?

The inference model (`python/zonos2/models/zonos2.py`) runs on flashinfer kernels,
a paged KV cache, and an ambient batch context; none of that supports backprop.
`finetune/model.py` is a trainable `nn.Module` mirror of the same math
(fused wkv, QK-norm with learnable `|temp|`, per-head sigmoid gating, interleaved
RoPE, dense-[h,gate] vs MoE-[gate,up] FFN ordering, EDA router with bias-aware
top-k, softcapped multi-codebook head) with **state-dict keys identical to the
checkpoint**, so weights round-trip losslessly:

- `Zonos2Trainable.from_pretrained("Zyphra/ZONOS2")` loads `model.pth` directly
  (including un-fusing `w1/w3`- or SonicMoE `w13`-format expert weights).
- Saved checkpoints contain `model.pth` + `params.json` and load with
  `python -m zonos2 --model-path <output-dir>` or `TTSLLM(model_path=...)`.

Training sequences reproduce the inference prompt layout
(`[speaker slot][background marker][accurate marker][conditioning rows][BOS bytes EOS][sheared audio]`),
including the 17-frame silence run-up, the codebook delay ("shear") pattern, and
the delayed EOA tail — so what the model sees in training is exactly what the
scheduler feeds it at inference.

## Requirements

The repo's own environment (`uv sync`) has everything: torch, torchaudio,
`descript-audio-codec` (DAC), transformers (for the Qwen3 speaker encoder).
Training imports **do not** require flashinfer/sgl_kernel — the pure zonos2
modules (prompt building, speaker encoder, text norm) are loaded by file path.
A single ≥24 GB GPU is enough for LoRA with gradient checkpointing; full
fine-tuning needs far more (all MoE experts are dense-materialized).

## 1. Prepare data

A JSONL manifest, one utterance per line:

```json
{"audio": "clips/utt1.wav", "text": "Hello there.", "language": "en_us"}
```

or a directory of audio files with same-stem `.txt` transcripts. Then:

```bash
uv run python finetune/preprocess.py \
    --model-path Zyphra/ZONOS2 \
    --manifest data/train.jsonl \
    --output-dir data/preprocessed \
    --device cuda
```

This caches, per utterance: DAC codes (44.1 kHz, 9 codebooks), a Qwen3 speaker
embedding computed from the utterance itself (self-cloning — the standard recipe
for voice fine-tuning), and NeMo-normalized text (raw text fallback, matching
server behavior). Keep clips under ~30 s (`--max-seconds`).

## 2. Train

```bash
# LoRA (default): attention + dense FFN projections
uv run python finetune/train.py \
    --model-path Zyphra/ZONOS2 \
    --data data/preprocessed \
    --output-dir runs/my_voice \
    --batch-size 2 --grad-accum 8 --epochs 3 --grad-checkpoint

# Full fine-tune
uv run python finetune/train.py --model-path Zyphra/ZONOS2 \
    --data data/preprocessed --output-dir runs/full --full --lr 1e-5
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--lora-r / --lora-alpha / --lora-targets` | LoRA rank/scale/modules (`wq wkv wo w_in w_out`; add `multi_output`, `speaker_projection`, `gater` to adapt those too) |
| `--max-frames` | skip samples whose sequence exceeds this (default 2048, the model's training context) |
| `--clean-background` | use the *clean* background marker (server default is noisy) |
| `--expressive` | omit the accurate-mode marker |
| `--no-quality` | omit the default `trailing_silence_s` quality row |
| `--save-every N` | periodic merged saves to `<output-dir>/stepN/` |
| `--dtype float32` | full-precision training (default bfloat16) |

Loss is next-frame cross-entropy over all 9 codebooks; prompt/silence positions
and shear-fill padding cells are masked, EOA cells are kept so the model learns
to stop.

## 3. Serve the result

```bash
uv run python -m zonos2 --model-path runs/my_voice --tts-default-voices-dir ./default_voices/
```

LoRA runs also drop a small `lora_adapter.pt` next to the merged checkpoint.

## Smoke test (no GPU / no checkpoint)

```bash
python finetune/train.py --smoke
```

Builds a tiny random model, runs the full pipeline (prompt building → shear →
LoRA training steps → merged export → strict round-trip reload) on CPU and
asserts the loss decreases and every checkpoint key survives.

## Design notes / caveats

- **Frozen during fine-tuning**: MoE `balancing_biases` (a training-time
  load-balancing artifact; kept fp32) — and with LoRA everything except the
  adapters. No auxiliary load-balancing loss is applied; for small fine-tunes
  the frozen biases keep routing close to the base model. Watch expert collapse
  if you do large full fine-tunes.
- The EDA router state threads across MoE layers exactly as in inference
  (cloned pre-norm; dense layers reset it).
- Speaker conditioning is injected by overwriting the hidden state at the
  reserved slot (position 0), after the optional LDA + speaker projection —
  identical to `zonos2.py:849-873`.
- `torch.nn.functional.scaled_dot_product_attention(..., enable_gqa=True)` is
  used for GQA; batches are right-padded, so causal masking alone is correct
  (pad rows are loss-masked).
- Top-k expert selection is non-differentiable (as in the reference); gradients
  reach the router through the gathered pre-bias probabilities.
