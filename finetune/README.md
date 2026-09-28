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

## Hướng dẫn fine-tune giọng tiếng Việt từ audio-pipeline

Quy trình đầy đủ, từ máy trắng đến checkpoint chạy được, cho dữ liệu đã qua
`speech_dataset/audio-pipeline` (đầu ra stage `s7_loudnorm`). Chạy trên Linux có
GPU NVIDIA, mọi lệnh từ gốc repo.

### Bước 0. Môi trường

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # nếu chưa có uv
uv python pin 3.12                                    # pynini 2.1.6 chỉ có wheel cho 3.10–3.12
uv sync
sudo apt-get install -y ffmpeg
uv pip install "torchcodec==0.8.*"                    # torchaudio 2.9 đọc file qua torchcodec
uv run python finetune/train.py --smoke               # kiểm tra pipeline train trên CPU, ~1 phút
```

Weights được tải tự động lần đầu vào `~/.cache/huggingface/hub/` (ZONOS2 ~16 GB,
speaker encoder ~3.4 GB) và `~/.cache/descript/dac/` (DAC ~300 MB). Đặt `HF_HOME`
nếu ổ home nhỏ.

### Bước 1. Xem trước dữ liệu qua bộ lọc

`finetune/prepare_vi_from_s7.py` đọc `s7_loudnorm/manifest.jsonl`, lọc theo tier A
của pipeline, chuẩn hóa số/ngày/viết tắt tiếng Việt (ZONOS2 không có NeMo cho
tiếng Việt), chia train/val trong từng speaker.

```bash
uv run python finetune/prepare_vi_from_s7.py \
    --s7-dir /path/audio-pipeline/work_XXX/s7_loudnorm \
    --out-dir data/zonos2_vi --dry-run
```

Đọc `accepted`, `hours_total`, `utts_per_speaker` và `rejected`. Mục tiêu tối thiểu
cho một giọng: 30 phút; tốt: 1–2 giờ. Nới ngưỡng khi thiếu, ví dụ
`--max-cer 0.05 --min-snr-db 15`. Giữ `--max-seconds` ≤ 20 vì `train.py` bỏ chuỗi
dài hơn 2048 frame (~23 s ở 86 frame/s, gồm cả prompt).

### Bước 2. Tạo manifest và tensor `.pt`

```bash
uv run python finetune/prepare_vi_from_s7.py \
    --s7-dir /path/audio-pipeline/work_XXX/s7_loudnorm \
    --out-dir data/zonos2_vi \
    --run-preprocess --model-path Zyphra/ZONOS2 --device cuda
```

Kết quả trong `data/zonos2_vi/`: `train.jsonl`, `val.jsonl`, `summary.json`, và hai
thư mục `train/`, `val/` chứa `.pt` (DAC codes `(T, 9)`, speaker embedding 2048-d
tính từ chính câu đó, text đã chuẩn hóa). Chạy lại sẽ dùng lại `.pt` đã có.
Nếu đã có sẵn JSONL dạng `{"audio","text","language"}`, dùng thẳng
`finetune/preprocess.py --manifest ... --no-normalize`.

**Từ dataset s8 thay vì s7.** Nếu chỉ có thư mục đóng gói `s8_package/` (đã gán tier,
audio nằm trong `dataset/wav/`), dùng `finetune/prepare_vi_from_s8.py`; nó chọn theo
tier thay vì từng ngưỡng, còn lại giống hệt script s7 (cùng cờ `--run-preprocess`,
`--max-seconds`, `--min-utts-per-speaker`):

```bash
uv run python finetune/prepare_vi_from_s8.py \
    --s8-dir /path/audio-pipeline/work_XXX/s8_package \
    --out-dir data/zonos2_vi --tiers A --dry-run          # hoặc --tiers A,B khi thiếu dữ liệu

uv run python finetune/prepare_vi_from_s8.py --s8-dir ... --out-dir data/zonos2_vi \
    --tiers A --run-preprocess --model-path Zyphra/ZONOS2 --device cuda
```

`--use-pipeline-split` giữ cách chia theo speaker của s8 (val/test gộp vào val);
`--from-manifest` đọc `manifest.jsonl` để lấy cả tier chưa được xuất, cần còn audio s7.

### Bước 3. Fine-tune LoRA

```bash
uv run python finetune/train.py \
    --model-path Zyphra/ZONOS2 \
    --data data/zonos2_vi/train \
    --output-dir runs/vi_podcast \
    --batch-size 2 --grad-accum 8 --epochs 3 \
    --grad-checkpoint --save-every 500
```

- Với 1 giờ dữ liệu, batch hiệu dụng 16, mỗi epoch khoảng 250 bước; 3 epoch là
  điểm bắt đầu hợp lý. Thiếu VRAM: `--batch-size 1 --grad-accum 16`.
- `--clean-background` chỉ khi audio tham chiếu sạch; mặc định là noisy, khớp
  server. Cờ này, `--expressive` và `--no-quality` phải dùng giống nhau lúc train
  và lúc gọi suy luận.
- Dữ liệu ít mà giọng chưa giống: thêm `--lora-targets wq wkv wo w_in w_out speaker_projection`
  hoặc tăng `--lora-r 32`. Full fine-tune (`--full --lr 1e-5`) chỉ khi có nhiều
  giờ dữ liệu và GPU đủ lớn, vì toàn bộ expert MoE được nạp dense.

Checkpoint merged nằm ở `runs/vi_podcast/` (`model.pth` + `params.json`), các bản
định kỳ ở `runs/vi_podcast/stepN/`, adapter riêng ở `lora_adapter.pt`.

### Bước 4. Nghe thử và so với model gốc

```bash
# model gốc
uv run python scripts/generate_vi.py --speaker-wav ref.wav \
    --text "Hôm nay là 24/9/2026, nhiệt độ 31,5 độ." --out out/base.wav
# model fine-tune
uv run python scripts/generate_vi.py --model-path runs/vi_podcast --speaker-wav ref.wav \
    --text "Hôm nay là 24/9/2026, nhiệt độ 31,5 độ." --out out/ft.wav
```

`ref.wav` là một clip 5–15 s của đúng giọng đã train. Lấy vài câu trong
`data/zonos2_vi/val.jsonl` (model chưa thấy) để so phát âm và độ giống; câu có số,
tên riêng và từ tiếng Anh xen kẽ là chỗ hay lộ lỗi nhất. Đánh giá nhanh bằng
Whisper large-v3 hoặc PhoWhisper trên audio sinh ra để lấy WER, và một model
speaker verification để lấy cosine similarity với `ref.wav`.

### Bước 5. Phục vụ

```bash
uv run python -m zonos2 --model-path runs/vi_podcast --tts-default-voices-dir ./default_voices/
```

Với tiếng Việt qua API, gửi `text_normalization: false` và tự chuẩn hóa text trước,
vì server sẽ từ chối `language: "vi"`; hàm `normalize_vietnamese` trong
`scripts/generate_vi.py` dùng lại được.

### Sự cố thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `Failed to build pynini==2.1.6` | Python 3.13 không có wheel; `uv python pin 3.12` rồi `rm -rf .venv && uv sync` |
| `TorchCodec is required for load_with_torchcodec` | `uv pip install "torchcodec==0.8.*"` và cài `ffmpeg` |
| `[warn] failed on ...` khi preprocess | file audio hỏng hoặc torchcodec thiếu; đối chiếu `index.json` với số dòng JSONL |
| Nhiều câu bị bỏ vì `max-frames` | giảm `--max-seconds` ở bước 1, hoặc tăng `--max-frames` khi train nếu VRAM cho phép |
| Giọng sau fine-tune giống hệt gốc | LoRA chưa đủ: kiểm tra loss có giảm không, tăng epoch/rank, thêm `speaker_projection` vào targets |
| Giọng vỡ hoặc lặp | learning rate cao hoặc quá nhiều epoch trên ít dữ liệu; lùi về `stepN/` sớm hơn |
| Số bị đọc từng chữ số | text chưa chuẩn hóa; dùng `prepare_vi_from_s7.py` (mặc định chuẩn hóa) hoặc chuẩn hóa trước khi `preprocess.py --no-normalize` |

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
