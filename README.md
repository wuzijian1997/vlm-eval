# vlm-eval

Evaluation pipeline for open-source video VLMs on **SurgCoT** (surgical video chain-of-thought benchmark).

- Paper: [SurgCoT (arXiv 2604.20319)](https://arxiv.org/abs/2604.20319)
- Data: [huggingface.co/datasets/wanggui/SurgCoT](https://huggingface.co/datasets/wanggui/SurgCoT)
- Code from the authors: [github.com/CVI-SZU/SurgCoT](https://github.com/CVI-SZU/SurgCoT)

> Status: **design stage.** The pipeline is being built one step at a time; sections marked *TODO* are not implemented yet.

## Pipeline overview

```
 raw videos (.mp4)
      │  1. preprocess.py      1 fps uniform sampling, resize to 210x360
      ▼
 frames/ + frame index
      │  2. infer.py           one model, one GPU; prompt templates from prompts/
      ▼
 outputs/<model>/<setting>/responses.jsonl     (raw responses + token counts, nothing scored)
      │  3. compute_metrics.py  per-class Acc, mIoU (temporal grounding), token cost
      ▼
 outputs/<model>/<setting>/metrics.json
```

Inference and scoring are deliberately separate: every model response is saved first, so metrics can be recomputed (or changed) without re-running any model.

## Data

Everything large lives on `/data` (the home directory has a 200G quota).

| Path | Content |
|---|---|
| `/data/zijianwu/SurgCoT/Q/{train,test,data}.json` | Direct QA: one question + reference answer per item (test: 4,825) |
| `/data/zijianwu/SurgCoT/Q_dq/…` | Decomposed CoT: main question guided by sub-questions (test: 4,811) |
| `/data/zijianwu/SurgCoT/Q_dq_clue/…` | Decomposed + clue (test split is **empty** in the HF release) |
| `/data/zijianwu/SurgCoT/GeneralSurg/` | Extracted `GeneralSurg.tar.gz`: 5,931 videos + speech transcripts, 327 GiB (archive deleted) |
| `/data/zijianwu/SurgCoT/frames/` | Preprocessed frames (*TODO*) |

Each annotation item has this shape:

```json
{"messages": [{"role": "user", "content": "<question>"},
              {"role": "assistant", "content": "<reference answer>"}],
 "videos": ["/data/wanggui/GeneralSurg/dataset/YouTubeData/<procedure>/<title>.mp4"]}
```

Video paths are the authors' absolute paths. They are remapped by replacing the prefix
`/data/wanggui/GeneralSurg/` with `/data/zijianwu/SurgCoT/GeneralSurg/`.

### ⚠ Missing annotations (blocker for step 3)

The paper defines each item as **Question / Option / Knowledge / Clue / Answer**, with multiple-choice
options, one of **5 reasoning classes**, and spatiotemporal clues (e.g. `At 647.0s`). The HF JSONs only
contain free-form question/answer pairs, **without options, class labels or timestamps**. Before metrics
can be computed we need one of:

1. the full annotation files (check inside `GeneralSurg.tar.gz` once downloaded, or ask the authors), or
2. a fallback: classify questions into the 5 classes and score free-form answers with an LLM judge
   (no mIoU possible without ground-truth time spans).

## Step 1 — Preprocessing (`preprocess.py`)

```bash
python preprocess.py                         # all videos in Q/test.json + Q_dq/test.json
python preprocess.py --limit 5 --out /tmp/f  # quick check on 5 videos
```

- Collects every video referenced by `--annotations` (default `Q/test.json Q_dq/test.json`: 1,992 videos, 389 h)
  and **uniformly samples 1 frame per second** with ffmpeg's `fps` filter.
  Frame `k` is the frame nearest to timestamp `k` seconds, so frame indices double as seconds for temporal grounding.
- Resizes every frame to **210×360 (H×W)** (bicubic). No cropping; most source videos are 640×360 (16:9), so distortion is small.
- Saves JPEGs as `frames/<path under GeneralSurg/ without .mp4>/%06d.jpg`, starting at `000000.jpg`
  (about 5–15 KB per frame, roughly 20 GB for the test videos).
- Writes `meta.json` per video (duration, original fps and size, number of frames) and merges them into `frames/index.json`.
- Idempotent: each video is written to a `.tmp` folder and renamed when complete; videos with a `meta.json` are skipped.
  Failures are listed in `frames/failures.json`.
- CPU only (`--workers`, default 32 ffmpeg processes); no GPU needed.

Open question: long videos can produce thousands of frames at 1 fps, which may exceed a model's context.
Inference may need a cap (`--max-frames`, uniformly subsampled), with the frame timestamps passed to the model.

## Step 2 — Inference (`infer.py`, *TODO*)

Candidate models (development runs):

| Model | Notes |
|---|---|
| Qwen3.5-9B | |
| InternVL3.5-8B | |

- **One GPU only** during development (`CUDA_VISIBLE_DEVICES=<id>`), e.g. GPU 0 or 5, which are usually free.
- Backend: vLLM; environment `/data/conda_envs/msswift` (torch 2.11, transformers 5.12, vLLM 0.24).
- Greedy decoding, matching the paper: `temperature=0.0, top_p=1.0, max_new_tokens=4096, repetition_penalty=1.0`.
- `--limit N` flag for quick debugging on a small subset.
- Resumable: items already in `responses.jsonl` are skipped.

Each line of `outputs/<model>/<setting>/responses.jsonl`:

```json
{"id": "...", "video": "...", "class": "...", "question": "...", "reference": "...",
 "prompt": "<fully rendered prompt>", "response": "<raw model output>",
 "num_frames": 312, "input_tokens": 51234, "output_tokens": 187, "latency_s": 4.2}
```

The fully rendered prompt is stored with every response, so it is always clear exactly what the model saw.

## Prompts (`prompts/`, *TODO*: to be reviewed in detail)

Prompts are **plain text template files**, kept out of the Python code so they can be read and edited directly:

```
prompts/
  system.txt          # shared system prompt (role, rules about output format)
  q_direct.txt        # setting Q: video + question
  q_dq.txt            # setting Q_dq: video + main question + sub-questions (multi-turn)
```

Templates use `{placeholders}` filled in by `infer.py`, for example:

```
The video is sampled at 1 frame per second; there are {num_frames} frames covering {duration} seconds.
Frame k shows the moment at k seconds.

Question: {question}
{options}

Respond in JSON: {"answer": "<option letter>", "time_span": [<start_s>, <end_s>]}
```

Design points to decide together:

- **Timestamps**: the model must be told how frames map to seconds, otherwise its predicted time spans are meaningless.
- **Output format**: a fixed, machine-parsable format (JSON, or `Answer: X` / `Time: [s, e]` lines) so step 3 can parse it reliably. Parse failures are counted, not silently dropped.
- **Chain-of-thought**: whether the model may reason before answering (more output tokens, possibly higher accuracy).
- **Per-class prompts**: whether temporal-grounding classes (e.g. micro-transition localization, anomaly onset) ask for a time span and the others do not.

## Step 3 — Metrics (`compute_metrics.py`, *TODO*)

Reads `responses.jsonl` only; never calls a model (except the optional LLM judge fallback).

Reported **per class** for the 5 SurgCoT reasoning classes, plus overall:

| Class | Abbrev. |
|---|---|
| Causal Action Ordering | CAO |
| Cue-Action Alignment | CAA |
| Affordance Mapping | AM |
| Micro-Transition Localization | MTL |
| Anomaly Onset Tracking | AOT |

| Metric | Definition |
|---|---|
| **Acc** | Fraction of items whose parsed answer matches the ground truth option. Unparsable responses count as wrong; their rate is reported separately. |
| **mIoU** | Mean temporal IoU between the predicted span `[s, e]` and the ground-truth span, `|pred ∩ gt| / |pred ∪ gt|`; a missing or invalid span gives IoU 0. Only for items that have a ground-truth span. |
| **Token cost** | Mean input tokens (visual + text), mean output tokens, and mean total tokens per item. |

Output: `outputs/<model>/<setting>/metrics.json`, plus a combined table across models.

## Layout (planned)

```
vlm-eval/
  README.md
  preprocess.py
  infer.py
  compute_metrics.py
  prompts/
  outputs/ -> /data/zijianwu/vlm-eval-outputs   (symlink, keeps large files off /home)
```
