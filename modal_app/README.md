# Chilean Spanish ASR Evaluation on Modal

> **Run Chilean Spanish ASR model evaluations 100x faster and cheaper with Modal**

This directory contains a Modal-based implementation for evaluating Chilean Spanish ASR models at scale. It distributes the evaluation workload across multiple GPUs in the cloud, enabling fast parallel processing.

## Overview

This implementation:
- **Stages datasets** to Modal Volumes for fast access
- **Distributes inference** across multiple GPU containers in parallel
- **Supports multiple frameworks**: NeMo (Canary, Parakeet) and Transformers (Whisper, Voxtral)
- **Caches models** in Modal Volumes for fast subsequent runs
- **Saves results** to Modal Volumes with WER and RTFx metrics

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Modal Volumes (Persistent Storage)                         │
├─────────────────────────────────────────────────────────────┤
│ • chilean-asr-datasets: Audio files + features             │
│ • chilean-asr-models: Cached model weights                 │
│ • chilean-asr-results: Evaluation results (CSV)            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Modal Functions (Serverless Compute)                       │
├─────────────────────────────────────────────────────────────┤
│ 1. stage_data: Download and prepare Chilean dataset        │
│ 2. run_inference: Parallel GPU transcription (10+ workers) │
│ 3. score_call: Calculate WER and RTFx metrics              │
│ 4. save_results: Write results to CSV                      │
└─────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Install uv (package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

### 2. Install dependencies

```bash
cd modal_app
uv sync
```

### 3. Setup Modal

```bash
# Create account at modal.com
modal setup

# Add API token (if needed)
modal token new
```

### 4. Authenticate with Hugging Face

For accessing gated models and the Chilean Spanish dataset:

```bash
# Set your HuggingFace token
export HF_TOKEN=hf_your_token_here
```

Or add to `.env` file in the `modal_app` directory:
```
HF_TOKEN=hf_your_token_here
```

## Usage

### Step 1: Stage the Chilean Spanish Dataset (One-time setup)

Download and prepare the dataset to Modal Volume:

```bash
modal run run.py::stage_data
```

This will:
- Download `astroza/es-cl-asr-test-only` from HuggingFace
- Convert audio to 16kHz WAV files
- Save to Modal Volume `chilean-asr-datasets`
- Create feature CSV for fast loading

**Note:** This only needs to be run once. Subsequent evaluations will use the cached dataset.

### Step 2: Run Batch Transcription

#### Option A: Auto-detect Model Type

```bash
# NeMo models (auto-detected)
modal run run.py::batch_transcription --model_id nvidia/parakeet-tdt-0.6b-v3

# Transformers models (auto-detected)
modal run run.py::batch_transcription --model_id openai/whisper-large-v3
```

#### Option B: Use Specific Entrypoint

**For NeMo models (Canary, Parakeet):**

```bash
modal run run.py::batch_transcription_nemo \
  --model_id nvidia/canary-1b-v2 \
  --gpu-type L40S \
  --gpu-batch-size 32 \
  --num-requests 10
```

**For Transformers models (Whisper, Voxtral):**

```bash
modal run run.py::batch_transcription_transformers \
  --model_id openai/whisper-large-v3 \
  --gpu-type L40S \
  --gpu-batch-size 16 \
  --num-requests 10
```

## Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | Required | Model identifier (e.g., `nvidia/canary-1b-v2` or `openai/whisper-large-v3`) |
| `--dataset` | `es-cl-asr-test-only` | Dataset name |
| `--split` | `test` | Dataset split |
| `--gpu-type` | `L40S` | GPU type (`L4`, `L40S`, `A100`, `H100`) |
| `--gpu-batch-size` | 32 (NeMo) / 16 (Transformers) | Samples per GPU batch |
| `--num-requests` | 10 | Number of parallel GPU containers |
| `--job-id` | Auto-generated | Job identifier for results |

### GPU Types and Pricing

| GPU Type | VRAM | Cost/hour | Best For |
|----------|------|-----------|----------|
| **L4** | 24GB | ~$0.50 | Small models, testing |
| **L40S** | 48GB | ~$1.20 | **Recommended** - Best value |
| **A100** | 40GB/80GB | ~$3.50 | Large models, max speed |
| **H100** | 80GB | ~$5.00 | Extreme performance |

## Supported Models

### NeMo Models

```bash
# Parakeet TDT 0.6B (Lightweight, fast)
modal run run.py::batch_transcription_nemo --model_id nvidia/parakeet-tdt-0.6b-v3

# Canary 1B v2 (Multilingual, high accuracy)
modal run run.py::batch_transcription_nemo --model_id nvidia/canary-1b-v2 --gpu-batch-size 64
```

### Transformers Models

```bash
# Whisper Large V3 (Best accuracy)
modal run run.py::batch_transcription_transformers --model_id openai/whisper-large-v3

# Whisper Large V3 Turbo (Faster inference)
modal run run.py::batch_transcription_transformers --model_id openai/whisper-large-v3-turbo

# Mistral Voxtral Mini 3B
modal run run.py::batch_transcription_voxtral --model_id mistralai/Voxtral-Mini-3B-2507
```

## Output

Results are saved to the Modal Volume `chilean-asr-results` in two formats:

### 1. Summary Results
**Path:** `/results_summaries/results_summary_{job_id}.csv`

Contains aggregated metrics per request:
- `num_samples`: Number of samples processed
- `wer`: Word Error Rate (%)
- `rtfx`: Real-time factor (speedup)
- `total_time`: Processing time (seconds)
- `total_audio_length`: Total audio duration (seconds)

### 2. Detailed Results
**Path:** `/results/results_{job_id}.csv`

Contains per-sample transcriptions and references:
- All columns from summary results
- `transcriptions`: Model predictions
- `original_text`: Ground truth
- `dataset`: Dataset name
- `split`: Dataset split

### Downloading Results

Results are automatically saved to Modal Volumes. To download them locally:

```bash
# List available results
modal volume ls chilean-asr-results results/

# Download specific result file
modal volume get chilean-asr-results results/results_NeMo_2025-10-21_12-00-00.csv results/
```

## Example Workflow

Here's a complete example evaluating 3 models:

```bash
# 1. Stage dataset (one-time)
modal run run.py::stage_data

# 2. Evaluate NeMo Parakeet
modal run run.py::batch_transcription \
  --model_id nvidia/parakeet-tdt-0.6b-v3 \
  --gpu-batch-size 64 \
  --num-requests 20

# 3. Evaluate NeMo Canary
modal run run.py::batch_transcription \
  --model_id nvidia/canary-1b-v2 \
  --gpu-batch-size 32 \
  --num-requests 15

# 4. Evaluate Whisper Large V3
modal run run.py::batch_transcription \
  --model_id openai/whisper-large-v3 \
  --gpu-batch-size 16 \
  --num-requests 10

# 5. Download all results
modal volume get chilean-asr-results results/ ./local_results/ --recursive
```

## Performance Optimization

### Tuning `--num-requests`

This controls the number of parallel GPU containers. More containers = faster completion, but costs scale linearly.

**Guidelines:**
- **Small datasets (<1000 samples):** `--num-requests 5-10`
- **Medium datasets (1000-5000 samples):** `--num-requests 10-20`
- **Large datasets (>5000 samples):** `--num-requests 20-50`

### Tuning `--gpu-batch-size`

This controls samples processed per GPU. Larger batches = better GPU utilization, but requires more VRAM.

**For NeMo models:**
- L4 (24GB): `--gpu-batch-size 16-32`
- L40S (48GB): `--gpu-batch-size 32-64`
- A100 (80GB): `--gpu-batch-size 64-128`

**For Transformers models:**
- L4 (24GB): `--gpu-batch-size 8-16`
- L40S (48GB): `--gpu-batch-size 16-32`
- A100 (80GB): `--gpu-batch-size 32-64`

### Cost Estimation

Example for 1000 audio samples (avg 10s each = 10,000s total):

**NeMo Parakeet (RTFx ~100):**
- Processing time: 10,000s / 100 = 100s
- With 10 containers: ~10s wall time
- Cost: 10 containers × 10s × $1.20/hr ÷ 3600 = **~$0.03**

**Whisper Large V3 (RTFx ~40):**
- Processing time: 10,000s / 40 = 250s
- With 10 containers: ~25s wall time
- Cost: 10 containers × 25s × $1.20/hr ÷ 3600 = **~$0.08**

## Troubleshooting

### Dataset not found

```bash
# Re-stage the dataset
modal run run.py::stage_data
```

### Model download timeout

First run downloads models to cache. Subsequent runs are faster.

```bash
# Use a timeout if models are large
modal run run.py::batch_transcription --model_id nvidia/canary-1b-v2
# This may take 5-10 minutes on first run
```

### Out of memory errors

Reduce `--gpu-batch-size`:

```bash
modal run run.py::batch_transcription \
  --model_id openai/whisper-large-v3 \
  --gpu-batch-size 8  # Reduced from default 16
```

Or upgrade GPU type:

```bash
modal run run.py::batch_transcription \
  --model_id openai/whisper-large-v3 \
  --gpu-type A100
```

## Directory Structure

```
modal_app/
├── README.md                  # This file
├── pyproject.toml             # Dependencies
├── run.py                     # Main entrypoints
├── app/
│   ├── __init__.py
│   ├── common.py              # Modal app, volumes, images
│   ├── stage_data.py          # Dataset staging
│   └── transcription.py       # NeMo and Transformers transcription
└── utils/
    ├── __init__.py
    ├── data.py                # Data utilities
    └── normalizer/            # Text normalization
        ├── __init__.py
        ├── data_utils.py
        ├── eval_utils.py
        ├── normalizer.py
        └── english_abbreviations.py
```

## Comparison: Local vs. Modal

| Aspect | Local Evaluation | Modal Evaluation |
|--------|------------------|------------------|
| **Speed** | 1x (single GPU) | 10-50x (parallel GPUs) |
| **Cost** | GPU rental fees | Pay per second used |
| **Setup** | Install CUDA, drivers, deps | `modal setup` |
| **Scalability** | Limited by local hardware | Elastic cloud GPUs |
| **Model caching** | Local disk | Modal Volumes |
| **Results** | Local files | Modal Volumes |

## Advanced Usage

### Evaluate Custom Dataset

1. Update `utils/data.py`:
```python
DATASET_CONFIG = [
    ("your-hf-username", "your-dataset-name", "test"),
]
```

2. Re-stage data:
```bash
modal run run.py::stage_data
```

3. Run evaluation:
```bash
modal run run.py::batch_transcription \
  --model_id nvidia/canary-1b-v2 \
  --dataset your-dataset-name
```

### Monitor Jobs

```bash
# View logs in real-time
modal app logs chilean-asr-evaluation

# View running containers
modal container list
```

### Clean Up Volumes

```bash
# Delete dataset (re-stage required)
modal volume delete chilean-asr-datasets

# Delete results
modal volume delete chilean-asr-results

# Delete model cache
modal volume delete chilean-asr-models
```

## Contributing

This Modal implementation is adapted from the [Modal Labs Open Batch Transcription](https://github.com/modal-labs/modal-examples) example, customized for Chilean Spanish ASR evaluation.

For issues or improvements, please open an issue in the main repository.

## License

Same as the parent repository. See [LICENSE](../LICENSE) for details.
