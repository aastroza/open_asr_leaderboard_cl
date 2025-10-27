# Chilean Spanish ASR Evaluation

> **Cloud-based evaluation framework for Automatic Speech Recognition (ASR) models on Chilean Spanish, powered by [Modal](https://modal.com).**

This repository contains the code for the Open ASR Leaderboard. The leaderboard is a Gradio Space that allows users to compare the accuracy of ASR models on a variety of datasets. The leaderboard is hosted at [idsudd/open_asr_leaderboard_cl](https://huggingface.co/spaces/idsudd/open_asr_leaderboard_cl).

## About This Repository

This repository is a **streamlined, task-specific version** of the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) evaluation framework, specifically adapted for benchmarking ASR models on the **Chilean Spanish dialect**.

### What is the Open ASR Leaderboard?

The [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) is a comprehensive benchmarking framework developed by Hugging Face, NVIDIA NeMo, and the community to evaluate ASR models across multiple English datasets (LibriSpeech, AMI, VoxPopuli, Earnings-22, GigaSpeech, SPGISpeech, TED-LIUM). It supports various ASR frameworks including Transformers, NeMo, SpeechBrain, and more, providing standardized WER and RTFx metrics.

### How This Repository Differs

This Chilean Spanish adaptation makes the following key modifications to focus exclusively on Chilean Spanish ASR evaluation:

| Aspect | Original Open ASR Leaderboard | This Repository |
|--------|-------------------------------|-----------------|
| **Target Language** | English (primarily) | Chilean Spanish |
| **Dataset** | 7 English datasets (LibriSpeech, AMI, etc.) | 3 Chilean Spanish datasets (Common Voice, Google Chilean Spanish, Datarisas) |
| **Text Normalization** | English text normalizer | **Multilingual normalizer** preserving Spanish accents (á, é, í, ó, ú, ñ) |
| **Model Focus** | Broad coverage (~50+ models) | **7 selected models** optimized for multilingual/Spanish ASR |
| **Execution** | Local GPU execution | **Cloud-based** parallel execution via Modal |

---

## Models Evaluated

This repository evaluates **9 state-of-the-art ASR models** selected for their multilingual or Spanish language support:

| Model | Type | Framework | Parameters | Notes |
|-------|------|-----------|------------|-------|
| **openai/whisper-large-v3** | Multilingual | Transformers | 1.5B | OpenAI's flagship ASR model |
| **openai/whisper-large-v3-turbo** | Multilingual | Transformers | 809M | Faster Whisper variant |
| **openai/whisper-small** | Multilingual | Transformers | 244M | Reference baseline model |
| **rcastrovexler/whisper-small-es-cl** | Chilean Spanish | Transformers | 244M | Only fine-tuned model found for Chilean Spanish |
| **nvidia/canary-1b-v2** | Multilingual | NeMo | 1B | NVIDIA's multilingual ASR |
| **nvidia/parakeet-tdt-0.6b-v3** | Multilingual | NeMo | 0.6B | Lightweight, fast inference |
| **microsoft/Phi-4-multimodal-instruct** | Multimodal | Phi | 14B | Microsoft's multimodal LLM with audio |
| **mistralai/Voxtral-Mini-3B-2507** | Speech-to-text | Transformers | 3B | Mistral's ASR model |
| **elevenlabs/scribe_v1** | API-based | API | N/A | ElevenLabs' commercial ASR API |

**Note:** `rcastrovexler/whisper-small-es-cl` is the only fine-tuned model we found specifically for Chilean Spanish transcription. It's included alongside `openai/whisper-small` as a reference baseline, since the Chilean model is a fine-tuning of the original Whisper Small.

## Dataset

- **Dataset**: [`astroza/es-cl-asr-test-only`](https://huggingface.co/datasets/astroza/es-cl-asr-test-only)
- **Language**: Spanish (Chilean variant)
- **Split**: `test`
- **Domain**: Chilean Spanish speech samples

## Metrics

Following the Open ASR Leaderboard standard, we report:

- **WER (Word Error Rate)**: ⬇️ Lower is better - Measures transcription accuracy
- **RTFx (Real-Time Factor)**: ⬆️ Higher is better - Measures inference speed (audio_duration / transcription_time)

---

## Quick Start

This project uses [Modal](https://modal.com) for cloud-based distributed GPU evaluation. All evaluation code is in the `modal_app/` directory.

### Prerequisites

1. Python 3.8+
2. Modal account (free tier available)
3. Hugging Face account

### Setup

```bash
# Clone the repository
git clone https://github.com/aastroza/open_asr_leaderboard_cl.git
cd open_asr_leaderboard_cl/modal_app

# Install dependencies
pip install uv  # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Setup Modal
modal setup

# Authenticate with Hugging Face
export HF_TOKEN=your_hf_token_here
```

### Run Evaluation

```bash
# Stage dataset (one-time setup)
modal run run.py::stage_data

# Evaluate a model
modal run run.py::batch_transcription --model_id nvidia/parakeet-tdt-0.6b-v3

# View results
modal volume ls chilean-asr-results results/
```

### 📖 Full Documentation

For detailed setup instructions, configuration options, supported models, and troubleshooting, see:

**[`modal_app/README.md`](modal_app/README.md)**

---

## Text Normalization for Spanish

This repository uses a **multilingual normalizer** configured to preserve Spanish characters:

```python
normalizer = BasicMultilingualTextNormalizer(remove_diacritics=False)
```

**What it does:**
- ✅ Preserves: `á, é, í, ó, ú, ñ, ü, ¿, ¡`
- ✅ Removes: Brackets `[...]`, parentheses `(...)`, special symbols
- ✅ Normalizes: Whitespace, capitalization (converts to lowercase)
- ❌ Does NOT remove: Accents or Spanish-specific characters

**Example:**
```python
Input:  "¿Cómo estás? [ruido] (suspiro)"
Output: "cómo estás"
```

This is critical for Spanish evaluation, as diacritics change word meaning:
- `esta` (this) vs. `está` (is)
- `si` (if) vs. `sí` (yes)
- `el` (the) vs. `él` (he)

---

## Citation

If you use this evaluation framework or results, please cite both the Chilean Spanish dataset and the original Open ASR Leaderboard:

```bibtex
@misc{astroza2024chilean,
  title={Chilean Spanish ASR Test Dataset},
  author={Alonso Astroza},
  year={2025},
  howpublished={\url{https://huggingface.co/datasets/astroza/es-cl-asr-test-only}}
}

@misc{open-asr-leaderboard,
  title={Open Automatic Speech Recognition Leaderboard},
  author={Srivastav, Vaibhav and Majumdar, Somshubra and Koluguri, Nithin and Moumen, Adel and Gandhi, Sanchit and Hugging Face Team and Nvidia NeMo Team},
  year={2023},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/spaces/hf-audio/open_asr_leaderboard}}
}
```

---

## Acknowledgments

- **Hugging Face, NVIDIA NeMo, and the Open ASR Leaderboard contributors** for the original evaluation framework
- All model developers (OpenAI, NVIDIA, Microsoft, Mistral AI, ElevenLabs) for their ASR models
- **[Modal](https://modal.com/)** for providing the cloud compute platform
