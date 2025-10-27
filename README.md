# Chilean Spanish ASR Evaluation

> **Cloud-based evaluation framework for Automatic Speech Recognition (ASR) models on Chilean Spanish, powered by [Modal](https://modal.com).**

## About This Repository

This repository is a **streamlined, task-specific version** of the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) evaluation framework, specifically adapted for benchmarking ASR models on the **Chilean Spanish dialect**.

### What is the Open ASR Leaderboard?

The [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) is a comprehensive benchmarking framework developed by Hugging Face, NVIDIA NeMo, and the community to evaluate ASR models across multiple English datasets (LibriSpeech, AMI, VoxPopuli, Earnings-22, GigaSpeech, SPGISpeech, TED-LIUM). It supports various ASR frameworks including Transformers, NeMo, SpeechBrain, and more, providing standardized WER and RTFx metrics.

### How This Repository Differs

This Chilean Spanish adaptation makes the following key modifications to focus exclusively on Chilean Spanish ASR evaluation:

| Aspect | Original Open ASR Leaderboard | This Repository |
|--------|-------------------------------|-----------------|
| **Target Language** | English (primarily) | Chilean Spanish |
| **Dataset** | 7 English datasets (LibriSpeech, AMI, etc.) | Single dataset: [`astroza/es-cl-asr-test-only`](https://huggingface.co/datasets/astroza/es-cl-asr-test-only) |
| **Text Normalization** | English text normalizer | **Multilingual normalizer** preserving Spanish accents (á, é, í, ó, ú, ñ) |
| **Model Focus** | Broad coverage (~50+ models) | **7 selected models** optimized for multilingual/Spanish ASR |
| **Frameworks** | 10+ frameworks (Transformers, NeMo, SpeechBrain, CTranslate2, etc.) | **4 frameworks**: Transformers, NeMo, Phi, API |
| **Execution** | Local GPU execution | **Cloud-based** parallel execution via Modal |
| **Speed** | Single GPU, sequential | **10-50x faster** with parallel GPUs |

---

## Models Evaluated

This repository evaluates **7 state-of-the-art ASR models** selected for their multilingual or Spanish language support:

| Model | Type | Framework | Parameters | Notes |
|-------|------|-----------|------------|-------|
| **openai/whisper-large-v3** | Multilingual | Transformers | 1.5B | OpenAI's flagship ASR model |
| **openai/whisper-large-v3-turbo** | Multilingual | Transformers | 809M | Faster Whisper variant |
| **nvidia/canary-1b-v2** | Multilingual | NeMo | 1B | NVIDIA's multilingual ASR |
| **nvidia/parakeet-tdt-0.6b-v3** | Multilingual | NeMo | 0.6B | Lightweight, fast inference |
| **microsoft/Phi-4-multimodal-instruct** | Multimodal | Phi | 14B | Microsoft's multimodal LLM with audio |
| **mistralai/Voxtral-Mini-3B-2507** | Speech-to-text | Transformers | 3B | Mistral's ASR model |
| **elevenlabs/scribe_v1** | API-based | API | N/A | ElevenLabs' commercial ASR API |

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

## Why Modal?

Running ASR evaluations on Modal provides significant advantages:

- ✅ **10-50x faster**: Parallel processing across multiple GPUs
- ✅ **No local GPU required**: Run from any machine with internet
- ✅ **Pay per second**: Only pay for compute used (~$0.03-0.10 per 1000 samples)
- ✅ **Easy setup**: No CUDA installation or driver management
- ✅ **Scalable**: Automatically scales to dataset size

---

## Repository Structure

```
open_asr_leaderboard_cl/
├── README.md                      # This file
├── modal_app/                     # Modal-based evaluation (all code here)
│   ├── README.md                  # Detailed documentation
│   ├── pyproject.toml             # Dependencies
│   ├── run.py                     # Main entrypoints
│   ├── app/                       # Modal functions
│   │   ├── common.py              # Volumes, images, app config
│   │   ├── stage_data.py          # Dataset staging
│   │   └── transcription.py       # Transcription workers
│   └── utils/                     # Utilities
│       ├── data.py                # Data loading
│       └── normalizer/            # Text normalization
└── LICENSE
```

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

## Contributing

This repository is a specialized fork focused on Chilean Spanish ASR evaluation. For contributions:

1. **Bug fixes & improvements**: Open an issue or pull request
2. **Adding new models**: Ensure they support Spanish and follow the existing evaluation structure
3. **General ASR leaderboard features**: Contribute to the upstream [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard)

---

## License

This repository maintains the same license as the original [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard). See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Hugging Face, NVIDIA NeMo, and the Open ASR Leaderboard contributors** for the original evaluation framework
- All model developers (OpenAI, NVIDIA, Microsoft, Mistral AI, ElevenLabs) for their ASR models
- **[Modal](https://modal.com/)** for providing the cloud compute platform

---

## Support

For issues specific to:
- **This Chilean Spanish adaptation**: Open an issue in this repository
- **Modal setup or execution**: See [modal_app/README.md](modal_app/README.md) or visit [Modal Docs](https://modal.com/docs)
- **Original Open ASR Leaderboard framework**: Visit [huggingface/open_asr_leaderboard](https://github.com/huggingface/open_asr_leaderboard)
- **Dataset issues**: Contact the [dataset author](https://huggingface.co/datasets/astroza/es-cl-asr-test-only)
- **Model-specific problems**: Check the respective model cards on Hugging Face
