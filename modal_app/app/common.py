import modal


_PYTHON_VERSION = "3.12"

app = modal.App(name="chilean-asr-evaluation")

dataset_volume = modal.Volume.from_name("chilean-asr-datasets", create_if_missing=True)
DATASETS_VOLPATH = "/datasets"

model_volume = modal.Volume.from_name("chilean-asr-models", create_if_missing=True)
MODELS_VOLPATH = "/models"

results_volume = modal.Volume.from_name("chilean-asr-results", create_if_missing=True)
RESULTS_VOLPATH = "/results"

# Image for NeMo ASR models (Canary, Parakeet)
nemo_transcription_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python=_PYTHON_VERSION
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": MODELS_VOLPATH,
            "CXX": "g++",
            "CC": "g++",
        }
    )
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
            "torch==2.7.1",
            "evaluate==0.4.3",
            "librosa==0.11.0",
            "soundfile==0.13.1",
            "hf_transfer==0.1.9",
            "huggingface_hub[hf-xet]==0.32.4",
            "cuda-python==12.8.0",
            "nemo_toolkit[asr]==2.3.1",

        )
    .entrypoint([])
    .add_local_dir("utils", remote_path="/root/utils")
)

# Image for Transformers models (Whisper)
transformers_transcription_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python=_PYTHON_VERSION
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": MODELS_VOLPATH,
        }
    )
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
            "torch==2.7.1",
            "transformers==4.48.1",
            "accelerate==1.3.0",
            "evaluate==0.4.3",
            "librosa==0.11.0",
            "hf_transfer==0.1.9",
            "huggingface_hub[hf-xet]==0.32.4",
            "datasets[audio]==4.0.0",
            "soundfile==0.13.1",
            "jiwer==4.0.0",
        )
    .entrypoint([])
    .add_local_dir("utils", remote_path="/root/utils")
)

# Image for Voxtral models (requires newer transformers and mistral-common)
voxtral_transcription_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python=_PYTHON_VERSION
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": MODELS_VOLPATH,
        }
    )
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
            "torch==2.7.1",
            "transformers==4.54.0",  # Voxtral requires >= 4.54.0
            "mistral-common[audio]>=1.8.1",  # Required for Voxtral audio processing
            "huggingface_hub[hf-xet]>=0.34.0",  # Required by transformers 4.54.0
            "accelerate==1.3.0",
            "evaluate==0.4.3",
            "librosa==0.11.0",
            "hf_transfer==0.1.9",
            "datasets[audio]==4.0.0",
            "soundfile==0.13.1",
            "jiwer==4.0.0",
        )
    .entrypoint([])
    .add_local_dir("utils", remote_path="/root/utils")
)

# Image for Phi-4 multimodal models
phi4_multimodal_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python=_PYTHON_VERSION
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": MODELS_VOLPATH,
        }
    )
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
            "wheel",
            "packaging",  # Required by flash-attn setup.py
            "torch==2.6.0",
            "torchvision==0.21.0",
            "torchaudio",
        )
    .pip_install(
            "flash-attn==2.7.4.post1",  # Install flash-attn after torch and packaging
        )
    .pip_install(
            "transformers==4.48.2",
            "accelerate==1.3.0",
            "evaluate",
            "datasets",
            "librosa",
            "jiwer",
            "soundfile",
            "pillow",
            "scipy",
            "peft==0.13.2",
            "backoff==2.2.1",
            "hf_transfer==0.1.9",
            "huggingface_hub[hf-xet]>=0.34.0",
        )
    .entrypoint([])
    .add_local_dir("utils", remote_path="/root/utils")
)

data_download_image = (
    modal.Image.debian_slim(python_version=_PYTHON_VERSION)
    .apt_install("ffmpeg", "libsndfile1", "libavcodec-dev", "libavformat-dev", "libavutil-dev")
    .pip_install(
        "datasets==3.1.0",  # Use older version to avoid torchcodec dependency
        "torch==2.4.0",     # Use compatible PyTorch version
        "soundfile==0.13.1",
        "librosa==0.10.2"   # Add librosa for audio processing
    )
    .env({"DATASET_AUDIO_BACKEND": "soundfile"})  # Force soundfile backend
    .add_local_dir("utils", remote_path="/root/utils")
)

runner_image = (
    modal.Image.debian_slim(python_version=_PYTHON_VERSION)
    .apt_install("libsndfile1")
    .pip_install(
            "pandas==2.3.1",
            "numpy==2.2.6",
            "evaluate==0.4.3",
            "librosa==0.11.0",
            "jiwer==4.0.0",
            "regex==2024.11.6",

        )
    .add_local_dir("utils", remote_path="/root/utils")
)

# Image for ElevenLabs API (no GPU needed)
elevenlabs_transcription_image = (
    modal.Image.debian_slim(python_version=_PYTHON_VERSION)
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
            "elevenlabs==2.20.1",
            "requests==2.32.3",
            "soundfile==0.13.1",
            "evaluate==0.4.3",
            "jiwer==4.0.0",
        )
    .add_local_dir("utils", remote_path="/root/utils")
)

# Image for OmniLingual ASR (fairseq2-based)
omnilingual_transcription_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python=_PYTHON_VERSION
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": MODELS_VOLPATH,
        }
    )
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
            "torch==2.7.1",
            "omnilingual-asr",
            "evaluate==0.4.3",
            "librosa==0.11.0",
            "soundfile==0.13.1",
            "hf_transfer==0.1.9",
            "huggingface_hub[hf-xet]==0.32.4",
            "jiwer==4.0.0",
        )
    .entrypoint([])
    .add_local_dir("utils", remote_path="/root/utils")
)
