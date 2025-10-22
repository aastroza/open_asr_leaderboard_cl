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

# Image for Transformers models (Whisper, Voxtral)
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

data_download_image = (
    modal.Image.debian_slim(python_version=_PYTHON_VERSION)
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "datasets[audio]==4.0.0",
        "torch==2.7.1",
        "soundfile==0.13.1"
    )
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
