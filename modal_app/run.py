import tempfile
import argparse
from datetime import datetime
import datasets

from app.common import (
    app,
    dataset_volume,
)
from app.stage_data import download_hf_dataset
from app.transcription import (
    NeMoAsrBatchTranscription,
    TransformersAsrBatchTranscription,
    VoxtralAsrBatchTranscription,
    Phi4MultimodalAsrBatchTranscription,
    ElevenLabsAsrBatchTranscription,
    TranscriptionRunner,
)
from utils.data import DATASET_CONFIG, DATASET_STORAGE_NAME


@app.local_entrypoint()
def stage_data():
    """
    Stage the Chilean Spanish dataset to Modal Volume.

    Usage:
        modal run run.py::stage_data
    """

    print("Staging Chilean Spanish ASR dataset...")

    prepped_datasets = []
    for data_dict in download_hf_dataset.starmap(DATASET_CONFIG):
        if data_dict is not None:
            prepped_datasets.append(datasets.Dataset.from_dict(data_dict))
        else:
            print("Warning: One of the dataset processing jobs returned None")

    if not prepped_datasets:
        print("Error: No datasets were successfully processed. Check the logs above for errors.")
        return

    print(f"Successfully processed {len(prepped_datasets)} dataset(s)")
    full_ds = datasets.concatenate_datasets(prepped_datasets).sort("audio_length_s", reverse=True)

    with tempfile.TemporaryFile() as temp_file:
        full_ds.to_csv(temp_file)
        temp_file.seek(0)
        with dataset_volume.batch_upload(force=True) as batch:
            batch.put_file(temp_file, f"/{DATASET_STORAGE_NAME}/full_features.csv")

    print("Dataset staging complete!")


@app.local_entrypoint()
def batch_transcription_nemo(*args):
    """
    Run batch transcription using NeMo ASR models (Canary, Parakeet).

    Usage:
        modal run run.py::batch_transcription_nemo --model_id nvidia/parakeet-tdt-0.6b-v3
        modal run run.py::batch_transcription_nemo --model_id nvidia/canary-1b-v2 --gpu-type A100 --gpu-batch-size 64
    """

    print("Running NeMo ASR batch transcription")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default=NeMoAsrBatchTranscription.DEFAULT_MODEL_ID,
        help="Model identifier. Should be loadable with NVIDIA NeMo.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="es-cl-asr-test-only-full",
        help="Dataset name (e.g., 'es-cl-asr-test-only-full')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (e.g., 'test')",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default=NeMoAsrBatchTranscription.DEFAULT_GPU_TYPE,
        help="The GPU type to run the pipeline on.",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=NeMoAsrBatchTranscription.DEFAULT_BATCH_SIZE,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=NeMoAsrBatchTranscription.DEFAULT_NUM_REQUESTS,
        help="Number of calls to make to the run_inference method.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results",
        help="Path to save the combined CSV file",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=f"NeMo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Job ID.",
    )
    cfg = parser.parse_args(args=args)

    print("Job Config:")
    print(cfg)

    runner = TranscriptionRunner(num_requests=cfg.num_requests, model_type="nemo")
    runner.run_transcription.remote(cfg)


@app.local_entrypoint()
def batch_transcription_transformers(*args):
    """
    Run batch transcription using Transformers models (Whisper).

    Usage:
        modal run run.py::batch_transcription_transformers --model_id openai/whisper-large-v3
        modal run run.py::batch_transcription_transformers --model_id openai/whisper-large-v3-turbo --gpu-batch-size 24
    """

    print("Running Transformers (Whisper) ASR batch transcription")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default=TransformersAsrBatchTranscription.DEFAULT_MODEL_ID,
        help="Model identifier. Should be loadable with Transformers.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="es-cl-asr-test-only-full",
        help="Dataset name (e.g., 'es-cl-asr-test-only-full')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (e.g., 'test')",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default=TransformersAsrBatchTranscription.DEFAULT_GPU_TYPE,
        help="The GPU type to run the pipeline on.",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=TransformersAsrBatchTranscription.DEFAULT_BATCH_SIZE,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=TransformersAsrBatchTranscription.DEFAULT_NUM_REQUESTS,
        help="Number of calls to make to the run_inference method.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results",
        help="Path to save the combined CSV file",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=f"Transformers_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Job ID.",
    )
    cfg = parser.parse_args(args=args)

    print("Job Config:")
    print(cfg)

    runner = TranscriptionRunner(num_requests=cfg.num_requests, model_type="transformers")
    runner.run_transcription.remote(cfg)


@app.local_entrypoint()
def batch_transcription_voxtral(*args):
    """
    Run batch transcription using Voxtral models.

    Usage:
        modal run run.py::batch_transcription_voxtral --model_id mistralai/Voxtral-Mini-3B-2507
        modal run run.py::batch_transcription_voxtral --model_id mistralai/Voxtral-Mini-3B-2507 --gpu-batch-size 8
    """

    print("Running Voxtral ASR batch transcription")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default=VoxtralAsrBatchTranscription.DEFAULT_MODEL_ID,
        help="Model identifier. Should be a Voxtral model.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="es-cl-asr-test-only-full",
        help="Dataset name (e.g., 'es-cl-asr-test-only-full')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (e.g., 'test')",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default=VoxtralAsrBatchTranscription.DEFAULT_GPU_TYPE,
        help="The GPU type to run the pipeline on.",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=VoxtralAsrBatchTranscription.DEFAULT_BATCH_SIZE,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=VoxtralAsrBatchTranscription.DEFAULT_NUM_REQUESTS,
        help="Number of calls to make to the run_inference method.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results",
        help="Path to save the combined CSV file",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=f"Voxtral_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Job ID.",
    )
    cfg = parser.parse_args(args=args)

    print("Job Config:")
    print(cfg)

    runner = TranscriptionRunner(num_requests=cfg.num_requests, model_type="voxtral")
    runner.run_transcription.remote(cfg)


@app.local_entrypoint()
def batch_transcription_phi4_multimodal(*args):
    """
    Run batch transcription using Phi-4 multimodal models.

    Usage:
        modal run run.py::batch_transcription_phi4_multimodal --model_id microsoft/Phi-4-multimodal-instruct
        modal run run.py::batch_transcription_phi4_multimodal --model_id microsoft/Phi-4-multimodal-instruct --gpu-batch-size 4
    """

    print("Running Phi-4 Multimodal ASR batch transcription")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default=Phi4MultimodalAsrBatchTranscription.DEFAULT_MODEL_ID,
        help="Model identifier. Should be a Phi-4 multimodal model.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="es-cl-asr-test-only-full",
        help="Dataset name (e.g., 'es-cl-asr-test-only-full')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (e.g., 'test')",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default=Phi4MultimodalAsrBatchTranscription.DEFAULT_GPU_TYPE,
        help="The GPU type to run the pipeline on.",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=Phi4MultimodalAsrBatchTranscription.DEFAULT_BATCH_SIZE,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=Phi4MultimodalAsrBatchTranscription.DEFAULT_NUM_REQUESTS,
        help="Number of calls to make to the run_inference method.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results",
        help="Path to save the combined CSV file",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=f"Phi4Multimodal_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Job ID.",
    )
    cfg = parser.parse_args(args=args)

    print("Job Config:")
    print(cfg)

    runner = TranscriptionRunner(num_requests=cfg.num_requests, model_type="phi4_multimodal")
    runner.run_transcription.remote(cfg)


@app.local_entrypoint()
def batch_transcription_elevenlabs(*args):
    """
    Run batch transcription using ElevenLabs API.

    Usage:
        modal run run.py::batch_transcription_elevenlabs --model_id scribe_v1
        modal run run.py::batch_transcription_elevenlabs --model_id scribe_v1 --batch-size 5

    Note: Requires ELEVENLABS_API_KEY to be set in Modal secrets.
    Create the secret with: modal secret create elevenlabs-api-key ELEVENLABS_API_KEY=your_key_here
    """

    print("Running ElevenLabs API batch transcription")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default=ElevenLabsAsrBatchTranscription.DEFAULT_MODEL_ID,
        help="ElevenLabs model ID (e.g., 'scribe_v1')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="es-cl-asr-test-only-full",
        help="Dataset name (e.g., 'es-cl-asr-test-only-full')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (e.g., 'test')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=ElevenLabsAsrBatchTranscription.DEFAULT_BATCH_SIZE,
        help="Number of concurrent API requests per batch.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=ElevenLabsAsrBatchTranscription.DEFAULT_NUM_REQUESTS,
        help="Number of calls to make to the run_inference method.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results",
        help="Path to save the combined CSV file",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=f"ElevenLabs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Job ID.",
    )
    cfg = parser.parse_args(args=args)

    # ElevenLabs doesn't use GPU, so set a dummy value
    cfg.gpu_type = "none"
    cfg.gpu_batch_size = cfg.batch_size

    print("Job Config:")
    print(cfg)

    runner = TranscriptionRunner(num_requests=cfg.num_requests, model_type="elevenlabs")
    runner.run_transcription.remote(cfg)


@app.local_entrypoint()
def batch_transcription(*args):
    """
    Run batch transcription with automatic model type detection.

    Usage:
        modal run run.py::batch_transcription --model_id nvidia/parakeet-tdt-0.6b-v3
        modal run run.py::batch_transcription --model_id openai/whisper-large-v3
    """

    print("Running batch transcription with auto-detection")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier (e.g., 'nvidia/canary-1b-v2' or 'openai/whisper-large-v3').",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="es-cl-asr-test-only",
        help="Dataset name (e.g., 'es-cl-asr-test-only')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (e.g., 'test')",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default="L40S",
        help="The GPU type to run the pipeline on.",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=None,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of calls to make to the run_inference method.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results",
        help="Path to save the combined CSV file",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Job ID.",
    )
    cfg = parser.parse_args(args=args)

    # Auto-detect model type
    if "nvidia" in cfg.model_id.lower() or "nemo" in cfg.model_id.lower():
        model_type = "nemo"
        default_batch_size = 32
    elif "voxtral" in cfg.model_id.lower():
        model_type = "voxtral"
        default_batch_size = 8
    elif "phi-4" in cfg.model_id.lower() or "phi4" in cfg.model_id.lower():
        model_type = "phi4_multimodal"
        default_batch_size = 4
    elif "elevenlabs" in cfg.model_id.lower() or "scribe" in cfg.model_id.lower():
        model_type = "elevenlabs"
        default_batch_size = 10
    else:
        model_type = "transformers"
        default_batch_size = 16

    if cfg.gpu_batch_size is None:
        cfg.gpu_batch_size = default_batch_size

    # ElevenLabs doesn't need GPU
    if model_type == "elevenlabs":
        cfg.gpu_type = "none"

    if cfg.job_id is None:
        model_name = cfg.model_id.replace("/", "-")
        cfg.job_id = f"{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    print(f"Detected model type: {model_type}")
    print("Job Config:")
    print(cfg)

    runner = TranscriptionRunner(num_requests=cfg.num_requests, model_type=model_type)
    runner.run_transcription.remote(cfg)
