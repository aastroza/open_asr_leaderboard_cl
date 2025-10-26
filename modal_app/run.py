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
    else:
        model_type = "transformers"
        default_batch_size = 16

    if cfg.gpu_batch_size is None:
        cfg.gpu_batch_size = default_batch_size

    if cfg.job_id is None:
        model_name = cfg.model_id.replace("/", "-")
        cfg.job_id = f"{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    print(f"Detected model type: {model_type}")
    print("Job Config:")
    print(cfg)

    runner = TranscriptionRunner(num_requests=cfg.num_requests, model_type=model_type)
    runner.run_transcription.remote(cfg)
