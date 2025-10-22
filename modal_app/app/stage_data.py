import traceback
import os
import numpy as np
import datasets
import soundfile
import modal

from app.common import (
    app, data_download_image, dataset_volume, DATASETS_VOLPATH
)
from utils.data import DEFAULT_MAX_THREADS, DATASETPATH_MODAL

@app.function(
    image=data_download_image,
    volumes={
        DATASETS_VOLPATH: dataset_volume,
    },
    cpu=DEFAULT_MAX_THREADS,
    timeout=60*60,
    secrets=[modal.Secret.from_dotenv()]
)
def download_hf_dataset(dataset_path, dataset_name, split):
    """
    Downloads a HuggingFace dataset and prepares it for Modal.

    Args:
        dataset_path: HuggingFace dataset path (e.g., 'astroza/es-cl-asr-test-only')
        dataset_name: Dataset config name (can be None for default config)
        split: Dataset split (e.g., 'test')

    Returns:
        Dictionary containing prepared dataset data
    """

    # Use dataset_path as the key for storage if dataset_name is None
    storage_key = dataset_name if dataset_name else dataset_path.split('/')[-1]
    dataset_path_dest = f"{DATASETPATH_MODAL}/{storage_key}/{split}"
    os.makedirs(dataset_path_dest, exist_ok=True)

    # Load dataset with or without config name
    if dataset_name:
        ds = datasets.load_dataset(dataset_path, dataset_name, split=split, token=True)
    else:
        ds = datasets.load_dataset(dataset_path, split=split, token=True)

    # Set audio backend to avoid torchcodec issues
    os.environ["DATASET_AUDIO_BACKEND"] = "soundfile"
    
    def prepare_data(batch):
        filenames = []
        filepaths = []
        durations = []

        for audio, id in zip(batch['audio'], batch['id']):
            # Make ID and wav filenames unique
            # Several datasets like earnings22 have a hierarchical structure
            # for eg. earnings22/test/4432298/281.wav, earnings22/test/4450488/281.wav
            filename = id.replace('/', '_') + ".wav"

            audiofile_path = f"{dataset_path_dest}/{filename}"

            audio_array = np.float32(audio["array"])
            sample_rate = audio["sampling_rate"]
            soundfile.write(audiofile_path, audio_array, sample_rate)

            duration = len(audio_array) / sample_rate

            filenames.append(filename)
            filepaths.append(audiofile_path)
            durations.append(duration)

        batch["filepath"] = filepaths
        batch["filename"] = filenames
        batch["audio_length_s"] = durations
        batch["split"] = [split] * len(filenames)
        return batch

    try:
        # Explicitly cast audio column to use soundfile backend
        ds = ds.cast_column("audio", datasets.Audio(sampling_rate=None, decode=True))
        
        ds = ds.map(
            prepare_data,
            batched=True,
            batch_size = len(ds)//DEFAULT_MAX_THREADS if len(ds) > DEFAULT_MAX_THREADS else len(ds),
            num_proc=DEFAULT_MAX_THREADS,
            remove_columns="audio"
        )
        ds.to_csv(f"{dataset_path_dest}/features.csv")
        return ds.to_dict()
    except Exception as e:
        print(f"Error downloading {dataset_name} {split}: {e}")
        print(traceback.format_exc())
        return None
