import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from app.common import DATASETS_VOLPATH

DEFAULT_MAX_THREADS = 32

# Chilean Spanish dataset configuration
CHILEAN_DATASET_NAME = "es-cl-asr-test-only"
CHILEAN_DATASET_PATH = "astroza"  # HuggingFace dataset path

DATASET_STORAGE_NAME = "chilean-spanish-asr-test"
DATASETPATH_HF = f"{CHILEAN_DATASET_PATH}/{CHILEAN_DATASET_NAME}"
DATASETPATH_MODAL = f"{DATASETS_VOLPATH}/{DATASET_STORAGE_NAME}"

# Dataset configuration for staging
DATASET_CONFIG = [
    (DATASETPATH_HF, CHILEAN_DATASET_NAME, "test"),
]

def distribute_audio(df, num_requests):
    """
    Distributes audio files across containers based on cumulative audio length.

    Args:
        df (pd.DataFrame): Input dataframe with 'audio_length_s' column
        num_requests (int): Number of batches to distribute job across (i.e. number of elements sent to `map`)

    Returns:
        list: List of dataframes, one for each container
    """

    print(f"Distributing {len(df)} audio files across {num_requests} containers")

    if df.empty:
        print("Input dataframe is empty")
        return []

    if 'audio_length_s' not in df.columns:
        print("Error: 'audio_length_s' column not found in dataframe")
        return []

    df = df.sample(frac=1).reset_index(drop=True)
    # split the DataFrame into `num_requests` batches of (roughly) equal size:
    batches = np.array_split(df, num_requests)

    assert len(batches) == num_requests, f"Expected {num_requests} batches, got {len(batches)}"

    # sort batches by audio length
    batches = [batch.sort_values('audio_length_s', ascending=False).reset_index(drop=True) for batch in batches]

    print(f"\nCreated {len(batches)} batches")
    return batches


def copy_concurrent(src: Path, dest: Path, filenames, max_threads: int = DEFAULT_MAX_THREADS) -> None:
    """
    A modified shutil.copytree which copies in parallel to increase bandwidth
    and maximize throughput for downloads of many individual files from Modal Volumes.
    """
    from multiprocessing.pool import ThreadPool

    max_threads = np.min([max_threads, len(filenames)])

    class MultithreadedCopier:
        def __init__(self, max_threads):
            self.pool = ThreadPool(max_threads)
            self.copy_jobs = []

        def copy(self, source, dest):
            res = self.pool.apply_async(
                shutil.copy2,
                args=(source, dest),
                # callback=lambda r: print(f"{source} copied to {dest}"),
                # NOTE: this should `raise` an exception for proper reliability.
                error_callback=lambda exc: print(
                    f"{source} failed: {exc}", file=sys.stderr
                ),
            )
            self.copy_jobs.append(res)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.pool.close()
            self.pool.join()

    def ignore_file(src, names):
        ignore_files = [n for n in names if (n.endswith('.wav') and n not in filenames) or n.endswith('.lock')]
        return ignore_files

    with MultithreadedCopier(max_threads=max_threads) as copier:
        shutil.copytree(src, dest, copy_function=copier.copy, dirs_exist_ok=True, ignore=ignore_file)
