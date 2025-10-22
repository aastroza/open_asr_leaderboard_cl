import argparse
from typing import Optional
import datasets
import evaluate
import soundfile as sf
import tempfile
import time
import os
import requests
import itertools
from tqdm import tqdm
from dotenv import load_dotenv
from io import BytesIO
import pandas as pd
from elevenlabs.client import ElevenLabs
from normalizer import data_utils
import concurrent.futures

load_dotenv()


def fetch_audio_urls(dataset_path, dataset, split, batch_size=100, max_retries=20):
    API_URL = "https://datasets-server.huggingface.co/rows"

    size_url = f"https://datasets-server.huggingface.co/size?dataset={dataset_path}&config={dataset}&split={split}"
    
    # Try to get size with retries and better error handling
    for attempt in range(max_retries):
        try:
            headers = {}
            if os.environ.get("HF_TOKEN") is not None:
                headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
            
            size_response = requests.get(size_url, headers=headers)
            size_response.raise_for_status()
            size_data = size_response.json()
            
            # Handle different response structures
            if "size" in size_data:
                if "config" in size_data["size"] and dataset in size_data["size"]["config"]:
                    total_rows = size_data["size"]["config"][dataset]["num_rows"]
                elif "splits" in size_data["size"]:
                    # Try to find the split in the splits list
                    splits = size_data["size"]["splits"]
                    split_info = next((s for s in splits if s["split"] == split), None)
                    if split_info:
                        total_rows = split_info["num_rows"]
                    else:
                        raise ValueError(f"Split '{split}' not found in available splits: {[s['split'] for s in splits]}")
                else:
                    # Fallback: try to get total rows directly
                    total_rows = size_data["size"].get("num_rows", None)
                    if total_rows is None:
                        raise ValueError(f"Could not determine number of rows from size response: {size_data}")
            else:
                # If no size field, try to get first batch to estimate
                print("Warning: Could not get exact dataset size, using batch approach")
                total_rows = None
                break
                
            print(f"Dataset size: {total_rows} rows")
            break
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed to get dataset size: {e}")
            if attempt == max_retries - 1:
                print("Warning: Could not get dataset size, proceeding without size information")
                total_rows = None
            else:
                time.sleep(2)
    
    # Fetch data in batches
    offset = 0
    fetched_count = 0
    
    while True:
        params = {
            "dataset": dataset_path,
            "config": dataset,
            "split": split,
            "offset": offset,
            "length": batch_size,
        }

        retries = 0
        batch_data = None
        
        while retries <= max_retries:
            try:
                headers = {}
                if os.environ.get("HF_TOKEN") is not None:
                    headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
                    
                response = requests.get(API_URL, params=params, headers=headers)
                response.raise_for_status()
                batch_data = response.json()
                
                if "rows" in batch_data and len(batch_data["rows"]) > 0:
                    yield from batch_data["rows"]
                    fetched_count += len(batch_data["rows"])
                    
                    # If we got less than batch_size, we're probably at the end
                    if len(batch_data["rows"]) < batch_size:
                        print(f"Fetched {fetched_count} total rows")
                        return
                        
                    offset += batch_size
                    break
                else:
                    # No more data
                    print(f"Fetched {fetched_count} total rows")
                    return
                    
            except (requests.exceptions.RequestException, ValueError) as e:
                retries += 1
                print(f"Error fetching batch at offset {offset}: {e}, retrying ({retries}/{max_retries})...")
                time.sleep(2)
                if retries > max_retries:
                    print(f"Max retries exceeded for batch at offset {offset}")
                    return
                    
        if batch_data is None:
            print(f"Failed to fetch batch at offset {offset}")
            return


def transcribe_with_retry(
    model_name: str,
    audio_file_path: Optional[str],
    sample: dict,
    max_retries=10,
    use_url=False,
):
    """
    Transcribe audio using ElevenLabs API with Spanish language support.
    """
    retries = 0
    while retries <= max_retries:
        try:
            if not model_name.startswith("elevenlabs/"):
                raise ValueError("Only ElevenLabs models are supported. Use 'elevenlabs/' prefix.")
            
            client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
            
            if use_url:
                response = requests.get(sample["row"]["audio"][0]["src"])
                audio_data = BytesIO(response.content)
                transcription = client.speech_to_text.convert(
                    file=audio_data,
                    model_id=model_name.split("/")[1],
                    language_code="es",  # Spanish language code
                    tag_audio_events=True,
                )
            else:
                with open(audio_file_path, "rb") as audio_file:
                    transcription = client.speech_to_text.convert(
                        file=audio_file,
                        model_id=model_name.split("/")[1],
                        language_code="es",  # Spanish language code
                        tag_audio_events=True,
                    )
            return transcription.text

        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise e

            if not use_url:
                sf.write(
                    audio_file_path,
                    sample["audio"]["array"],
                    sample["audio"]["sampling_rate"],
                    format="WAV",
                )
            delay = 1
            print(
                f"API Error: {str(e)}. Retrying in {delay}s... (Attempt {retries}/{max_retries})"
            )
            time.sleep(delay)


def transcribe_dataset(
    dataset_path,
    dataset,
    split,
    model_name,
    use_url=False,
    max_samples=None,
    max_workers=4,
):
    if use_url:
        audio_rows = fetch_audio_urls(dataset_path, dataset, split)
        if max_samples:
            audio_rows = itertools.islice(audio_rows, max_samples)
        ds = audio_rows
    else:
        ds = datasets.load_dataset(dataset_path, dataset, split=split, streaming=False)
        ds = data_utils.prepare_data(ds)
        if max_samples:
            ds = ds.take(max_samples)

    results = {
        "references": [],
        "predictions": [],
        "audio_length_s": [],
        "transcription_time_s": [],
        "dataset_types": [],  # New field for dataset classification
    }

    print(f"Transcribing with model: {model_name}")

    def infer_dataset_from_audio_info(sample):
        """
        Infer dataset type based on sample information.
        For Chilean Spanish data, try to categorize by content type.
        """
        # Try to get info from filename or other metadata
        if use_url:
            # For URL-based samples, try to infer from row data
            row_data = sample.get("row", {})
            # You can customize this based on your dataset structure
            if "tedx" in str(row_data).lower():
                return "tedx"
            elif "news" in str(row_data).lower() or "noticias" in str(row_data).lower():
                return "news"
            elif "conversation" in str(row_data).lower() or "conv" in str(row_data).lower():
                return "conversation"
            else:
                return "other"
        else:
            # For local dataset samples, try to infer from text content or other fields
            text = sample.get("norm_text", "").lower()
            # You can add more sophisticated classification here
            if len(text) > 100:  # Longer texts might be lectures/talks
                return "lecture"
            elif len(text) < 20:  # Short texts might be commands/quick speech
                return "short_speech"
            else:
                return "general"

    def process_sample(sample):
        if use_url:
            reference = sample["row"]["text"].strip() or " "
            audio_duration = sample["row"]["audio_length_s"]
            dataset_type = infer_dataset_from_audio_info(sample)
            start = time.time()
            try:
                transcription = transcribe_with_retry(
                    model_name, None, sample, use_url=True
                )
            except Exception as e:
                print(f"Failed to transcribe after retries: {e}")
                return None

        else:
            reference = sample.get("norm_text", "").strip() or " "
            dataset_type = infer_dataset_from_audio_info(sample)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                sf.write(
                    tmpfile.name,
                    sample["audio"]["array"],
                    sample["audio"]["sampling_rate"],
                    format="WAV",
                )
                tmp_path = tmpfile.name
                audio_duration = (
                    len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
                )

            start = time.time()
            try:
                transcription = transcribe_with_retry(
                    model_name, tmp_path, sample, use_url=False
                )
            except Exception as e:
                print(f"Failed to transcribe after retries: {e}")
                os.unlink(tmp_path)
                return None
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                else:
                    print(f"File {tmp_path} does not exist")

        transcription_time = time.time() - start
        return reference, transcription, audio_duration, transcription_time, dataset_type

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(process_sample, sample): sample for sample in ds
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_sample),
            total=len(future_to_sample),
            desc="Transcribing",
        ):
            result = future.result()
            if result:
                reference, transcription, audio_duration, transcription_time, dataset_type = result
                results["predictions"].append(transcription)
                results["references"].append(reference)
                results["audio_length_s"].append(audio_duration)
                results["transcription_time_s"].append(transcription_time)
                results["dataset_types"].append(dataset_type)

    results["predictions"] = [
        data_utils.normalizer(transcription) or " "
        for transcription in results["predictions"]
    ]
    results["references"] = [
        data_utils.normalizer(reference) or " " for reference in results["references"]
    ]

    # Save overall results
    manifest_path = data_utils.write_manifest(
        results["references"],
        results["predictions"],
        model_name.replace("/", "-"),
        dataset_path,
        dataset,
        split,
        audio_length=results["audio_length_s"],
        transcription_time=results["transcription_time_s"],
    )

    print("Results saved at path:", manifest_path)

    # Calculate overall metrics
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(
        references=results["references"], predictions=results["predictions"]
    )
    wer_percent = round(100 * wer, 2)
    rtfx = round(
        sum(results["audio_length_s"]) / sum(results["transcription_time_s"]), 2
    )

    print("OVERALL RESULTS:")
    print("WER:", wer_percent, "%")
    print("RTFx:", rtfx)

    # Create detailed results by dataset type
    if results["dataset_types"]:
        print("\n" + "="*60)
        print("RESULTS BY DATASET TYPE:")
        print("="*60)
        
        # Create DataFrame for easier analysis
        import pandas as pd
        df = pd.DataFrame({
            'dataset_type': results["dataset_types"],
            'reference': results["references"],
            'prediction': results["predictions"],
            'audio_length_s': results["audio_length_s"],
            'transcription_time_s': results["transcription_time_s"]
        })
        
        dataset_summary = []
        for dataset_type in df['dataset_type'].unique():
            subset = df[df['dataset_type'] == dataset_type]
            
            # Calculate WER for this subset
            subset_wer = wer_metric.compute(
                references=subset['reference'].tolist(), 
                predictions=subset['prediction'].tolist()
            )
            subset_wer_percent = round(100 * subset_wer, 2)
            
            # Calculate RTFx for this subset
            total_audio = subset['audio_length_s'].sum()
            total_time = subset['transcription_time_s'].sum()
            subset_rtfx = round(total_audio / total_time, 2) if total_time > 0 else 0
            
            num_samples = len(subset)
            
            dataset_summary.append({
                'dataset_type': dataset_type,
                'num_samples': num_samples,
                'wer_percent': subset_wer_percent,
                'rtfx': subset_rtfx,
                'total_audio_length_s': total_audio,
                'total_transcription_time_s': total_time
            })
            
            print(f"\nDataset: {dataset_type}")
            print(f"  Samples: {num_samples}")
            print(f"  WER: {subset_wer_percent}%")
            print(f"  RTFx: {subset_rtfx}")
            print(f"  Total Audio: {total_audio:.1f}s ({total_audio/60:.1f}min)")
        
        # Save dataset summary to CSV
        summary_df = pd.DataFrame(dataset_summary)
        summary_path = manifest_path.replace('.jsonl', '_by_dataset.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nDataset summary saved to: {summary_path}")
        
        return summary_df
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ElevenLabs Spanish Transcription Script with Dataset Analysis"
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--model_name",
        required=True,
        help="ElevenLabs model name with 'elevenlabs/' prefix (e.g., 'elevenlabs/eleven_multilingual_v2')",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--max_workers", type=int, default=10, help="Number of concurrent threads (reduced for API limits)"
    )
    parser.add_argument(
        "--use_url",
        action="store_true",
        help="Use URL-based audio fetching instead of datasets",
    )

    args = parser.parse_args()

    transcribe_dataset(
        dataset_path=args.dataset_path,
        dataset=args.dataset,
        split=args.split,
        model_name=args.model_name,
        use_url=args.use_url,
        max_samples=args.max_samples,
        max_workers=args.max_workers,
    )
