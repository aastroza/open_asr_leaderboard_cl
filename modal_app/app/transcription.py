import modal
from pathlib import Path
import pandas as pd
import time
import logging

from app.common import (
    app,
    nemo_transcription_image,
    transformers_transcription_image,
    voxtral_transcription_image,
    phi4_multimodal_image,
    elevenlabs_transcription_image,
    omnilingual_transcription_image,
    runner_image,
    dataset_volume,
    model_volume,
    results_volume,
    DATASETS_VOLPATH,
    MODELS_VOLPATH,
    RESULTS_VOLPATH
)
from utils.data import DATASET_STORAGE_NAME, DATASETPATH_MODAL

MINUTES = 60 # seconds

# ============================================================================
# NeMo ASR Models (Canary, Parakeet)
# ============================================================================

with nemo_transcription_image.imports():
    import nemo.collections.asr as nemo_asr
    import torch
    import evaluate
    import utils.normalizer.data_utils as du
    from pathlib import Path
    import time
    from utils.data import copy_concurrent


@app.cls(
    image=nemo_transcription_image,
    timeout=60*MINUTES,
    volumes={
        DATASETS_VOLPATH: dataset_volume,
        MODELS_VOLPATH: model_volume,
        RESULTS_VOLPATH: results_volume,
    },
    scaledown_window=5,
)
class NeMoAsrBatchTranscription():
    DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
    DEFAULT_GPU_TYPE = "L40S"
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_NUM_REQUESTS = 10
    model_id: str = modal.parameter(default=DEFAULT_MODEL_ID)
    gpu_batch_size: int = modal.parameter(default=DEFAULT_BATCH_SIZE)


    @modal.enter()
    def setup(self):

        self._COMPUTE_DTYPE = torch.bfloat16

        # silence chatty logs from nemo
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(self.model_id)
        self.asr_model.to(self._COMPUTE_DTYPE)
        self.asr_model.eval()

        # Configure decoding strategy
        if self.asr_model.cfg.decoding.strategy != "beam":
            self.asr_model.cfg.decoding.strategy = "greedy_batch"
            self.asr_model.change_decoding_strategy(self.asr_model.cfg.decoding)

    @modal.method()
    async def run_inference(self, audio_filepaths):

        local_filepaths = [path.replace(DATASETPATH_MODAL, '/tmp') for path in audio_filepaths]
        filenames = [filepath.split('/')[-1] for filepath in local_filepaths]

        copy_concurrent(Path(DATASETPATH_MODAL), Path('/tmp/'), filenames)

        start_time = time.perf_counter()
        with torch.autocast("cuda", enabled=False, dtype=self._COMPUTE_DTYPE), torch.inference_mode(), torch.no_grad():
            if 'canary' in self.model_id:
                # Canary v2 uses pnc, v1 uses nopnc
                pnc = 'pnc' if 'v2' in self.model_id else 'nopnc'
                transcriptions = self.asr_model.transcribe(local_filepaths, batch_size=self.gpu_batch_size, verbose=False, pnc=pnc, num_workers=1, source_lang='es', target_lang='es')
            else:
                transcriptions = self.asr_model.transcribe(local_filepaths, batch_size=self.gpu_batch_size, num_workers=1)

        total_time = time.perf_counter() - start_time
        print("Total time:", total_time)

        # Process transcriptions
        if isinstance(transcriptions, tuple) and len(transcriptions) == 2:
            transcriptions = transcriptions[0]
        predictions = [pred.text for pred in transcriptions]

        return {
            "num_samples": len(filenames),
            "transcriptions": predictions,
            "total_time": total_time,
        }


# ============================================================================
# Transformers Models (Whisper)
# ============================================================================

with transformers_transcription_image.imports():
    import torch
    from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
    import evaluate
    import utils.normalizer.data_utils as du
    from pathlib import Path
    import time
    from utils.data import copy_concurrent
    import soundfile as sf


@app.cls(
    image=transformers_transcription_image,
    timeout=60*MINUTES,
    volumes={
        DATASETS_VOLPATH: dataset_volume,
        MODELS_VOLPATH: model_volume,
        RESULTS_VOLPATH: results_volume,
    },
    scaledown_window=5,
)
class TransformersAsrBatchTranscription():
    DEFAULT_MODEL_ID = "openai/whisper-large-v3"
    DEFAULT_GPU_TYPE = "L40S"
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_NUM_REQUESTS = 10
    model_id: str = modal.parameter(default=DEFAULT_MODEL_ID)
    gpu_batch_size: int = modal.parameter(default=DEFAULT_BATCH_SIZE)


    @modal.enter()
    def setup(self):

        self._COMPUTE_DTYPE = torch.bfloat16
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Load model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self._COMPUTE_DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(device)

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self._COMPUTE_DTYPE,
            device=device,
        )

    @modal.method()
    async def run_inference(self, audio_filepaths):

        local_filepaths = [path.replace(DATASETPATH_MODAL, '/tmp') for path in audio_filepaths]
        filenames = [filepath.split('/')[-1] for filepath in local_filepaths]

        copy_concurrent(Path(DATASETPATH_MODAL), Path('/tmp/'), filenames)

        # Load audio files
        audio_data = []
        for filepath in local_filepaths:
            audio_array, sample_rate = sf.read(filepath)
            audio_data.append({"array": audio_array, "sampling_rate": sample_rate})

        start_time = time.perf_counter()

        # Process in batches
        transcriptions = []
        for i in range(0, len(audio_data), self.gpu_batch_size):
            batch = audio_data[i:i + self.gpu_batch_size]
            results = self.pipe(batch, batch_size=self.gpu_batch_size, generate_kwargs={"language": "spanish"})
            transcriptions.extend([r["text"] for r in results])

        total_time = time.perf_counter() - start_time
        print("Total time:", total_time)

        return {
            "num_samples": len(filenames),
            "transcriptions": transcriptions,
            "total_time": total_time,
        }


# ============================================================================
# Voxtral Models
# ============================================================================

with voxtral_transcription_image.imports():
    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    import evaluate
    import utils.normalizer.data_utils as du
    from pathlib import Path
    import time
    from utils.data import copy_concurrent


@app.cls(
    image=voxtral_transcription_image,
    timeout=60*MINUTES,
    volumes={
        DATASETS_VOLPATH: dataset_volume,
        MODELS_VOLPATH: model_volume,
        RESULTS_VOLPATH: results_volume,
    },
    scaledown_window=5,
)
class VoxtralAsrBatchTranscription():
    DEFAULT_MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
    DEFAULT_GPU_TYPE = "L40S"
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_NUM_REQUESTS = 10
    model_id: str = modal.parameter(default=DEFAULT_MODEL_ID)
    gpu_batch_size: int = modal.parameter(default=DEFAULT_BATCH_SIZE)


    @modal.enter()
    def setup(self):

        self._COMPUTE_DTYPE = torch.bfloat16
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Voxtral uses VoxtralForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self._COMPUTE_DTYPE,
            device_map=self.device
        )

    @modal.method()
    async def run_inference(self, audio_filepaths):

        local_filepaths = [path.replace(DATASETPATH_MODAL, '/tmp') for path in audio_filepaths]
        filenames = [filepath.split('/')[-1] for filepath in local_filepaths]

        copy_concurrent(Path(DATASETPATH_MODAL), Path('/tmp/'), filenames)

        start_time = time.perf_counter()

        # Voxtral-specific transcription using apply_transcription_request
        transcriptions = []
        for filepath in local_filepaths:
            # Process each audio file individually for Voxtral
            inputs = self.processor.apply_transcription_request(
                language="es",  # Spanish target language
                audio=filepath,
                model_id=self.model_id
            )
            inputs = inputs.to(self.device, dtype=self._COMPUTE_DTYPE)

            # Generate transcription
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=500)

            # Decode output
            decoded = self.processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            transcriptions.extend(decoded)

        total_time = time.perf_counter() - start_time
        print("Total time:", total_time)

        return {
            "num_samples": len(filenames),
            "transcriptions": transcriptions,
            "total_time": total_time,
        }


# ============================================================================
# Phi-4 Multimodal Models
# ============================================================================

with phi4_multimodal_image.imports():
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
    import evaluate
    import utils.normalizer.data_utils as du
    from pathlib import Path
    import time
    from utils.data import copy_concurrent
    import soundfile


@app.cls(
    image=phi4_multimodal_image,
    timeout=60*MINUTES,
    volumes={
        DATASETS_VOLPATH: dataset_volume,
        MODELS_VOLPATH: model_volume,
        RESULTS_VOLPATH: results_volume,
    },
    scaledown_window=5,
)
class Phi4MultimodalAsrBatchTranscription():
    DEFAULT_MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
    DEFAULT_GPU_TYPE = "L40S"
    DEFAULT_BATCH_SIZE = 4
    DEFAULT_NUM_REQUESTS = 10
    model_id: str = modal.parameter(default=DEFAULT_MODEL_ID)
    gpu_batch_size: int = modal.parameter(default=DEFAULT_BATCH_SIZE)


    @modal.enter()
    def setup(self):

        self._COMPUTE_DTYPE = torch.bfloat16
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Initialize Phi-4 multimodal model
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype='auto',
            _attn_implementation='flash_attention_2',
        ).cuda()

        self.generation_config = GenerationConfig.from_pretrained(
            self.model_id,
            'generation_config.json'
        )

        # Define prompts for Phi-4
        self.user_prompt = '<|user|>'
        self.assistant_prompt = '<|assistant|>'
        self.prompt_suffix = '<|end|>'

    @modal.method()
    async def run_inference(self, audio_filepaths):

        local_filepaths = [path.replace(DATASETPATH_MODAL, '/tmp') for path in audio_filepaths]
        filenames = [filepath.split('/')[-1] for filepath in local_filepaths]

        copy_concurrent(Path(DATASETPATH_MODAL), Path('/tmp/'), filenames)

        start_time = time.perf_counter()

        # Phi-4 transcription - process each file individually
        transcriptions = []
        for filepath in local_filepaths:
            # Transcribe Spanish audio to text
            speech_prompt = "Transcribe este audio en español a texto."

            prompt = f'{self.user_prompt}<|audio_1|>{speech_prompt}{self.prompt_suffix}{self.assistant_prompt}'
            audio = soundfile.read(filepath)

            inputs = self.processor(text=prompt, audios=[audio], return_tensors='pt').to('cuda')

            # Generate transcription
            with torch.inference_mode():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2000,
                    generation_config=self.generation_config,
                )

            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

            transcription = self.processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            transcriptions.append(transcription.strip())

        total_time = time.perf_counter() - start_time
        print("Total time:", total_time)

        return {
            "num_samples": len(filenames),
            "transcriptions": transcriptions,
            "total_time": total_time,
        }


# ============================================================================
# ElevenLabs API
# ============================================================================

with elevenlabs_transcription_image.imports():
    from elevenlabs.client import ElevenLabs
    import soundfile as sf
    import os
    from pathlib import Path
    import time
    from utils.data import copy_concurrent
    from io import BytesIO


@app.cls(
    image=elevenlabs_transcription_image,
    timeout=60*MINUTES,
    volumes={
        DATASETS_VOLPATH: dataset_volume,
        RESULTS_VOLPATH: results_volume,
    },
    scaledown_window=5,
    secrets=[modal.Secret.from_name("elevenlabs-api-key")],
)
class ElevenLabsAsrBatchTranscription():
    DEFAULT_MODEL_ID = "scribe_v1"
    DEFAULT_BATCH_SIZE = 10  # API calls, not GPU batches
    DEFAULT_NUM_REQUESTS = 10
    model_id: str = modal.parameter(default=DEFAULT_MODEL_ID)
    batch_size: int = modal.parameter(default=DEFAULT_BATCH_SIZE)

    @modal.enter()
    def setup(self):
        # Initialize ElevenLabs client with API key from secret
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment")
        self.client = ElevenLabs(api_key=api_key)
        
        # Extract actual model name from HF-style format if needed
        # Convert "elevenlabs/scribe_v1" -> "scribe_v1"
        if "/" in self.model_id:
            self.actual_model_id = self.model_id.split("/")[-1]
        else:
            self.actual_model_id = self.model_id

    @modal.method()
    async def run_inference(self, audio_filepaths):
        """
        Transcribe audio files using ElevenLabs API.

        Args:
            audio_filepaths: List of audio file paths in the dataset volume

        Returns:
            Dictionary with transcriptions, num_samples, and timing info
        """
        local_filepaths = [path.replace(DATASETPATH_MODAL, '/tmp') for path in audio_filepaths]
        filenames = [filepath.split('/')[-1] for filepath in local_filepaths]

        # Copy audio files from volume to local tmp
        copy_concurrent(Path(DATASETPATH_MODAL), Path('/tmp/'), filenames)

        transcriptions = []
        start_time = time.perf_counter()

        # Process each audio file
        for filepath in local_filepaths:
            max_retries = 5
            retry_count = 0

            while retry_count <= max_retries:
                try:
                    # Read audio file
                    with open(filepath, "rb") as audio_file:
                        transcription = self.client.speech_to_text.convert(
                            file=audio_file,
                            model_id=self.actual_model_id,
                            language_code="es",  # Spanish language code
                        )
                    transcriptions.append(transcription.text)
                    break  # Success, exit retry loop

                except Exception as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"Failed to transcribe {filepath} after {max_retries} retries: {e}")
                        transcriptions.append("")  # Add empty transcription on failure
                    else:
                        print(f"Retry {retry_count}/{max_retries} for {filepath}: {e}")
                        time.sleep(1 * retry_count)  # Exponential backoff

        total_time = time.perf_counter() - start_time
        print(f"Total time: {total_time}")

        return {
            "num_samples": len(filenames),
            "transcriptions": transcriptions,
            "total_time": total_time,
        }


# ============================================================================
# OmniLingual ASR Models
# ============================================================================

with omnilingual_transcription_image.imports():
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    import evaluate
    import utils.normalizer.data_utils as du
    from pathlib import Path
    import time
    from utils.data import copy_concurrent


@app.cls(
    image=omnilingual_transcription_image,
    timeout=60*MINUTES,
    volumes={
        DATASETS_VOLPATH: dataset_volume,
        MODELS_VOLPATH: model_volume,
        RESULTS_VOLPATH: results_volume,
    },
    scaledown_window=5,
)
class OmnilingualAsrBatchTranscription():
    DEFAULT_MODEL_ID = "omniASR_LLM_7B"
    DEFAULT_GPU_TYPE = "L40S"
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_NUM_REQUESTS = 10
    model_id: str = modal.parameter(default=DEFAULT_MODEL_ID)
    gpu_batch_size: int = modal.parameter(default=DEFAULT_BATCH_SIZE)

    @modal.enter()
    def setup(self):
        # Initialize OmniLingual ASR pipeline
        # Extract model card from model_id (could be "omniASR_LLM_7B" or full path)
        if "/" in self.model_id:
            # If it's a full path like "facebook/omniASR_LLM_7B", use the last part
            model_card = self.model_id.split("/")[-1]
        else:
            model_card = self.model_id

        self.pipeline = ASRInferencePipeline(model_card=model_card)

    @modal.method()
    async def run_inference(self, audio_filepaths):
        local_filepaths = [path.replace(DATASETPATH_MODAL, '/tmp') for path in audio_filepaths]
        filenames = [filepath.split('/')[-1] for filepath in local_filepaths]

        copy_concurrent(Path(DATASETPATH_MODAL), Path('/tmp/'), filenames)

        start_time = time.perf_counter()

        # OmniLingual ASR transcription
        # The language code for Spanish (Latin America) is spa_Latn
        lang = ["spa_Latn"] * len(local_filepaths)

        transcriptions = self.pipeline.transcribe(
            local_filepaths,
            lang=lang,
            batch_size=self.gpu_batch_size
        )

        total_time = time.perf_counter() - start_time
        print("Total time:", total_time)

        return {
            "num_samples": len(filenames),
            "transcriptions": transcriptions,
            "total_time": total_time,
        }


# ============================================================================
# Runner for Scoring and Saving Results
# ============================================================================

with runner_image.imports():
    import evaluate
    import utils.normalizer.data_utils as du
    from utils.data import distribute_audio


@app.cls(
    image=runner_image,
    timeout=60*60,
    volumes={
        DATASETS_VOLPATH: dataset_volume,
        RESULTS_VOLPATH: results_volume,
    },
)
class TranscriptionRunner():
    num_requests: int = modal.parameter()
    model_type: str = modal.parameter(default="nemo")  # "nemo", "transformers", "voxtral", "phi4_multimodal", "elevenlabs", or "omnilingual"

    @modal.method()
    def run_transcription(self, cfg):

        print(f"Starting transcription job: {cfg.job_id}...")
        start_time = time.perf_counter()
        batch_start_creation_time = time.perf_counter()

        # Load the dataset features
        data_df = pd.read_csv(f"{DATASETS_VOLPATH}/{DATASET_STORAGE_NAME}/{cfg.dataset}/{cfg.split}/features.csv")
        dfs = distribute_audio(data_df, self.num_requests)
        batch_creation_time = time.perf_counter() - batch_start_creation_time
        print(f"Batch creation time: {batch_creation_time} seconds")

        results = []

        # Choose the appropriate transcription class based on model type
        if self.model_type == "nemo":
            transcription_cls = NeMoAsrBatchTranscription.with_options(
                gpu=cfg.gpu_type,
            )(
                model_id=cfg.model_id,
                gpu_batch_size=cfg.gpu_batch_size,
            )
        elif self.model_type == "voxtral":
            transcription_cls = VoxtralAsrBatchTranscription.with_options(
                gpu=cfg.gpu_type,
            )(
                model_id=cfg.model_id,
                gpu_batch_size=cfg.gpu_batch_size,
            )
        elif self.model_type == "phi4_multimodal":
            transcription_cls = Phi4MultimodalAsrBatchTranscription.with_options(
                gpu=cfg.gpu_type,
            )(
                model_id=cfg.model_id,
                gpu_batch_size=cfg.gpu_batch_size,
            )
        elif self.model_type == "elevenlabs":
            # ElevenLabs doesn't need GPU, uses API calls
            transcription_cls = ElevenLabsAsrBatchTranscription(
                model_id=cfg.model_id,
                batch_size=cfg.gpu_batch_size,  # For API, this is batch size not GPU batch
            )
        elif self.model_type == "omnilingual":
            transcription_cls = OmnilingualAsrBatchTranscription.with_options(
                gpu=cfg.gpu_type,
            )(
                model_id=cfg.model_id,
                gpu_batch_size=cfg.gpu_batch_size,
            )
        else:  # transformers (Whisper)
            transcription_cls = TransformersAsrBatchTranscription.with_options(
                gpu=cfg.gpu_type,
            )(
                model_id=cfg.model_id,
                gpu_batch_size=cfg.gpu_batch_size,
            )

        print("Running inference...")
        for result in transcription_cls.run_inference.map([df['filepath'].tolist() for df in dfs]):
            results.append(result)

        total_runtime = time.perf_counter() - start_time
        print(f"Total runtime: {total_runtime} seconds")

        for result, df in zip(results, dfs):
            result['total_runtime'] = total_runtime
            result['job_id'] = cfg.job_id
            result['model_id'] = cfg.model_id
            result['audio_length_s'] = df['audio_length_s'].tolist()
            result['original_text'] = df['text'].tolist()
            # Use the actual dataset values from the DataFrame instead of cfg.dataset
            result['dataset'] = df['dataset'].tolist()
            result['split'] = [cfg.split] * len(df)

        print("Scoring results...")
        scored_results = []
        for scored_result in self.score_call.map(results):
            scored_results.append(scored_result)

        self.save_results(scored_results, cfg)

        print(f"Transcription job {cfg.job_id} complete.")

    @modal.method()
    def score_call(self, results):
        wer_metric = evaluate.load("wer")

        # Calculate metrics
        normalized_predictions = [du.normalizer(pred) for pred in results['transcriptions']]
        normalized_references = [du.normalizer(ref) for ref in results['original_text']]
        wer = wer_metric.compute(references=normalized_references, predictions=normalized_predictions)
        wer = round(100 * wer, 2)

        audio_length = sum(results['audio_length_s'])
        rtfx = audio_length / results['total_time']
        rtfx = round(rtfx, 2)

        results['wer'] = wer
        results['rtfx'] = rtfx
        results['total_audio_length'] = audio_length

        return results

    def save_results(self, results, cfg):
        # Expand the results properly - each result dict contains lists that need to be expanded
        expanded_rows = []
        for result in results:
            # Each result is a batch with lists of values
            num_samples = len(result['transcriptions'])
            for i in range(num_samples):
                row = {
                    'transcriptions': result['transcriptions'][i],
                    'original_text': result['original_text'][i],
                    'audio_length_s': result['audio_length_s'][i],
                    'dataset': result['dataset'][i],
                    'split': result['split'][i],
                    'total_runtime': result['total_runtime'],
                    'job_id': result['job_id'],
                    'model_id': result['model_id'],
                    'batch_wer': result['wer'],  # Keep batch-level metrics for reference
                    'batch_rtfx': result['rtfx'],
                    'batch_total_time': result['total_time']
                }
                expanded_rows.append(row)
        
        # Create DataFrame from expanded rows
        results_df = pd.DataFrame(expanded_rows)
        
        # Also create a batch-level summary (original format)
        batch_summary = []
        for result in results:
            batch_row = {
                'num_samples': len(result['transcriptions']),
                'total_time': result['total_time'],
                'total_runtime': result['total_runtime'],
                'job_id': result['job_id'],
                'model_id': result['model_id'],
                'wer': result['wer'],
                'rtfx': result['rtfx'],
                'total_audio_length': sum(result['audio_length_s'])
            }
            batch_summary.append(batch_row)
        
        batch_df = pd.DataFrame(batch_summary)

        results_summary_dir = Path(f"{RESULTS_VOLPATH}/results_summaries")
        results_summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Save batch-level summary (existing functionality)
        batch_df.to_csv(results_summary_dir / f"results_summary_{cfg.job_id}.csv", index=False)

        # Save aggregated results by dataset
        if 'dataset' in results_df.columns:
            print("Creating dataset summary...")
            print(f"Unique datasets: {results_df['dataset'].unique()}")
            
            # Calculate dataset-level metrics directly from results (like batch summary)
            dataset_summary_data = []
            wer_metric = evaluate.load("wer")
            
            # Group samples by dataset across all batches
            dataset_groups = {}
            for result in results:
                # Group samples from this batch by dataset
                for i in range(len(result['transcriptions'])):
                    dataset_name = result['dataset'][i]
                    if dataset_name not in dataset_groups:
                        dataset_groups[dataset_name] = {
                            'transcriptions': [],
                            'original_text': [],
                            'audio_length_s': [],
                            'total_time': 0
                        }
                    
                    dataset_groups[dataset_name]['transcriptions'].append(result['transcriptions'][i])
                    dataset_groups[dataset_name]['original_text'].append(result['original_text'][i])
                    dataset_groups[dataset_name]['audio_length_s'].append(result['audio_length_s'][i])
                    
                    # Add proportional time for this sample
                    proportional_time = result['total_time'] / len(result['transcriptions'])
                    dataset_groups[dataset_name]['total_time'] += proportional_time
            
            # Calculate metrics for each dataset
            for dataset_name, dataset_data in dataset_groups.items():
                # Calculate WER for this dataset
                normalized_predictions = [du.normalizer(pred) for pred in dataset_data['transcriptions']]
                normalized_references = [du.normalizer(ref) for ref in dataset_data['original_text']]
                dataset_wer = wer_metric.compute(references=normalized_references, predictions=normalized_predictions)
                dataset_wer = round(100 * dataset_wer, 2)
                
                # Calculate total audio length and RTFx
                total_audio_length = sum(dataset_data['audio_length_s'])
                total_time = dataset_data['total_time']
                rtfx = total_audio_length / total_time if total_time > 0 else 0
                
                dataset_summary_data.append({
                    'dataset': dataset_name,
                    'num_samples': len(dataset_data['transcriptions']),
                    'total_time': total_time,
                    'total_runtime': results[0]['total_runtime'],  # Same for all
                    'job_id': results[0]['job_id'],  # Same for all
                    'model_id': results[0]['model_id'],  # Same for all
                    'wer': dataset_wer,
                    'rtfx': round(rtfx, 2),
                    'total_audio_length': total_audio_length
                })
            
            dataset_summary = pd.DataFrame(dataset_summary_data)
            dataset_summary.to_csv(results_summary_dir / f"results_by_dataset_{cfg.job_id}.csv", index=False)
            print(f"Dataset summary saved to {results_summary_dir / f'results_by_dataset_{cfg.job_id}.csv'}")
            print("Dataset summary:")
            print(dataset_summary)

        results_dir = Path(f"{RESULTS_VOLPATH}/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_dir / f"results_{cfg.job_id}.csv", index=False)

        print(f"Results saved to {results_dir} and {results_summary_dir}")
