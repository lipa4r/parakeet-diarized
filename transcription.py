import os
import logging
import tempfile
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np

from models import WhisperSegment, TranscriptionResponse

logger = logging.getLogger(__name__)


def _apply_decoder_config(model, use_cuda_graph_decoder: bool) -> None:
    """
    Toggle the CUDA graph decoder on a NeMo RNNT/TDT model.

    Idempotent — tracks the last applied value on the model and skips the
    rebuild when nothing changed. Stream capture for greedy decoding is
    unstable on bleeding-edge stacks (e.g. PyTorch 2.11 + CUDA 13 + Blackwell
    sm_120) where it surfaces as intermittent "illegal memory access" crashes
    in currentStreamCaptureStatusMayInitCtx.
    """
    current = getattr(model, "_use_cuda_graph_decoder", None)
    if current == use_cuda_graph_decoder:
        return
    try:
        from omegaconf import open_dict
        decoding_cfg = model.cfg.decoding
        with open_dict(decoding_cfg):
            if "greedy" in decoding_cfg:
                decoding_cfg.greedy.use_cuda_graph_decoder = use_cuda_graph_decoder
        model.change_decoding_strategy(decoding_cfg)
        model._use_cuda_graph_decoder = use_cuda_graph_decoder
        logger.info(f"CUDA graph decoder set to {use_cuda_graph_decoder}")
    except Exception as e:
        logger.warning(f"Could not change CUDA graph decoder: {e}")


def compute_batch_size(gpu_mem_gb: float, chunk_duration_s: int) -> int:
    """
    Estimate a safe batch size for model.transcribe() from available GPU memory.

    Heuristic (Parakeet-TDT 0.6B):
      - model weights + NeMo runtime: ~3.5 GB baseline
      - FastConformer activation memory: ~1 GB per 300 s of audio

    Capped at 8 to leave headroom; returns at least 1.
    """
    model_overhead_gb = 3.5
    mem_per_chunk_gb = max(0.5, chunk_duration_s / 300.0)
    available_gb = gpu_mem_gb - model_overhead_gb
    if available_gb <= 0:
        return 1
    return max(1, min(int(available_gb / mem_per_chunk_gb), 8))


def load_model(model_id: str = "nvidia/parakeet-tdt-0.6b-v3",
               use_cuda_graph_decoder: bool = False):
    """
    Load the ASR model (Parakeet-TDT)

    Args:
        model_id: The HuggingFace model ID to load
        use_cuda_graph_decoder: Enable NeMo's RNNT/TDT greedy CUDA graph
            decoder. Defaults to False because stream capture is unstable
            on some new GPU/torch combinations.

    Returns:
        The loaded model
    """
    try:
        import nemo.collections.asr as nemo_asr

        logger.info(f"Loading model {model_id}")
        # Use ASRModel base class — auto-detects the correct subclass (CTC/TDT/RNNT)
        # from the checkpoint config, so it works with any NeMo ASR model.
        model = nemo_asr.models.ASRModel.from_pretrained(model_id)

        if torch.cuda.is_available():
            model = model.cuda()
            logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available, running on CPU (will be slow)")

        _apply_decoder_config(model, use_cuda_graph_decoder)

        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def _format_timestamp(seconds: float, always_include_hours: bool = False,
                      decimal_marker: str = '.') -> str:
    hours = int(seconds / 3600)
    seconds = seconds % 3600
    minutes = int(seconds / 60)
    seconds = seconds % 60
    hours_marker = f"{hours}:" if always_include_hours or hours > 0 else ""
    if decimal_marker == ',':
        return f"{hours_marker}{minutes:02d}:{seconds:06.3f}".replace('.', decimal_marker)
    return f"{hours_marker}{minutes:02d}:{seconds:06.3f}"


def format_srt(segments: List[WhisperSegment]) -> str:
    srt_content = ""
    for i, segment in enumerate(segments):
        segment_id = i + 1
        start = _format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')
        end = _format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')
        text = segment.text.strip().replace('-->', '->')
        speaker_prefix = f"[{segment.speaker}] " if hasattr(segment, "speaker") and segment.speaker else ""
        srt_content += f"{segment_id}\n{start} --> {end}\n{speaker_prefix}{text}\n\n"
    return srt_content.strip()


def format_vtt(segments: List[WhisperSegment]) -> str:
    vtt_content = "WEBVTT\n\n"
    for i, segment in enumerate(segments):
        start = _format_timestamp(segment.start, always_include_hours=True)
        end = _format_timestamp(segment.end, always_include_hours=True)
        text = segment.text.strip()
        speaker_prefix = f"<v {segment.speaker}>" if hasattr(segment, "speaker") and segment.speaker else ""
        vtt_content += f"{start} --> {end}\n{speaker_prefix}{text}\n\n"
    return vtt_content.strip()


def _parse_nemo_results(
    transcriptions, chunk_paths: List[str]
) -> List[Tuple[str, List[WhisperSegment]]]:
    """Convert NeMo transcription results into (text, segments) pairs."""
    results = []
    for i, result in enumerate(transcriptions):
        text = getattr(result, 'text', '') or ''
        ts = getattr(result, 'timestamp', None)
        segments: List[WhisperSegment] = []
        if ts is not None and isinstance(ts, dict) and ts.get('segment'):
            for j, stamp in enumerate(ts['segment']):
                segments.append(WhisperSegment(
                    id=j,
                    start=stamp['start'],
                    end=stamp['end'],
                    text=stamp['segment'],
                ))
        else:
            segments.append(WhisperSegment(
                id=0,
                start=0.0,
                end=len(text.split()) / 2.0,
                text=text,
            ))
        results.append((text, segments))
    return results


def transcribe_audio_chunks(
    model,
    chunk_paths: List[str],
    batch_size: int = 1,
    language: Optional[str] = None,
    word_timestamps: bool = False,
    force_fp32: bool = False,
    use_cuda_graph_decoder: Optional[bool] = None,
) -> List[Tuple[str, List[WhisperSegment]]]:
    """
    Transcribe a list of audio chunks in a single batched model.transcribe() call.

    When multiple chunks exist NeMo processes them as a batch on the GPU,
    which is significantly faster than sequential single-file calls.
    batch_size controls how many chunks are fed to the model at once;
    use compute_batch_size() to derive a safe value from GPU memory.

    Returns a list of (text, segments) tuples in the same order as chunk_paths.
    Timestamps inside each tuple are relative to the start of that chunk;
    the caller is responsible for adding inter-chunk offsets.
    """
    if not chunk_paths:
        return []

    if use_cuda_graph_decoder is not None:
        _apply_decoder_config(model, use_cuda_graph_decoder)

    autocast_ctx = (
        torch.amp.autocast("cuda", enabled=False)
        if force_fp32 and torch.cuda.is_available()
        else nullcontext()
    )

    try:
        with torch.no_grad(), autocast_ctx:
            # num_workers=0 keeps DataLoader in-process — avoids semaphore leaks
            transcriptions = model.transcribe(
                chunk_paths,
                batch_size=batch_size,
                timestamps=True,
                num_workers=0,
            )

        if not transcriptions:
            logger.warning(f"model.transcribe() returned empty for {len(chunk_paths)} chunk(s)")
            return [("", []) for _ in chunk_paths]

        return _parse_nemo_results(transcriptions, chunk_paths)

    except Exception as e:
        logger.error(f"Error transcribing {len(chunk_paths)} chunk(s) with batch_size={batch_size}: {e}")
        return [("", []) for _ in chunk_paths]


def transcribe_audio_chunk(
    model,
    audio_path: str,
    language: Optional[str] = None,
    word_timestamps: bool = False,
    force_fp32: bool = False,
    use_cuda_graph_decoder: Optional[bool] = None,
) -> Tuple[str, List[WhisperSegment]]:
    """Single-chunk convenience wrapper around transcribe_audio_chunks."""
    results = transcribe_audio_chunks(
        model, [audio_path], batch_size=1,
        language=language, word_timestamps=word_timestamps,
        force_fp32=force_fp32, use_cuda_graph_decoder=use_cuda_graph_decoder,
    )
    return results[0] if results else ("", [])
