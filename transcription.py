import os
import logging
import tempfile
from contextlib import nullcontext
from typing import List, Optional, Dict, Any, Union, Tuple

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

        # Move model to GPU if available
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
    """
    Format a timestamp as a string (HH:MM:SS.mmm)

    Args:
        seconds: Time in seconds
        always_include_hours: Always include hours in the output
        decimal_marker: Marker to use for decimal point

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds / 3600)
    seconds = seconds % 3600
    minutes = int(seconds / 60)
    seconds = seconds % 60

    hours_marker = f"{hours}:" if always_include_hours or hours > 0 else ""

    # Handle different format requirements (SRT vs VTT)
    if decimal_marker == ',':  # SRT format
        return f"{hours_marker}{minutes:02d}:{seconds:06.3f}".replace('.', decimal_marker)
    else:  # VTT format
        return f"{hours_marker}{minutes:02d}:{seconds:06.3f}"

def format_srt(segments: List[WhisperSegment]) -> str:
    """
    Format segments as SRT subtitle format

    Args:
        segments: List of transcription segments

    Returns:
        SRT formatted string
    """
    srt_content = ""
    for i, segment in enumerate(segments):
        segment_id = i + 1
        start = _format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')
        end = _format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')
        text = segment.text.strip().replace('-->', '->')

        # Format for SRT (with speaker if available)
        speaker_prefix = f"[{segment.speaker}] " if hasattr(segment, "speaker") and segment.speaker else ""
        srt_content += f"{segment_id}\n{start} --> {end}\n{speaker_prefix}{text}\n\n"

    return srt_content.strip()

def format_vtt(segments: List[WhisperSegment]) -> str:
    """
    Format segments as WebVTT subtitle format

    Args:
        segments: List of transcription segments

    Returns:
        WebVTT formatted string
    """
    vtt_content = "WEBVTT\n\n"
    for i, segment in enumerate(segments):
        start = _format_timestamp(segment.start, always_include_hours=True)
        end = _format_timestamp(segment.end, always_include_hours=True)
        text = segment.text.strip()

        # Format for VTT (with speaker if available)
        speaker_prefix = f"<v {segment.speaker}>" if hasattr(segment, "speaker") and segment.speaker else ""
        vtt_content += f"{start} --> {end}\n{speaker_prefix}{text}\n\n"

    return vtt_content.strip()

def transcribe_audio_chunk(model, audio_path: str, language: Optional[str] = None,
                          word_timestamps: bool = False,
                          force_fp32: bool = False,
                          use_cuda_graph_decoder: Optional[bool] = None) -> Tuple[str, List[WhisperSegment]]:
    """
    Transcribe a single audio chunk using the Parakeet-TDT model

    Args:
        model: The loaded ASR model
        audio_path: Path to the audio file
        language: Optional language code
        word_timestamps: Whether to generate word-level timestamps
        force_fp32: Disable autocast inside model.transcribe() to force FP32
        use_cuda_graph_decoder: When set, toggle the NeMo greedy CUDA graph
            decoder for this call (idempotent — only rebuilds when value changed)

    Returns:
        Tuple of (transcription text, list of WhisperSegment objects)
    """
    try:
        if use_cuda_graph_decoder is not None:
            _apply_decoder_config(model, use_cuda_graph_decoder)

        if force_fp32 and torch.cuda.is_available():
            autocast_ctx = torch.amp.autocast("cuda", enabled=False)
        else:
            autocast_ctx = nullcontext()

        # Use the NeMo model to transcribe audio
        with torch.no_grad(), autocast_ctx:
            # Simply pass the audio path(s) as a list to the transcribe method
            transcription = model.transcribe(
                [audio_path],
                timestamps=True  # Always request timestamps for segmentation
            )

        # Extract the text from the result
        if not transcription or len(transcription) == 0:
            logger.warning(f"No transcription generated for {audio_path}")
            return "", []

        result = transcription[0]  # Get the first result
        text = result.text

        # Create segments from the timestamp information if available
        segments = []

        # Check if we have timestamp information
        if hasattr(result, 'timestamp') and 'segment' in result.timestamp:
            for i, stamp in enumerate(result.timestamp['segment']):
                segments.append(WhisperSegment(
                    id=i,
                    start=stamp['start'],
                    end=stamp['end'],
                    text=stamp['segment']
                ))
        else:
            # If no segments available, create a single segment for the entire chunk
            segments.append(WhisperSegment(
                id=0,
                start=0.0,
                end=len(text.split()) / 2.0,  # Rough estimate based on word count
                text=text
            ))

        return text, segments

    except Exception as e:
        logger.error(f"Error transcribing audio chunk: {str(e)}")
        return "", []
