import gc
import os
import logging
import re
from typing import Annotated, List, Literal, Optional
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from models import WhisperSegment, TranscriptionResponse, ModelInfo, ModelList
from audio import convert_audio_to_wav, split_audio_into_chunks
from transcription import load_model, format_srt, format_vtt, transcribe_audio_chunk
from diarization import Diarizer
from config import get_config

logger = logging.getLogger(__name__)

# Singletons — loaded once on startup, released on shutdown
asr_model = None
diarizer = None

config = get_config()

# Audio file extensions accepted by ffmpeg / the conversion pipeline
_ALLOWED_AUDIO_EXTENSIONS = frozenset({
    ".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac",
    ".webm", ".aac", ".opus", ".wma", ".aiff", ".aif",
})

# language codes: ISO 639-1 (2 chars), ISO 639-2 (3 chars), or full name
_LANGUAGE_RE = re.compile(r"^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{2,8})?$|^[a-zA-Z]{4,64}$")


def _validate_request_file(file: UploadFile) -> None:
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")
    ext = Path(filename).suffix.lower()
    if not ext:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot determine file type from filename {filename!r}. "
                   f"Supported extensions: {', '.join(sorted(_ALLOWED_AUDIO_EXTENSIONS))}",
        )
    if ext not in _ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {ext!r}. "
                   f"Supported: {', '.join(sorted(_ALLOWED_AUDIO_EXTENSIONS))}",
        )


def _validate_request_overrides(
    chunk_duration: Optional[int],
    language: Optional[str],
) -> None:
    errors: list[str] = []

    if chunk_duration is not None and not (1 <= chunk_duration <= 7200):
        errors.append(f"chunk_duration must be 1–7200 seconds (got {chunk_duration})")

    if language is not None:
        lang = language.strip()
        if not lang:
            errors.append("language must not be blank")
        elif not _LANGUAGE_RE.match(lang):
            errors.append(
                f"language must be a valid language code (e.g. 'en', 'pl', 'english'), got {language!r}"
            )

    if errors:
        raise HTTPException(
            status_code=400,
            detail="Invalid request parameters: " + "; ".join(errors),
        )


def create_app() -> FastAPI:
    app = FastAPI(title="Parakeet Whisper-Compatible API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        global asr_model, diarizer

        # Clean up stale temp files left by a previous crash (SIGKILL skips shutdown_event)
        temp_dir = Path(config.temp_dir)
        if temp_dir.exists():
            for stale in temp_dir.iterdir():
                try:
                    stale.unlink()
                except Exception:
                    pass
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("CUDA not available, using CPU (this will be slow)")

            torch.backends.cudnn.benchmark = config.cudnn_benchmark
            logger.info(f"cuDNN benchmark: {config.cudnn_benchmark}")

            asr_model = load_model(
                config.model_id,
                use_cuda_graph_decoder=config.use_cuda_graph_decoder,
            )
            logger.info(f"Model {config.model_id} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")

        try:
            hf_token = config.get_hf_token()
            if hf_token:
                diarizer = Diarizer(access_token=hf_token)
                logger.info("Diarization pipeline loaded")
            else:
                logger.info("No HuggingFace token — diarization disabled")
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")

    @app.on_event("shutdown")
    async def shutdown_event():
        global asr_model, diarizer

        if diarizer is not None:
            try:
                del diarizer.pipeline
            except Exception:
                pass
            diarizer = None

        if asr_model is not None:
            del asr_model
            asr_model = None

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        gc.collect()
        logger.info("Shutdown complete — model and diarizer released")

    @app.post("/v1/audio/transcriptions")
    async def transcribe_audio(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Literal["json", "text", "srt", "vtt", "verbose_json"] = Form("json"),
        temperature: Annotated[float, Form(ge=0.0, le=1.0)] = 0.0,
        timestamps: bool = Form(False),
        timestamp_granularities: Optional[List[str]] = Form(None),
        vad_filter: bool = Form(False),
        word_timestamps: bool = Form(False),
        diarize: bool = Form(True),
        include_diarization_in_text: Optional[bool] = Form(None),
        chunk_duration: Optional[int] = Form(None),
        use_cuda_graph_decoder: Optional[bool] = Form(None),
        force_fp32: Optional[bool] = Form(None),
        cudnn_benchmark: Optional[bool] = Form(None),
    ):
        """Transcribe audio — compatible with the OpenAI Whisper API"""
        global asr_model, diarizer

        if asr_model is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again in a few moments.")

        # ── Input validation ──────────────────────────────────────────────────
        _validate_request_file(file)
        _validate_request_overrides(chunk_duration, language)

        logger.info(f"Transcription requested: {file.filename}, format: {response_format}")

        # Resolve per-request flags (None → fall back to config)
        effective_chunk_duration = chunk_duration if chunk_duration is not None else config.chunk_duration
        effective_use_cuda_graph_decoder = (
            use_cuda_graph_decoder if use_cuda_graph_decoder is not None
            else config.use_cuda_graph_decoder
        )
        effective_force_fp32 = force_fp32 if force_fp32 is not None else config.force_fp32
        if cudnn_benchmark is not None:
            # Sticky — affects subsequent requests too
            torch.backends.cudnn.benchmark = cudnn_benchmark

        temp_file = None
        wav_file = None
        audio_chunks = []

        try:
            temp_dir = Path(config.temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)

            temp_file = temp_dir / f"upload_{os.urandom(8).hex()}{Path(file.filename).suffix}"
            with open(temp_file, "wb") as f:
                f.write(await file.read())

            wav_file = convert_audio_to_wav(str(temp_file))
            audio_chunks = split_audio_into_chunks(wav_file, chunk_duration=effective_chunk_duration)

            # Diarization (uses the global singleton loaded at startup)
            active_diarizer = diarizer if diarize else None
            diarization_result = None
            if active_diarizer:
                logger.info("Performing speaker diarization")
                diarization_result = active_diarizer.diarize(wav_file)
                logger.info(f"Diarization found {diarization_result.num_speakers} speakers")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            elif diarize and diarizer is None:
                logger.warning("Diarization requested but pipeline not available (no HF token or load failed)")

            # Transcribe all chunks
            all_text = []
            all_segments = []
            for i, chunk_path in enumerate(audio_chunks):
                logger.info(f"Processing chunk {i+1}/{len(audio_chunks)}")
                chunk_text, chunk_segments = transcribe_audio_chunk(
                    asr_model,
                    chunk_path,
                    language=language,
                    word_timestamps=word_timestamps,
                    force_fp32=effective_force_fp32,
                    use_cuda_graph_decoder=effective_use_cuda_graph_decoder,
                )
                if i > 0:
                    offset = i * effective_chunk_duration
                    for seg in chunk_segments:
                        seg.start += offset
                        seg.end += offset
                all_text.append(chunk_text)
                all_segments.extend(chunk_segments)

            full_text = " ".join(all_text)

            # Merge diarization into segments
            if active_diarizer and diarization_result and diarization_result.segments:
                all_segments = active_diarizer.merge_with_transcription(diarization_result, all_segments)

                use_diarization_in_text = (
                    include_diarization_in_text
                    if include_diarization_in_text is not None
                    else config.include_diarization_in_text
                )

                if use_diarization_in_text:
                    previous_speaker = None
                    seen_speakers = set()
                    for segment in all_segments:
                        if not (hasattr(segment, 'speaker') and segment.speaker):
                            continue
                        label = segment.speaker
                        if not label.startswith("SPEAKER_"):
                            if label != previous_speaker:
                                segment.text = f"Speaker: {segment.text}"
                            previous_speaker = label
                            continue
                        try:
                            num = int(label.split("_")[-1]) + 1
                            if label != previous_speaker:
                                prefix = f"Speaker {num}: " if label not in seen_speakers else f"{num}: "
                                seen_speakers.add(label)
                                segment.text = f"{prefix}{segment.text}"
                            previous_speaker = label
                        except (ValueError, IndexError):
                            pass

                    full_text = " ".join(seg.text for seg in all_segments)
                    logger.info(f"Speaker diarization applied to {len(all_segments)} segments and included in text")
                else:
                    logger.info("Speaker diarization applied to segments but not included in text")
            else:
                logger.warning("Diarization not applied or returned no speakers")

            response = TranscriptionResponse(
                text=full_text,
                segments=all_segments if timestamps or response_format == "verbose_json" else None,
                language=language,
                duration=sum(len(s.text.split()) for s in all_segments) / 150 if all_segments else 0,
                model=config.model_id,
            )

            if response_format == "json":
                return response.dict()
            elif response_format == "text":
                return PlainTextResponse(full_text)
            elif response_format == "srt":
                return PlainTextResponse(format_srt(all_segments))
            elif response_format == "vtt":
                return PlainTextResponse(format_vtt(all_segments))
            else:  # verbose_json
                return response.dict()

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Always clean up temp files — runs even when an exception is raised
            for path in [temp_file] + ([wav_file] if wav_file != str(temp_file) else []) + audio_chunks:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception:
                        pass

    @app.get("/health")
    async def health_check():
        global asr_model
        ready = asr_model is not None
        payload = {
            "status": "ok" if ready else "starting",
            "version": "1.0.0",
            "model_loaded": ready,
            "model_id": config.model_id,
            "cuda_available": torch.cuda.is_available(),
            "gpu_info": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "config": config.as_dict(),
        }
        if not ready:
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.get("/v1/models")
    async def list_models():
        models = [
            ModelInfo(
                id="whisper-1",
                created=1677649963,
                owned_by="parakeet",
                root="whisper-1",
                permission=[{
                    "id": "modelperm-1", "object": "model_permission",
                    "created": 1677649963, "allow_create_engine": False,
                    "allow_sampling": True, "allow_logprobs": True,
                    "allow_search_indices": False, "allow_view": True,
                    "allow_fine_tuning": False, "organization": "*",
                    "group": None, "is_blocking": False,
                }]
            )
        ]
        return ModelList(data=models)

    return app
