# Configuration settings for Parakeet
import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("dotenv package not installed. Environment variables will only be loaded from system.")

# API settings
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEBUG_MODE = os.environ.get("DEBUG", "0") == "1"

# Model settings
DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_CHUNK_DURATION = 500

# Hugging Face configuration
HF_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

# Diarization settings
DEFAULT_DIARIZE = True
DEFAULT_NUM_SPEAKERS = None
DEFAULT_INCLUDE_DIARIZATION_IN_TEXT = True

# GPU/inference stability flags
DEFAULT_USE_CUDA_GRAPH_DECODER = False
DEFAULT_CUDNN_BENCHMARK = False
DEFAULT_FORCE_FP32 = False


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw!r}")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be a number, got: {raw!r}")


class Config:
    """Global configuration for Parakeet"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # API settings
        self.host = os.environ.get("HOST", DEFAULT_HOST)
        self.port = _env_int("PORT", DEFAULT_PORT)
        self.debug = DEBUG_MODE

        # Model settings
        self.model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
        self.temperature = _env_float("TEMPERATURE", DEFAULT_TEMPERATURE)
        self.chunk_duration = _env_int("CHUNK_DURATION", DEFAULT_CHUNK_DURATION)

        # Diarization settings
        self.hf_token = HF_TOKEN
        self.enable_diarization = os.environ.get("ENABLE_DIARIZATION", str(DEFAULT_DIARIZE)).lower() == "true"
        self.include_diarization_in_text = os.environ.get(
            "INCLUDE_DIARIZATION_IN_TEXT", str(DEFAULT_INCLUDE_DIARIZATION_IN_TEXT)
        ).lower() == "true"
        self.default_num_speakers = DEFAULT_NUM_SPEAKERS

        # GPU/inference stability flags
        self.use_cuda_graph_decoder = os.environ.get(
            "USE_CUDA_GRAPH_DECODER", str(DEFAULT_USE_CUDA_GRAPH_DECODER)
        ).lower() == "true"
        self.cudnn_benchmark = os.environ.get(
            "CUDNN_BENCHMARK", str(DEFAULT_CUDNN_BENCHMARK)
        ).lower() == "true"
        self.force_fp32 = os.environ.get(
            "FORCE_FP32", str(DEFAULT_FORCE_FP32)
        ).lower() == "true"

        # File paths
        self.temp_dir = os.environ.get("TEMP_DIR", "/tmp/parakeet")

        self._validate()

        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized configuration: debug={self.debug}, model={self.model_id}")

    def _validate(self):
        errors: list[str] = []

        if not self.model_id or not self.model_id.strip():
            errors.append("MODEL_ID must not be empty")

        if not (1 <= self.port <= 65535):
            errors.append(f"PORT must be 1–65535 (got {self.port})")

        if not (0.0 <= self.temperature <= 1.0):
            errors.append(f"TEMPERATURE must be 0.0–1.0 (got {self.temperature})")

        if not (1 <= self.chunk_duration <= 7200):
            errors.append(f"CHUNK_DURATION must be 1–7200 seconds (got {self.chunk_duration})")

        if not self.temp_dir or not self.temp_dir.strip():
            errors.append("TEMP_DIR must not be empty")

        if errors:
            raise ValueError(
                "Invalid startup configuration:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def update_hf_token(self, token: str) -> None:
        self.hf_token = token
        logger.info("Updated HuggingFace token")

    def get_hf_token(self) -> Optional[str]:
        return self.hf_token

    def as_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "chunk_duration": self.chunk_duration,
            "enable_diarization": self.enable_diarization,
            "include_diarization_in_text": self.include_diarization_in_text,
            "has_hf_token": self.hf_token is not None,
            "use_cuda_graph_decoder": self.use_cuda_graph_decoder,
            "cudnn_benchmark": self.cudnn_benchmark,
            "force_fp32": self.force_fp32,
        }


# Create a global instance
config = Config()


def get_config() -> Config:
    return config
