import os
import warnings
import logging
import uvicorn
import torch

# NeMo's DataLoader (even with num_workers=0) may leave one semaphore at Python
# exit — harmless in a container/systemd environment where the OS reclaims all
# IPC resources when the process dies.  Suppress the noisy resource_tracker
# warning so it doesn't pollute Docker/journald logs.
warnings.filterwarnings(
    "ignore",
    message="resource_tracker: There appear to be",
    category=UserWarning,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Enable debug logging if requested
if os.environ.get('DEBUG', '0') == '1':
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled")

# Import from modularized components
from api import create_app
from config import get_config

# Get the configuration
config = get_config()

# Create the FastAPI application
app = create_app()

# For backwards compatibility - re-export the split_audio_into_chunks function
# This is needed for the test_chunking.py to work without modification
from audio import split_audio_into_chunks, convert_audio_to_wav
from transcription import load_model, format_srt, format_vtt
from models import WhisperSegment, TranscriptionResponse

# Run the server if executed directly
if __name__ == "__main__":
    # Log startup information
    logger.info(f"Starting Parakeet Whisper-Compatible API on {config.host}:{config.port}")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available, using CPU (this will be slow)")
    
    # Start the server
    uvicorn.run(app, host=config.host, port=config.port, reload=config.debug)