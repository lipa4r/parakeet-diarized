#!/bin/bash
set -e

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
DEBUG=0
PORT=8000
HOST="0.0.0.0"
CHECK_DEPS=1
HF_TOKEN=""
CUDA_GRAPH_DECODER=0
CUDNN_BENCHMARK_FLAG=0
FORCE_FP32=0
CHUNK_DURATION_ARG=""
CPU_THREADS=""
WORKERS=1

# Process command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --debug)
            DEBUG=1
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --skip-deps-check)
            CHECK_DEPS=0
            shift
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --cuda-graph-decoder)
            CUDA_GRAPH_DECODER=1
            shift
            ;;
        --cudnn-benchmark)
            CUDNN_BENCHMARK_FLAG=1
            shift
            ;;
        --force-fp32)
            FORCE_FP32=1
            shift
            ;;
        --chunk-duration)
            CHUNK_DURATION_ARG="$2"
            shift 2
            ;;
        --cpu-threads)
            CPU_THREADS="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --help)
            echo -e "${BLUE}Parakeet Whisper-Compatible API Server${NC}"
            echo -e "Usage: $0 [options]"
            echo -e "Options:"
            echo -e "  --debug                    Enable debug mode"
            echo -e "  --port PORT                Set server port (default: 8000)"
            echo -e "  --host HOST                Set server host (default: 0.0.0.0)"
            echo -e "  --skip-deps-check          Skip dependency checking"
            echo -e "  --hf-token TOKEN           Set HuggingFace access token for speaker diarization"
            echo -e "  --cuda-graph-decoder       Enable NeMo CUDA graph decoder (default: off, unstable on sm_120)"
            echo -e "  --cudnn-benchmark          Enable torch.backends.cudnn.benchmark (default: off)"
            echo -e "  --force-fp32               Force FP32 inference, disable autocast (default: off)"
            echo -e "  --chunk-duration SEC       Audio chunk duration in seconds (default: 500)"
            echo -e "  --cpu-threads N            Limit CPU threads for PyTorch/OpenMP/MKL (OMP_NUM_THREADS)"
            echo -e "  --workers N                Number of uvicorn worker processes (default: 1)."
            echo -e "                             Each worker loads the model separately — N>1 uses N×VRAM."
            echo -e "                             Workers >1 disables --reload."
            echo -e "  --help                     Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $key${NC}" >&2
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Starting Parakeet Whisper-Compatible API Server${NC}"

# Check for ffmpeg
if [[ $CHECK_DEPS -eq 1 ]]; then
    echo -e "${BLUE}Checking dependencies...${NC}"
    if ! command -v ffmpeg &> /dev/null; then
        echo -e "${RED}ERROR: ffmpeg is required but not installed.${NC}"
        echo -e "${YELLOW}Please install ffmpeg using your package manager:${NC}"
        echo -e "${YELLOW}  - Ubuntu/Debian: sudo apt-get install ffmpeg${NC}"
        echo -e "${YELLOW}  - MacOS: brew install ffmpeg${NC}"
        exit 1
    else
        echo -e "${GREEN}ffmpeg found: $(ffmpeg -version | head -n 1)${NC}"
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv || { echo -e "${RED}Failed to create virtual environment. Make sure python3-venv is installed.${NC}"; exit 1; }
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate || { echo -e "${RED}Failed to activate virtual environment.${NC}"; exit 1; }

# Install requirements if needed
if [ ! -f "venv/.requirements_installed" ] && [[ $CHECK_DEPS -eq 1 ]]; then
    echo -e "${YELLOW}Installing requirements...${NC}"
    pip install -r requirements.txt || { echo -e "${RED}Failed to install requirements.${NC}"; exit 1; }
    touch venv/.requirements_installed
fi

# Check for CUDA
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${GREEN}CUDA is available. Using GPU.${NC}"
    CUDA_INFO=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo -e "${GREEN}GPU Device: ${CUDA_INFO}${NC}"
else
    echo -e "${YELLOW}WARNING: CUDA is not available. Using CPU, which will be much slower.${NC}"
    echo -e "${YELLOW}Consider installing CUDA for better performance.${NC}"
fi

# Set environment variables
if [[ $DEBUG -eq 1 ]]; then
    echo -e "${YELLOW}Debug mode enabled. Verbose output will be shown.${NC}"
    export DEBUG=1
fi

if [[ -n "$HF_TOKEN" ]]; then
    echo -e "${GREEN}HuggingFace access token set. Speaker diarization will be available.${NC}"
    export HUGGINGFACE_ACCESS_TOKEN="$HF_TOKEN"
fi

if [[ $CUDA_GRAPH_DECODER -eq 1 ]]; then
    echo -e "${YELLOW}CUDA graph decoder enabled.${NC}"
    export USE_CUDA_GRAPH_DECODER=true
fi

if [[ $CUDNN_BENCHMARK_FLAG -eq 1 ]]; then
    echo -e "${YELLOW}cuDNN benchmark enabled.${NC}"
    export CUDNN_BENCHMARK=true
fi

if [[ $FORCE_FP32 -eq 1 ]]; then
    echo -e "${YELLOW}Forcing FP32 inference (autocast disabled).${NC}"
    export FORCE_FP32=true
fi

if [[ -n "$CHUNK_DURATION_ARG" ]]; then
    echo -e "${BLUE}Audio chunk duration: ${CHUNK_DURATION_ARG}s${NC}"
    export CHUNK_DURATION="$CHUNK_DURATION_ARG"
fi

if [[ -n "$CPU_THREADS" ]]; then
    echo -e "${BLUE}CPU threads limited to: ${CPU_THREADS}${NC}"
    export OMP_NUM_THREADS="$CPU_THREADS"
    export MKL_NUM_THREADS="$CPU_THREADS"
fi

# Run the server
echo -e "${GREEN}Starting server on ${HOST}:${PORT}...${NC}"

if [[ $WORKERS -gt 1 ]]; then
    echo -e "${YELLOW}Running with ${WORKERS} workers (each loads model separately, uses ${WORKERS}x VRAM).${NC}"
    if [[ $DEBUG -eq 1 ]]; then
        uvicorn main:app --host $HOST --port $PORT --workers $WORKERS --log-level debug
    else
        uvicorn main:app --host $HOST --port $PORT --workers $WORKERS
    fi
else
    if [[ $DEBUG -eq 1 ]]; then
        uvicorn main:app --host $HOST --port $PORT --reload --log-level debug
    else
        uvicorn main:app --host $HOST --port $PORT --reload
    fi
fi
