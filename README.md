# Parakeet Whisper-Compatible API

A simple FastAPI server that provides an OpenAI Whisper API-compatible endpoint backed by [NVIDIA's Parakeet-TDT model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) for speech recognition + [Pyannote](https://github.com/pyannote/pyannote-audio) for speaker diarization.

## Features

- Complete drop-in replacement for OpenAI's Whisper API
- Uses [NVIDIA's Parakeet-TDT 0.6B v3 model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) for high-quality transcription
- Supports all Whisper API response formats (json, text, srt, vtt, verbose_json)
- Supports word-level and segment-level timestamps
- Optional speaker diarization using [Pyannote.audio](https://github.com/pyannote/pyannote-audio)
- Automatic encoder attention switching for long audio (local attention reduces O(n²) → O(n))
- BF16 inference support — recommended on RTX 4070 / 5070 (Ada/Blackwell)
- VAD silence filtering — skips silent chunks without breaking timestamps
- FastAPI-based server with automatic OpenAPI documentation

## Requirements

- NVIDIA GPU with CUDA support (recommended)
- Python 3.8 or higher
- HuggingFace account and access token (required for speaker diarization)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/jfgonsalves/parakeet-diarized
   cd parakeet-diarized
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up speaker diarization (optional):
   - Create a free account at [HuggingFace](https://huggingface.co/)
   - Generate an access token at [HuggingFace Settings](https://huggingface.co/settings/tokens)
   - Accept the user agreement for the [Pyannote speaker diarization model](https://huggingface.co/pyannote/speaker-diarization-3.1)

5. Run the server:

   **With speaker diarization:**
   ```bash
   ./run.sh --hf-token "your_token_here"
   ```

   **Without speaker diarization:**
   ```bash
   ./run.sh
   ```

   **Other options:**
   ```bash
   ./run.sh --help  # See all available options
   ./run.sh --port 8080 --debug --hf-token "your_token_here"
   ```

## Usage

### API Endpoints

The API mimics the OpenAI Whisper API interface:

#### Transcribe Audio

```
POST /v1/audio/transcriptions
```

Parameters:
- `file`: The audio file to transcribe (multipart/form-data)
- `model`: Model to use (defaults to "whisper-1", but will use Parakeet regardless)
- `language`: Language of the audio (optional)
- `response_format`: Format of the response (defaults to "json", options: json, text, srt, vtt, verbose_json)
- `timestamps`: Whether to include timestamps (defaults to false)
- `timestamp_granularities`: Timestamp detail level (accepts "segment")
- `temperature`: Temperature for sampling (defaults to 0.0)
- `vad_filter`: Skip silent audio chunks before transcription (defaults to false). Uses RMS energy detection — does not modify audio, so timestamps remain correct.
- `prompt`: Optional prompt to guide the transcription (ignored but accepted for compatibility)
- `diarize`: Enable speaker diarization (defaults to true, requires HuggingFace token)
- `include_diarization_in_text`: Include speaker labels in transcript text (defaults to true)
- `chunk_duration`: Override audio chunk length in seconds for this request (falls back to `CHUNK_DURATION` env var, default 500)
- `use_cuda_graph_decoder`: Toggle the NeMo RNNT/TDT greedy CUDA graph decoder (falls back to `USE_CUDA_GRAPH_DECODER` env var, default false). Disable if you hit `illegal memory access` crashes during transcription on bleeding-edge GPU stacks.
- `force_fp32`: Disable autocast inside `model.transcribe()` to force FP32 (falls back to `FORCE_FP32` env var, default false)
- `use_bf16`: Use BF16 autocast inside `model.transcribe()` (falls back to `USE_BF16` env var, default false). Recommended on RTX 4070/5070 — better numerical stability than FP16 at equal throughput. Mutually exclusive with `force_fp32`.
- `cudnn_benchmark`: Flip `torch.backends.cudnn.benchmark` (falls back to `CUDNN_BENCHMARK` env var, default false). Sticky — affects subsequent requests too.
- `auto_attention`: Automatically switch encoder attention based on audio length (falls back to `AUTO_ATTENTION` env var, default false). When enabled, uses local attention (`rel_pos_local_attn`, ±20 s context) for audio longer than `LOCAL_ATTENTION_THRESHOLD` seconds, reducing computation from O(n²) to O(n). Reverts to global attention for short files.

Example with curl:
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file=@/path/to/your/audio.wav \
  -F model=whisper-1 \
  -F timestamps=true \
  -F diarize=true
```

#### Health Check

```
GET /health
```

Returns the health status of the API and the loaded model.

## Compatibility with OpenAI Whisper API

This API is designed to be a drop-in replacement for the OpenAI Whisper API:

1. Supports all Whisper API response formats (json, text, srt, vtt, verbose_json)
2. Accepts all major Whisper API parameters for compatibility
3. Returns responses in the same format as the OpenAI Whisper API
4. Provides a `/v1/models` endpoint for application compatibility

Minor differences:
1. The `model` parameter is accepted but ignored - always uses Parakeet-TDT
2. Some advanced Whisper-specific parameters might have no effect
3. Performance characteristics may differ from OpenAI's implementation

## API Response Formats

The API supports multiple response formats:

### JSON (default)
```json
{
  "text": "Full transcription text goes here"
}
```

### Verbose JSON
```json
{
  "text": "Full transcription text goes here",
  "task": "transcribe",
  "language": "en",
  "duration": 10.5,
  "model": "parakeet-tdt-0.6b-v3",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Segment text",
      "tokens": [50364, 2425, 286, 257],
      "temperature": 0.0,
      "avg_logprob": -0.5,
      "compression_ratio": 1.0,
      "no_speech_prob": 0.1
    },
    {
      "id": 1,
      "start": 2.5,
      "end": 5.0,
      "text": "Another segment",
      "tokens": [50364, 5816, 2121],
      "temperature": 0.0,
      "avg_logprob": -0.6,
      "compression_ratio": 1.0,
      "no_speech_prob": 0.05
    }
  ]
}
```

### Plain Text
```
Full transcription text goes here
```

### SRT
```
1
00:00:00,000 --> 00:00:02,500
Segment text

2
00:00:02,500 --> 00:00:05,000
Another segment
```

### VTT
```
WEBVTT

00:00:00.000 --> 00:00:02.500
Segment text

00:00:02.500 --> 00:00:05.000
Another segment
```

The `segments` field is included when the `timestamps` parameter is set to `true` or when using `verbose_json` format.

## Speaker Diarization

The API includes speaker diarization capabilities using [Pyannote.audio](https://github.com/pyannote/pyannote-audio):

### Setup Requirements

For speaker diarization to work, you need:

1. **HuggingFace Account**: Create a free account at [huggingface.co](https://huggingface.co/)
2. **Access Token**: Generate a token at [HuggingFace Settings](https://huggingface.co/settings/tokens)
3. **Model Agreement**: Accept the user agreement for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
4. **Environment Variable**: Set `HUGGINGFACE_ACCESS_TOKEN` with your token

### Features

- Automatic speaker detection and labeling
- Integration with transcription segments
- Optional speaker labels in transcript text
- Support for multiple speakers per audio file

### Usage

Enable diarization by setting `diarize=true` in your API request:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file=@/path/to/your/audio.wav \
  -F diarize=true \
  -F include_diarization_in_text=true
```

When `include_diarization_in_text=true`, the transcript will include speaker labels:
```
Speaker 1: Hello, how are you today?
Speaker 2: I'm doing well, thank you for asking.
```

### Configuration

Use the `run.sh` script to configure and start the server:

```bash
./run.sh --help
# Options:
#   --debug             Enable debug mode
#   --port PORT         Set server port (default: 8000)
#   --host HOST         Set server host (default: 0.0.0.0)
#   --skip-deps-check   Skip dependency checking
#   --hf-token TOKEN    Set HuggingFace access token for speaker diarization
#   --help              Show help message
```

**Environment Variables** (for settings not available as command line arguments):

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_ID` | NeMo/HuggingFace model ID | `nvidia/parakeet-tdt-0.6b-v3` |
| `HUGGINGFACE_ACCESS_TOKEN` | HF token (required for diarization) | — |
| `ENABLE_DIARIZATION` | Enable diarization globally | `true` |
| `INCLUDE_DIARIZATION_IN_TEXT` | Include speaker labels in transcript text | `true` |
| `CHUNK_DURATION` | Audio chunk length in seconds | `500` |
| `PORT` | Server port | `8000` |
| `TEMP_DIR` | Temporary directory for audio files | `/tmp/parakeet` |
| `USE_CUDA_GRAPH_DECODER` | NeMo RNNT/TDT greedy CUDA graph decoder — unstable on some stacks | `false` |
| `CUDNN_BENCHMARK` | `torch.backends.cudnn.benchmark` | `false` |
| `FORCE_FP32` | Disable autocast in `model.transcribe()` | `false` |
| `USE_BF16` | BF16 autocast in `model.transcribe()` — recommended on RTX 4070/5070 | `false` |
| `AUTO_ATTENTION` | Auto-switch encoder attention based on audio length | `false` |
| `LOCAL_ATTENTION_THRESHOLD` | Duration threshold (s) for switching to local attention | `60` |

## Performance

The [NVIDIA Parakeet-TDT model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) offers:
- Fast transcription (top model on the HF Open ASR leaderboard)
- Support for punctuation and capitalization
- High accuracy with word error rates as low as 1.69% on LibriSpeech test-clean

[Pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker diarization adds:
- Automatic speaker identification using state-of-the-art models
- Real-time speaker change detection
- Support for unlimited number of speakers

## Acknowledgments

This project builds upon excellent work by:

- **NVIDIA NeMo Team**: For the outstanding [Parakeet-TDT model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) that provides state-of-the-art speech recognition
- **Pyannote Team**: For the powerful [Pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker diarization toolkit

## License

This project is released under MIT License. However, the Parakeet-TDT model is governed by the CC-BY-4.0 license.
