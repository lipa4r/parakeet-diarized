# CLAUDE.md — Parakeet Diarized

## O projekcie

FastAPI serwer transkrypcji audio zgodny z API OpenAI Whisper (`/v1/audio/transcriptions`).

- **Transkrypcja**: NVIDIA Parakeet-TDT 0.6B v3 via NeMo toolkit
- **Diaryzacja mówców**: pyannote.audio 3.1 (wymaga tokenu HuggingFace)
- **Formaty odpowiedzi**: `json`, `text`, `srt`, `vtt`, `verbose_json`

## Uruchomienie

```bash
# Wymagania: Python 3.10–3.11, ffmpeg, token HuggingFace (dla diaryzacji)
./run.sh --hf-token <TOKEN>

# Opcje:
./run.sh --port 8080 --debug --hf-token <TOKEN>
```

### Zmienne środowiskowe

| Zmienna | Opis | Domyślna |
|---------|------|----------|
| `MODEL_ID` | ID modelu NeMo/HuggingFace | `nvidia/parakeet-tdt-0.6b-v3` |
| `HUGGINGFACE_ACCESS_TOKEN` | Token HF (wymagany do diaryzacji) | — |
| `ENABLE_DIARIZATION` | Włącz diaryzację | `true` |
| `INCLUDE_DIARIZATION_IN_TEXT` | Etykiety mówców w tekście | `true` |
| `CHUNK_DURATION` | Długość fragmentu audio (s) | `500` |
| `PORT` | Port serwera | `8000` |
| `TEMP_DIR` | Katalog na pliki tymczasowe | `/tmp/parakeet` |
| `USE_CUDA_GRAPH_DECODER` | Włącz CUDA graph decoder w NeMo (RNNT/TDT greedy) | `false` |
| `CUDNN_BENCHMARK` | `torch.backends.cudnn.benchmark` | `false` |
| `FORCE_FP32` | Wyłącz autocast w `model.transcribe()` (FP32) | `false` |
| `BEAM_SIZE` | Szerokość wiązki dekodowania: 1=greedy (szybko), >1=beam search (lepszy WER, wolniej) | `1` |

### Parametry endpointu (Form, override per request)

Wszystkie opcjonalne — `None` oznacza użycie wartości z configu.

| Parametr | Opis |
|----------|------|
| `chunk_duration` | Długość fragmentu audio (s) na ten request |
| `use_cuda_graph_decoder` | Toggle CUDA graph decodera (idempotent, `change_decoding_strategy`) |
| `force_fp32` | FP32 dla tej transkrypcji (`autocast(enabled=False)`) |
| `cudnn_benchmark` | Globalny flag torch — uwaga: sticky, wpływa na kolejne requesty |
| `beam_size` | Szerokość wiązki: 1=greedy, >1=beam search (wyłącza CUDA graph decoder automatycznie) |

## Kluczowe pliki

```
main.py            — entry point, uruchamia serwer uvicorn
api.py             — endpointy FastAPI (/v1/audio/transcriptions, /health, /v1/models)
transcription.py   — ładowanie modelu NeMo i transkrypcja audio
diarization/       — moduł diaryzacji mówców (pyannote.audio)
audio.py           — konwersja i podział audio (ffmpeg)
config.py          — konfiguracja z env vars + .env
models.py          — modele Pydantic (WhisperSegment, TranscriptionResponse)
```

## Pipeline transkrypcji

```
Plik audio
  → convert_audio_to_wav()         # ffmpeg → 16kHz mono WAV
  → split_audio_into_chunks()      # podział na fragmenty (domyślnie 500s)
  ├─ diarizer.diarize()            # pyannote na pełnym pliku
  └─ transcribe_audio_chunk()      # NeMo na każdym fragmencie
  → diarizer.merge_with_transcription()  # połączenie po nakładaniu się czasu
  → format odpowiedzi (json/srt/vtt/...)
```

## Instalacja zależności

```bash
pip install -U nemo_toolkit[asr]
pip install -r requirements.txt
```

> **Uwaga**: NeMo wymaga Pythona 3.10–3.11. Na Pythonie 3.13+ mogą wystąpić problemy z `libstdc++` (szczególnie przy conda). W razie problemów użyj środowiska z Python 3.11.

## Testy

```bash
python -m pytest tests/ -v
```

- `tests/test_api.py` — testy endpointów API (mockowane zależności)
- `tests/test_chunking.py` — testy logiki podziału audio

Testy nie wymagają GPU ani rzeczywistego modelu.

## Przykłady użycia

```bash
# Transkrypcja z diaryzacją (domyślnie)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-1

# Format SRT z etykietami mówców
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F model=whisper-1 \
  -F response_format=srt

# Bez diaryzacji
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-1 \
  -F diarize=false
```

## Model: Parakeet-TDT 0.6B v3

- Architektura: FastConformer (Token-and-Duration Transducer)
- Parametry: ~600M
- Języki: 25 języków europejskich z auto-detekcją
- WER na LibriSpeech test-clean: ~1.93%
- Ładowany przez: `nemo_asr.models.ASRModel.from_pretrained()` (auto-wykrywa typ)
