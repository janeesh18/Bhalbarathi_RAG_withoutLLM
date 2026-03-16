# Vidya — Offline Multilingual Chatbot

An offline voice/text chatbot for students that answers questions from a knowledge base — no LLM, no internet required. Designed to run on both Windows and Raspberry Pi.

---

## Features

- Supports **English, Hindi, and Marathi** (auto-detects language)
- **Voice input** with automatic silence detection
- **Text input** mode
- **Hybrid search** — semantic + keyword fusion for accurate answers
- **No LLM** — fully offline, works on low-power devices
- Text-to-speech output in all three languages

---

## Architecture

```
User speaks/types
      ↓
Speech-to-Text (IndicWhisper / Whisper large-v3-turbo)
      ↓
Hybrid Search (ChromaDB semantic + keyword fusion)
      ↓
Text-to-Speech (pyttsx3 / Meta MMS-TTS)
      ↓
Spoken + printed answer
```

---

## Requirements

### Python
Python 3.9 or higher

### Install dependencies

```bash
pip install faster-whisper chromadb sentence-transformers transformers torch sounddevice soundfile pyttsx3 huggingface-hub numpy
```

---

## Models (Auto-downloaded on first run)

| Model | Language | Size | Purpose |
|-------|----------|------|---------|
| `indicwhisper-hi-ct2` | Hindi / Marathi | ~500 MB | Speech-to-Text |
| `Systran/faster-whisper-large-v3-turbo` | English | ~800 MB | Speech-to-Text |
| `facebook/mms-tts-hin` | Hindi | ~40 MB | Text-to-Speech |
| `facebook/mms-tts-mar` | Marathi | ~40 MB | Text-to-Speech |

> IndicWhisper must be manually placed at `indicwhisper-hi-ct2/` in the project directory (requires `model.bin`).

---

## Knowledge Base

Place dataset files in a `datasets/` folder:

```
datasets/
  balbharati_class6_rechunked.json   ← Class 6 Balbharati textbook content
  vidya_knowledge_base.json          ← Maharashtra general knowledge (optional)
```

The chatbot works with either or both files. Missing files are skipped with a warning.

---

## Project Structure

```
rag_engine.py                        ← Main script
datasets/
  balbharati_class6_rechunked.json
  vidya_knowledge_base.json
indicwhisper-hi-ct2/                 ← IndicWhisper model (manual download)
chatbot_knowledge/                   ← ChromaDB vector store (auto-generated)
mms-tts-hin/                         ← Hindi TTS model (auto-downloaded)
mms-tts-mar/                         ← Marathi TTS model (auto-downloaded)
whisper-large-v3-turbo/              ← English STT model (auto-downloaded)
```

---

## Usage

```bash
python rag_engine.py
```

On startup:
1. Select language: `1` Marathi, `2` Hindi, `3` English, `4` Auto-detect
2. Select mode: `1` Text, `2` Voice

### Text mode commands
| Input | Action |
|-------|--------|
| Any question | Get answer |
| `voice` | Switch to voice mode |
| `quit` | Exit |

### Voice mode commands
| Input | Action |
|-------|--------|
| Press ENTER | Start recording |
| Silence (~1.2s) | Auto-stop recording |
| Type `text` | Switch to text mode |
| Type `quit` | Exit |

---

## How Search Works

1. **Semantic search** — query is converted to a vector using `paraphrase-multilingual-MiniLM-L12-v2` and matched against indexed chunks in ChromaDB
2. **Keyword search** — keywords are extracted from the query (stopwords removed), synonyms expanded, and matched against chunk content/topic/tags
3. **Score fusion** — final score = `semantic × 0.6 + keyword × 0.4`
4. Language match bonus of `+0.15` if chunk language matches query language
5. Returns `NO_ANSWER` message if best score < 0.20

---

## Platform Support

| Platform | STT | TTS | Notes |
|----------|-----|-----|-------|
| Windows | IndicWhisper + Whisper | pyttsx3 (en) + MMS-TTS (hi/mr) | Full support |
| Raspberry Pi | IndicWhisper + Whisper | espeak (en) + MMS-TTS (hi/mr) | Uses `aplay` for audio playback |

---

## First Run Notes

- ChromaDB index is built automatically on first run and cached in `chatbot_knowledge/`
- Subsequent runs load from cache (faster startup)
- MMS-TTS and Whisper models are downloaded once and stored locally

---

## License

For educational use.
