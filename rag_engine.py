"""
Vidya v18 — Cross-Platform, No LLM
====================================
Architecture:
  Voice/Text -> STT -> Hybrid Search (multilingual semantic + keyword) -> TTS

STT  : IndicWhisper (hi/mr) + Whisper large-v3-turbo (en)
Search: ChromaDB (paraphrase-multilingual-MiniLM-L12-v2) + keyword fusion
TTS  : pyttsx3 (en) + Meta MMS-TTS (hi/mr, ~40 MB each, auto-downloaded)

Platform: Windows + Raspberry Pi (auto-detected)
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import time
import re
import json
import threading
import queue

IS_WINDOWS = sys.platform == "win32"

# =============================================================================
# PATHS & CONFIG
# =============================================================================

_BASE = os.path.dirname(os.path.abspath(__file__))

# MMS-TTS models (auto-downloaded on first run, ~40 MB each)
_MMS_HIN_REPO  = "facebook/mms-tts-hin"
_MMS_HIN_LOCAL = os.path.join(_BASE, "mms-tts-hin")
_MMS_MAR_REPO  = "facebook/mms-tts-mar"
_MMS_MAR_LOCAL = os.path.join(_BASE, "mms-tts-mar")

# Whisper large-v3-turbo for English (auto-downloaded, ~800 MB)
_WHISPER_EN_REPO  = "Systran/faster-whisper-large-v3-turbo"
_WHISPER_EN_LOCAL = os.path.join(_BASE, "whisper-large-v3-turbo")

if IS_WINDOWS:
    CONFIG = {
        "whisper_indic":        os.path.join(_BASE, "indicwhisper-hi-ct2"),
        "whisper_en":           _WHISPER_EN_LOCAL,
        "rag_db_path":          os.path.join(_BASE, "chatbot_knowledge"),
        "vidya_knowledge":      os.path.join(_BASE, "datasets", "vidya_knowledge_base.json"),
        "balbharati_knowledge": os.path.join(_BASE, "datasets", "balbharati_class6_rechunked.json"),
        "embedding_model":      "paraphrase-multilingual-MiniLM-L12-v2",
        "default_language":     "mr",
    }
else:
    CONFIG = {
        "whisper_indic":        "/home/pi/vidya/models/indicwhisper-hi-ct2",
        "whisper_en":           "/home/pi/vidya/models/whisper-large-v3-turbo",
        "rag_db_path":          "/home/pi/vidya/chatbot_knowledge",
        "vidya_knowledge":      "/home/pi/vidya/datasets/vidya_knowledge_base.json",
        "balbharati_knowledge": "/home/pi/vidya/datasets/balbharati_class6_rechunked.json",
        "embedding_model":      "paraphrase-multilingual-MiniLM-L12-v2",
        "default_language":     "mr",
    }

NO_ANSWER = {
    "mr": "मला याबद्दल माहिती उपलब्ध नाही. कृपया शिक्षकांना विचारा.",
    "hi": "मुझे इसकी जानकारी नहीं है। कृपया शिक्षक से पूछें।",
    "en": "I don't have this information. Please ask your teacher.",
}

# =============================================================================
# CANNED RESPONSES
# =============================================================================

CANNED = {
    "mr": {
        "greet": "नमस्कार! मी विद्या आहे. तुम्हाला कशाबद्दल माहिती हवी आहे?",
        "thanks": "धन्यवाद! आणखी काही प्रश्न असल्यास विचारा.",
        "bye":   "निरोप! पुन्हा भेटू.",
    },
    "hi": {
        "greet": "नमस्ते! मैं विद्या हूँ। आप क्या जानना चाहते हैं?",
        "thanks": "धन्यवाद! और कोई सवाल हो तो पूछें।",
        "bye":   "अलविदा! फिर मिलेंगे।",
    },
    "en": {
        "greet": "Hello! I'm Vidya. What would you like to know?",
        "thanks": "You're welcome! Feel free to ask more questions.",
        "bye":   "Goodbye! See you next time.",
    },
}

_GREET_RE  = re.compile(r'^\s*(hi|hello|hey|नमस्कार|नमस्ते|हॅलो|हेलो)\s*[!.]?\s*$', re.IGNORECASE)
_THANKS_RE = re.compile(r'^\s*(thanks|thank\s*you|धन्यवाद|शुक्रिया|आभारी|थँक्स)\b', re.IGNORECASE)
_BYE_RE    = re.compile(r'^\s*(bye|goodbye|निरोप|बाय)\s*[!.]?\s*$', re.IGNORECASE)


def _canned(text, language):
    lang = language if language in CANNED else "en"
    if _GREET_RE.match(text):  return CANNED[lang]["greet"]
    if _THANKS_RE.match(text): return CANNED[lang]["thanks"]
    if _BYE_RE.match(text):    return CANNED[lang]["bye"]
    return None


# =============================================================================
# LANGUAGE DETECTION
# =============================================================================

MARATHI_WORDS = {
    "आहे", "आहेत", "आहेस", "नाही", "नाहीत", "नाहीस",
    "होते", "होती", "होता", "होतो", "होतात",
    "झाले", "झाली", "झाला", "झालो",
    "करतो", "करते", "करतात", "करतोय",
    "केले", "केली", "केला", "केलो",
    "असतो", "असते", "असतात",
    "शकतो", "शकते", "शकतात",
    "लागतो", "लागते", "वाटते", "मिळतो", "मिळते",
    "बोलतो", "बोलते", "बोलतात",
    "येतो", "येते", "जातो", "जाते",
    "दिसतो", "दिसते", "बसतो", "बसते",
    "आला", "आली", "आले", "गेला", "गेली", "गेले",
    "दिला", "दिली", "दिले", "घेतला", "घेतली",
    "मला", "तुला", "त्याला", "तिला", "आम्हाला", "तुम्हाला", "त्यांना",
    "माझा", "माझी", "माझे", "तुझा", "तुझी", "तुझे",
    "आमचा", "आमची", "तुमचा", "तुमची",
    "मध्ये", "बद्दल", "विषयी", "साठी", "पासून", "पर्यंत", "बरोबर",
    "आणि", "पण", "म्हणून", "म्हणजे", "किंवा",
    "काय", "कसे", "कसा", "कशी", "कोण", "कुठे", "केव्हा", "किती",
    "कोणता", "कोणती", "कोणते", "कोणाला",
    "सांगा", "सांग", "द्या", "घ्या", "करा", "बघा", "बोला", "चला",
    "नमस्कार", "धन्यवाद", "कृपया",
    "मराठी", "महाराष्ट्राची", "महाराष्ट्राचे", "महाराष्ट्राला", "महाराष्ट्रातील",
    "फक्त", "खूप", "थोडे", "अधिक", "सगळे",
    "चांगले", "चांगला", "चांगली", "वाईट",
    "अभ्यास", "प्रश्न", "उत्तर", "पाठ", "धडा",
    "किल्ला", "किल्ले", "राज्य",
}

HINDI_WORDS = {
    "है", "हैं", "हूँ", "हूं", "था", "थी", "थे",
    "करता", "करती", "करते", "किया", "किये",
    "सकता", "सकती", "सकते",
    "रहा", "रही", "रहे",
    "गया", "गई", "गए", "गये",
    "चाहता", "चाहती",
    "मुझे", "तुम्हें", "उसे", "उन्हें", "हमें",
    "मेरा", "मेरी", "मेरे", "तुम्हारा", "तुम्हारी",
    "मैं", "मैंने", "हम", "हमने", "तुम", "तुमने",
    "क्या", "कैसे", "कैसा", "कैसी", "कौन", "कहाँ", "कहां", "कब",
    "लेकिन", "इसलिए", "क्योंकि",
    "बताओ", "बताइए", "बताएं", "बोलो", "दीजिए", "कीजिए",
    "नमस्ते", "शुक्रिया",
    "हिंदी", "बहुत", "अच्छा", "अच्छी",
    "ज्यादा", "ज़्यादा", "थोड़ा",
}


def detect_language(text):
    """Marathi default for ambiguous Devanagari."""
    if not text or not text.strip():
        return "en"
    text_clean = text.strip()
    devanagari = sum(1 for c in text_clean if '\u0900' <= c <= '\u097F')
    latin      = sum(1 for c in text_clean if 'a' <= c.lower() <= 'z')
    total = devanagari + latin
    if total == 0:       return "en"
    if devanagari / max(total, 1) < 0.3: return "en"

    hindi_evidence = marathi_evidence = 0
    words = set(text_clean.split())

    if "ता है" in text_clean or "ती है" in text_clean or "ते हैं" in text_clean:
        hindi_evidence += 10
    if "ता था" in text_clean or "ती थी" in text_clean or "ते थे" in text_clean:
        hindi_evidence += 10
    if "सकता है" in text_clean or "सकती है" in text_clean:
        hindi_evidence += 10

    strong_hindi = {
        "क्या", "कैसे", "कैसा", "मुझे", "हमें",
        "मैं", "मैंने", "हमने", "तुमने", "मेरा", "मेरी",
        "बताओ", "बताइए", "बोलो", "दीजिए",
        "लेकिन", "इसलिए", "क्योंकि",
        "नमस्ते", "शुक्रिया", "हिंदी", "बहुत", "अच्छा", "अच्छी", "थोड़ा",
    }
    hindi_evidence   += sum(3 for w in words if w in strong_hindi)
    marathi_evidence += sum(3 for w in words if w in MARATHI_WORDS)

    for word in words:
        if len(word) >= 3:
            if word.endswith("तो") or word.endswith("ते") or word.endswith("तात"):
                marathi_evidence += 2
            if word.endswith("ची") or word.endswith("चा") or word.endswith("चे"):
                marathi_evidence += 2
            if word.endswith("ळा") or word.endswith("ळे") or word.endswith("ळी"):
                marathi_evidence += 3

    hindi_evidence += sum(1 for w in words if w in {"है", "हैं"})

    if hindi_evidence > marathi_evidence and hindi_evidence >= 3:
        return "hi"
    return "mr"


# =============================================================================
# STT — IndicWhisper (hi/mr) + Whisper large-v3-turbo (en)
# =============================================================================

class SpeechToText:

    _INITIAL_PROMPT = (
        "महाराष्ट्र, मुख्यमंत्री, यशवंतराव चव्हाण, शिवाजी महाराज, रायगड, "
        "सह्याद्री, गोदावरी, कृष्णा नदी, मराठी, हिंदी, राजधानी मुंबई, "
        "Maharashtra, Chief Minister, capital, fort, river, history, geography."
    )

    def __init__(self, forced_lang=None):
        from faster_whisper import WhisperModel
        self.forced_lang  = forced_lang
        self.model_indic  = None
        self.model_en     = None

        # IndicWhisper for Hindi / Marathi
        print("  Loading IndicWhisper (hi/mr)...", end=" ", flush=True)
        try:
            path = CONFIG["whisper_indic"]
            if not os.path.exists(os.path.join(path, "model.bin")):
                raise FileNotFoundError(f"model.bin not found in {path}")
            self.model_indic = WhisperModel(path, device="cpu", compute_type="int8")
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

        # Whisper large-v3-turbo for English
        if forced_lang != "mr":
            print("  Loading Whisper large-v3-turbo (en)...", end=" ", flush=True)
            try:
                path = CONFIG["whisper_en"]
                if not os.path.exists(os.path.join(path, "model.bin")):
                    import huggingface_hub
                    print("(downloading ~800 MB, first run only)...", end=" ", flush=True)
                    huggingface_hub.snapshot_download(
                        _WHISPER_EN_REPO, local_dir=path, local_dir_use_symlinks=False
                    )
                self.model_en = WhisperModel(path, device="cpu", compute_type="int8")
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")

    def _normalize_audio(self, path):
        try:
            import soundfile as sf
            audio, sr = sf.read(path)
            mx = np.abs(audio).max()
            if mx > 0 and (mx < 0.3 or mx > 1.0):
                audio = audio * (0.8 / mx)
                sf.write(path, audio, sr)
        except Exception:
            pass

    def transcribe(self, audio_path, forced_lang=None):
        self._normalize_audio(audio_path)
        lang  = forced_lang or self.forced_lang
        model = self.model_en if lang == "en" else self.model_indic
        if not model:
            model = self.model_en or self.model_indic
        if not model:
            return {"text": "", "language": lang or "en"}

        kwargs = {
            "beam_size": 1,
            "without_timestamps": True,
            "condition_on_previous_text": False,
            "initial_prompt": self._INITIAL_PROMPT,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 400, "speech_pad_ms": 200},
        }
        if lang:
            kwargs["language"] = "hi" if lang == "mr" else lang

        try:
            segs, _ = model.transcribe(audio_path, **kwargs)
            text = " ".join(s.text.strip() for s in segs).strip()
        except Exception as e:
            print(f"  Transcription error: {e}")
            return {"text": "", "language": lang or "en"}

        detected = lang or (detect_language(text) if text else "en")
        return {"text": text, "language": detected}

    def record(self, forced_lang=None):
        import sounddevice as sd
        import soundfile as sf

        rec = []
        go  = True
        silence_chunks    = 0
        SILENCE_THRESHOLD = 500
        SILENCE_LIMIT     = 12

        def cb(d, f, t, s):
            nonlocal silence_chunks
            if not go: return
            rec.append(d.copy())
            if np.abs(d).max() < SILENCE_THRESHOLD:
                silence_chunks += 1
            else:
                silence_chunks = 0

        st = sd.InputStream(samplerate=16000, channels=1, dtype='int16',
                            callback=cb, blocksize=2048)
        st.start()
        print("\nSpeak... (auto-stops on silence, or press ENTER)")

        has_speech = False
        while go:
            try:
                if IS_WINDOWS:
                    import msvcrt
                    if msvcrt.kbhit(): msvcrt.getch(); break
                    time.sleep(0.05)
                else:
                    import select
                    r, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if r: sys.stdin.readline(); break
            except Exception:
                time.sleep(0.1)

            dur = len(rec) * 2048 / 16000 if rec else 0
            if dur > 0.5:
                if np.abs(rec[-1]).max() > SILENCE_THRESHOLD:
                    has_speech = True
                if has_speech and silence_chunks > SILENCE_LIMIT and dur > 1.0:
                    print(" (auto-stop)"); break
            if dur > 15:
                print(" (max length)"); break

        go = False; st.stop(); st.close()

        if not rec:
            return {"text": "", "language": "en"}
        audio = np.concatenate(rec)
        dur   = len(audio) / 16000
        if dur < 0.5:
            print("  Too short"); return {"text": "", "language": "en"}

        p = os.path.join(tempfile.gettempdir(), "vidya_in.wav")
        sf.write(p, audio, 16000)
        print(f"  {dur:.1f}s. Transcribing...", end=" ", flush=True)
        t0 = time.time()
        result = self.transcribe(p, forced_lang=forced_lang)
        print(f"({time.time()-t0:.1f}s)")
        return result


# =============================================================================
# TTS — pyttsx3 (en) + Meta MMS-TTS (hi/mr)
# =============================================================================

class TextToSpeech:
    """
    English  -> pyttsx3 (Windows SAPI / espeak — instant, no download)
    Hindi    -> Meta MMS-TTS facebook/mms-tts-hin  (~40 MB, auto-downloaded)
    Marathi  -> Meta MMS-TTS facebook/mms-tts-mar  (~40 MB, auto-downloaded)
    """

    def __init__(self):
        self._mms = {}

        print("  Loading pyttsx3 (English)...", end=" ", flush=True)
        try:
            import pyttsx3
            e = pyttsx3.init(); e.stop()
            self.pyttsx3_ok = True
            print("OK")
        except Exception as ex:
            self.pyttsx3_ok = False
            print(f"FAILED: {ex}")

        self._load_mms("hi", _MMS_HIN_REPO, _MMS_HIN_LOCAL)
        self._load_mms("mr", _MMS_MAR_REPO, _MMS_MAR_LOCAL)

    def _load_mms(self, lang, repo, local_dir):
        label = {"hi": "Hindi", "mr": "Marathi"}.get(lang, lang)
        print(f"  Loading MMS-TTS ({label})...", end=" ", flush=True)
        try:
            import huggingface_hub
            from transformers import VitsModel, AutoTokenizer
            if not os.path.exists(os.path.join(local_dir, "config.json")):
                print("(downloading ~40 MB)...", end=" ", flush=True)
                huggingface_hub.snapshot_download(
                    repo, local_dir=local_dir, local_dir_use_symlinks=False
                )
            model     = VitsModel.from_pretrained(local_dir)
            tokenizer = AutoTokenizer.from_pretrained(local_dir)
            model.eval()
            self._mms[lang] = {
                "model": model, "tokenizer": tokenizer,
                "sample_rate": model.config.sampling_rate,
            }
            print("OK")
        except Exception as ex:
            print(f"FAILED: {ex}")

    def speak(self, text, language):
        if not text: return
        if language in self._mms:
            self._mms_speak(text, language)
        else:
            self._pyttsx3_speak(text)

    def _pyttsx3_speak(self, text):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
        except Exception as ex:
            print(f"  pyttsx3 error: {ex}")

    def _mms_speak(self, text, language):
        import torch
        mms = self._mms.get(language)
        if not mms:
            self._pyttsx3_speak(text); return
        # MMS-TTS only handles Devanagari — strip other chars
        clean = re.sub(r'[^\u0900-\u097F\s।,!?०-९]', '', text).strip()
        if not clean:
            self._pyttsx3_speak(text); return
        try:
            inputs = mms["tokenizer"](clean, return_tensors="pt")
            with torch.no_grad():
                waveform = mms["model"](**inputs).waveform.squeeze().numpy()
            import soundfile as sf
            out = os.path.join(tempfile.gettempdir(), f"vidya_out_{language}.wav")
            sf.write(out, waveform, mms["sample_rate"])
            self._play(out)
        except Exception as ex:
            print(f"  MMS-TTS ({language}) error: {ex}")
            self._pyttsx3_speak(text)

    def _play(self, path):
        if not os.path.exists(path) or os.path.getsize(path) < 100:
            return
        if IS_WINDOWS:
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
        else:
            subprocess.run(["aplay", "-q", path], timeout=60, check=False)


# =============================================================================
# TTS QUEUE — sentences play in background while next search is ready
# =============================================================================

class TTSQueue:
    def __init__(self, tts: TextToSpeech, language: str):
        self._tts  = tts
        self._lang = language
        self._q    = queue.Queue()
        self._t    = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done(); break
            self._tts.speak(item, self._lang)
            self._q.task_done()

    def speak(self, text: str):
        self._q.put(text)

    def wait(self):
        self._q.join()
        self._q.put(None)
        self._t.join()


# =============================================================================
# KNOWLEDGE BASE — Hybrid Search (Multilingual Semantic + Keyword)
# =============================================================================

class KnowledgeBase:
    """
    1. Semantic  — ChromaDB + paraphrase-multilingual-MiniLM-L12-v2
    2. Keyword   — in-memory keyword + substring + synonym expansion
    3. Fusion    — semantic x 0.6 + keyword x 0.4
    Loads vidya_knowledge_base.json + balbharati_class6_rechunked.json
    """

    STOPWORDS = {
        "can", "you", "tell", "me", "something", "about", "what", "is", "the",
        "a", "an", "in", "of", "for", "to", "and", "or", "how", "do", "does",
        "did", "are", "was", "were", "which", "who", "where", "when", "why",
        "please", "give", "some", "any", "i", "my",
        "क्या", "कौन", "कैसे", "कहां", "कहाँ", "कब", "और", "या", "है", "हैं",
        "का", "की", "के", "में", "से", "पर", "को", "ने", "मुझे", "मैं",
        "हम", "तुम", "आप", "कुछ", "बारे", "बताओ", "बताइए", "मुझको",
        "सा", "सी", "कोई", "करो", "दो",
        "काय", "कसे", "कसा", "कोण", "कुठे", "केव्हा", "आणि", "पण", "किंवा",
        "आहे", "आहेत", "मला", "तुला", "मध्ये", "बद्दल", "साठी", "सांगा",
        "सांग", "विषयी", "काही", "कोणता", "कोणती",
    }

    SYNONYMS = {
        "तहवार":    ["त्योहार", "सण", "festival", "उत्सव", "चतुर्थी"],
        "तहवा":     ["त्योहार", "सण", "festival", "उत्सव"],
        "त्यौहार":  ["त्योहार", "सण", "festival"],
        "फेस्टिवल": ["festival", "त्योहार", "सण", "उत्सव", "गणेश"],
        "festival":  ["त्योहार", "सण", "उत्सव", "चतुर्थी", "गणेश"],
        "festivals": ["त्योहार", "सण", "उत्सव", "चतुर्थी", "गणेश"],
        "राष्ट्रपक्षी": ["राष्ट्रीय", "पक्षी", "national", "bird", "मोर"],
        "राष्ट्रपक्षू": ["राष्ट्रीय", "पक्षी", "bird", "मोर"],
        "राष्ट्रपशु":   ["राष्ट्रीय", "पशु", "national", "animal", "वाघ"],
        "गणपति":    ["गणेश", "ganesh", "chaturthi", "चतुर्थी"],
        "पूजा":     ["सण", "festival", "चतुर्थी", "उत्सव"],
        "history":  ["इतिहास", "formation", "गठन", "स्थापना", "ऐतिहासिक"],
        "culture":  ["संस्कृती", "संस्कृति", "कला", "नृत्य", "लावणी", "सण"],
        "food":     ["खाना", "पदार्थ", "cuisine", "भोजन", "पाककला", "खाद्य"],
        "fort":     ["किल्ला", "किला", "गड", "दुर्ग"],
        "forts":    ["किल्ला", "किला", "गड", "दुर्ग", "fort"],
        "किल्ला":   ["fort", "गड", "दुर्ग"],
        "bird":     ["पक्षी", "मोर", "peacock"],
        "national": ["राष्ट्रीय", "राष्ट्र"],
        "president": ["राष्ट्रपती", "राष्ट्रपति"],
        "prime":    ["पंतप्रधान", "प्रधानमंत्री"],
        "कल्चर":   ["संस्कृती", "संस्कृति", "culture", "कला"],
        "प्रसिद्ध": ["famous", "popular", "लोकप्रिय"],
        "geography": ["भूगोल"],
        "temperature": ["तापमान", "weather", "climate"],
        "ocean":    ["महासागर", "sea"],
        "rocks":    ["खडक", "minerals"],
        "energy":   ["ऊर्जा", "resources"],
    }

    def __init__(self, db_path, vidya_json_path, balbharati_json_path, embedding_model):
        self.chunks     = []
        self.chroma_col = None

        print("  Loading knowledge base...", end=" ", flush=True)
        self._load_vidya(vidya_json_path)
        vidya_count = len(self.chunks)
        self._load_balbharati(balbharati_json_path)
        bb_count = len(self.chunks) - vidya_count
        print(f"({vidya_count} vidya + {bb_count} balbharati = {len(self.chunks)} total)")

        print("  Loading multilingual embeddings...", end=" ", flush=True)
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            client = chromadb.PersistentClient(path=db_path)
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
            self.chroma_col = client.get_or_create_collection(
                name="vidya_multilingual",
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"}
            )
            if self.chroma_col.count() == 0 and self.chunks:
                print(f"\n  Indexing {len(self.chunks)} chunks (first run)...", end=" ", flush=True)
                self._index_chunks()
                print("OK")
            else:
                print(f"OK ({self.chroma_col.count()} indexed)")
        except Exception as e:
            print(f"FAILED: {e}\n  -> Keyword-only mode")

    @staticmethod
    def _is_garbled(text):
        """Skip chunks with >4% non-ASCII/non-Devanagari characters (corrupted PDF text)."""
        if not text or len(text) < 10:
            return True
        weird = sum(
            1 for c in text
            if ord(c) >= 128
            and not (0x0900 <= ord(c) <= 0x097F)
            and ord(c) not in (0x200B, 0x200C, 0x200D, 0x2013, 0x2014,
                               0x2018, 0x2019, 0x201C, 0x201D, 0x2026,
                               0x0964, 0x0965)
        )
        return (weird / len(text)) > 0.04

    def _load_vidya(self, json_path):
        if not json_path or not os.path.exists(json_path):
            print(f"\n  WARNING: Vidya KB not found: {json_path}")
            return
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for chunk in data.get("chunks", []):
            content = chunk.get("content", "").strip()
            if not content or len(content) < 20 or self._is_garbled(content):
                continue
            topic    = chunk.get("topic", "").lower()
            tags     = chunk.get("tags", [])
            tags_str = " ".join(tags).lower() if isinstance(tags, list) else str(tags).lower()
            searchable = f"{topic} {tags_str} {content}".lower()
            self.chunks.append({
                "id": str(len(self.chunks)),
                "content": content,
                "language": chunk.get("language", "en"),
                "topic": chunk.get("topic", ""),
                "tags": tags_str,
                "searchable": searchable,
            })

    def _load_balbharati(self, json_path):
        if not json_path or not os.path.exists(json_path):
            print(f"\n  WARNING: Balbharati KB not found: {json_path}")
            return
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("chunks", [])
        for item in items:
            content = item.get("content", "").strip()
            if not content or len(content) < 20 or self._is_garbled(content):
                continue
            topic    = item.get("chapter_title", item.get("topic", "")).strip()
            subject  = item.get("subject", "")
            cls      = item.get("class", "")
            tags_str = f"{subject} class{cls} {topic}".lower()
            searchable = f"{topic} {tags_str} {content}".lower()
            self.chunks.append({
                "id": str(len(self.chunks)),
                "content": content,
                "language": item.get("language", "en"),
                "topic": topic,
                "tags": tags_str,
                "searchable": searchable,
            })

    def _index_chunks(self):
        BATCH = 50
        for i in range(0, len(self.chunks), BATCH):
            batch = self.chunks[i:i + BATCH]
            self.chroma_col.add(
                ids=[c["id"] for c in batch],
                documents=[c["content"] for c in batch],
                metadatas=[{"language": c["language"], "topic": c["topic"]} for c in batch],
            )

    def find_answer(self, query, language="en"):
        if not query or not query.strip():
            return NO_ANSWER.get(language, NO_ANSWER["en"]), 0.0

        keywords = self._get_keywords(query.lower())

        # 1. Semantic search
        semantic_results = {}
        if self.chroma_col and self.chroma_col.count() > 0:
            try:
                n = min(10, self.chroma_col.count())
                results = self.chroma_col.query(
                    query_texts=[query], n_results=n,
                    include=["documents", "metadatas", "distances"]
                )
                if results["documents"] and results["documents"][0]:
                    for i, doc in enumerate(results["documents"][0]):
                        doc_id    = results["ids"][0][i]
                        dist      = results["distances"][0][i]
                        sem_score = max(0.0, 1.0 - dist)
                        meta      = results["metadatas"][0][i] if results.get("metadatas") else {}
                        semantic_results[doc_id] = {
                            "content": doc,
                            "language": meta.get("language", "en"),
                            "topic": meta.get("topic", ""),
                            "sem_score": sem_score,
                        }
            except Exception as e:
                print(f"  [WARN] ChromaDB: {e}")

        # 2. Keyword search
        keyword_results = {}
        if keywords:
            for chunk in self.chunks:
                hits = 0.0
                for kw in keywords:
                    if kw in chunk["searchable"]:
                        hits += 1.0
                        if kw in chunk["topic"].lower() or kw in chunk["tags"]:
                            hits += 0.5
                    elif len(kw) >= 3 and kw[:max(3, len(kw)-1)] in chunk["searchable"]:
                        hits += 0.3
                if hits > 0:
                    keyword_results[chunk["id"]] = {
                        "content": chunk["content"],
                        "language": chunk["language"],
                        "topic": chunk["topic"],
                        "kw_score": hits / len(keywords),
                    }

        # 3. Score fusion
        fused = []
        for doc_id in set(semantic_results) | set(keyword_results):
            sem       = semantic_results.get(doc_id, {})
            kw        = keyword_results.get(doc_id, {})
            sem_score = sem.get("sem_score", 0.0)
            kw_score  = kw.get("kw_score",  0.0)
            score     = sem_score * 0.6 + kw_score * 0.4
            content   = sem.get("content") or kw.get("content", "")
            lang      = sem.get("language") or kw.get("language", "en")
            topic     = sem.get("topic") or kw.get("topic", "")
            if not content: continue
            if lang == language: score += 0.15
            fused.append({
                "id": doc_id, "content": content, "language": lang,
                "topic": topic, "score": score,
                "sem_score": sem_score, "kw_score": kw_score,
            })

        if not fused:
            return NO_ANSWER.get(language, NO_ANSWER["en"]), 0.0

        fused.sort(key=lambda x: -x["score"])

        for r in fused[:3]:
            print(f"  [SEARCH] score={r['score']:.2f} sem={r['sem_score']:.2f} "
                  f"kw={r['kw_score']:.2f} lang={r['language']} | {r['topic'][:30]}")

        best = fused[0]
        for r in fused[:5]:
            if r["language"] == language and r["score"] >= best["score"] * 0.75:
                best = r; break

        if best["score"] < 0.20:
            return NO_ANSWER.get(language, NO_ANSWER["en"]), best["score"]

        return best["content"], best["score"]

    def _get_keywords(self, query):
        words    = re.split(r'[\s,.\?!।:;]+', query)
        keywords = [w for w in words if w and len(w) > 1 and w.lower() not in self.STOPWORDS]
        expanded = list(keywords)
        for kw in keywords:
            for key in (kw, kw.lower()):
                if key in self.SYNONYMS:
                    expanded.extend(self.SYNONYMS[key])
        seen, result = set(), []
        for w in expanded:
            if w.lower() not in seen:
                seen.add(w.lower()); result.append(w)
        return result


# =============================================================================
# CHATBOT
# =============================================================================

class Chatbot:
    LANG_NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

    def __init__(self, forced_lang=None):
        platform_name = "Windows" if IS_WINDOWS else "Raspberry Pi"
        print("=" * 55)
        print(f"  Vidya v18 — {platform_name} | No LLM")
        print("=" * 55, "\n")

        self.forced_lang = forced_lang

        print("[1/3] Speech-to-Text")
        self.stt = SpeechToText(forced_lang=forced_lang)

        print("\n[2/3] Knowledge Base")
        self.kb = KnowledgeBase(
            CONFIG["rag_db_path"],
            CONFIG["vidya_knowledge"],
            CONFIG["balbharati_knowledge"],
            CONFIG["embedding_model"],
        )

        print("\n[3/3] Text-to-Speech")
        self.tts = TextToSpeech()

        print("\n  Vidya ready!\n")

    def _respond(self, text, lang):
        response_lang = self.forced_lang if self.forced_lang else lang
        lang_name     = self.LANG_NAMES.get(response_lang, response_lang)
        t0 = time.time()

        # Canned responses — instant
        canned = _canned(text, response_lang)
        if canned:
            print(f"\n  Vidya ({lang_name}): {canned}\n")
            self.tts.speak(canned, response_lang)
            print(f"  Total: {time.time()-t0:.1f}s"); print("-" * 50)
            return

        # Search
        answer, score = self.kb.find_answer(text, language=response_lang)
        answer = self._trim(answer)
        print(f"  Search: {time.time()-t0:.2f}s (score: {score:.2f})")
        print(f"\n  Vidya ({lang_name}): {answer}\n")

        t2 = time.time()
        self.tts.speak(answer, response_lang)
        print(f"  TTS: {time.time()-t2:.1f}s | Total: {time.time()-t0:.1f}s")
        print("-" * 50)

    def _trim(self, text, max_sentences=3):
        sentences = re.split(r'(?<=[.।!?])\s+', text)
        return ' '.join(sentences[:max_sentences]) if len(sentences) > max_sentences else text

    def voice_loop(self):
        print("  VOICE MODE — press ENTER -> speak -> auto-stop")
        print("  Type 'text' to switch, 'quit' to exit\n")
        while True:
            try:
                cmd = input("ENTER to record (or type): ").strip().lower()
                if cmd == "text":   return
                if cmd == "quit":   sys.exit(0)
                if cmd:
                    self._respond(cmd, self.forced_lang or detect_language(cmd))
                    continue

                result = self.stt.record(forced_lang=self.forced_lang)
                if not result["text"]:
                    print("  (nothing heard)"); continue

                text = result["text"]
                lang = self.forced_lang or result["language"]
                print(f"  You ({self.LANG_NAMES.get(lang, '')}): {text}")
                self._respond(text, lang)

            except KeyboardInterrupt:
                print("\nStopped."); break

    def text_loop(self):
        print("  TEXT MODE — type your question")
        print("  Type 'voice' to switch, 'quit' to exit\n")
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:    continue
            if text.lower() == "quit":  break
            if text.lower() == "voice":
                self.voice_loop(); print("\nText mode.\n"); continue
            self._respond(text, self.forced_lang or detect_language(text))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n  Select Language:")
    print("  1. Marathi   2. Hindi   3. English   4. Auto-detect")
    choice = input("  Choose (1/2/3/4): ").strip()
    forced = {"1": "mr", "2": "hi", "3": "en"}.get(choice)
    names  = {"mr": "Marathi", "hi": "Hindi", "en": "English"}
    print(f"\n  {names.get(forced, 'Auto-detect')} selected\n")

    bot = Chatbot(forced_lang=forced)

    print("  1. Text Mode  2. Voice Mode")
    c = input("  Choose (1/2): ").strip()
    if c == "2":
        bot.voice_loop()
    else:
        bot.text_loop()


if __name__ == "__main__":
    main()
