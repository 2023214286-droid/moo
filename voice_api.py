import os
import base64
import io

import requests
import edge_tts
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

# 🔥 MODEL LLM
LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-20b")

# 🔥 SUARA LELAKI - MR. MOO
# Pilihan suara lelaki edge-tts:
#   en-SG-WayneNeural      - Lelaki Singapore English (paling sesuai Manglish)
#   en-MY-AhmadNeural      - Lelaki Malaysia (kalau ada)
#   en-US-GuyNeural        - Lelaki US
#   en-GB-RyanNeural       - Lelaki UK
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-SG-WayneNeural")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ============================================================
# ✅ APP
# ============================================================

app = FastAPI(title="MOO MOO Voice API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_api_key():
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")


# ============================================================
# GROQ WHISPER - SPEECH TO TEXT
# ============================================================

def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    check_api_key()
    url = f"{GROQ_BASE_URL}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    files = {"file": (filename, audio_bytes, "audio/webm")}
    data = {
        "model": WHISPER_MODEL,
        "language": "ms",
        "response_format": "json",
        "temperature": "0"
    }
    response = requests.post(url, headers=headers, files=files, data=data, timeout=120)
    if not response.ok:
        raise HTTPException(status_code=502, detail=f"Groq Whisper error: {response.text}")
    result = response.json()
    transcript = result.get("text", "").strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="Empty transcript.")
    return transcript


# ============================================================
# GROQ LLM - MANGLISH RESPONSE (MR. MOO)
# ============================================================

def generate_response(transcript, voice_step="greeting", user_name="", user_genre="",
                     user_career="", waiting_for_confirmation="false", confirmation_type=""):
    check_api_key()

    system_prompt = """
Kau Mr. Moo, suara AI mesra untuk booth MOO MOO Fun AI Swap Photobooth.
Kau cakap MANGLISH (Bahasa Melayu campur English).

3 STEP SAHAJA:
STEP 1 - NAMA: Tanya "Hi! Nama saya Mister Moo. Siapa nama awak?" then confirm "Nama awak [name], betul ke?"
STEP 2 - GENRE: Tanya "Awak nak jadi apa? Ada Realistic, Fantasy, Horror, Sci-Fi, Animation, atau Funny."
STEP 3 - CAREER: Tanya "Dalam [genre], awak nak jadi apa? Ada [careers]."

RULES:
- Max 15 patah perkataan
- MANGLISH
- Jawapan pendek sahaja
- Mesra dan comel
""".strip()

    user_prompt = f"""
Step: {voice_step}
Nama user: {user_name or 'belum tau'}
Genre: {user_genre or 'belum pilih'}
Career: {user_career or 'belum pilih'}

User cakap: "{transcript}"

Bagi SATU jawapan Manglish pendek (max 15 perkataan).
""".strip()

    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 80
    }

    # Cuba LLM
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.ok:
            result = response.json()
            text = result["choices"][0]["message"]["content"].strip()
            if text:
                print(f"✅ LLM: {text}")
                return text
            else:
                print(f"⚠️ LLM return kosong")
        else:
            print(f"⚠️ LLM error {response.status_code}: {response.text[:200]}")
    except Exception as e:
        print(f"⚠️ LLM exception: {e}")

    # Fallback
    print(f"🔄 Fallback untuk step: {voice_step}")
    fallbacks = {
        'greeting': "Hi! Saya Mr. Moo. Siapa nama awak?",
        'confirm_name': f"Nama awak {user_name}, betul ke?",
        'genre': "Awak nak jadi apa? Ada Realistic, Fantasy, Horror, Sci-Fi, Animation atau Funny.",
        'confirm_genre': f"{user_genre}, betul ke?",
        'career': f"Dalam {user_genre}, awak nak jadi apa?",
        'confirm_career': f"Awak nak jadi {user_career}, betul ke?",
        'done': "Jom ambil gambar!"
    }
    return fallbacks.get(voice_step, "Cuba sebut sekali lagi.")


# ============================================================
# EDGE-TTS (SUARA LELAKI - MR. MOO)
# ============================================================

async def text_to_speech_base64(text: str) -> str:
    if not text or not text.strip():
        text = "Ok"

    text = text.strip()

    try:
        communicate = edge_tts.Communicate(
            text=text,
            voice=EDGE_TTS_VOICE,
            rate="-5%",       # Sikit lambat untuk jelas
            pitch="-2Hz"      # Sikit rendah untuk bunyi lelaki
        )
        audio_buffer = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_bytes = audio_buffer.getvalue()
        if not audio_bytes:
            return ""

        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        return f"data:audio/mpeg;base64,{encoded}"
    except Exception as exc:
        print(f"⚠️ TTS error: {exc}")
        return ""


# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/api/voice")
async def voice_endpoint(
    audio: UploadFile = File(...),
    voice_step: str = Form("greeting"),
    user_name: str = Form(""),
    genre: str = Form(""),
    career: str = Form(""),
    waiting_for_confirmation: str = Form("false"),
    confirmation_type: str = Form("")
):
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio.")

        transcript = transcribe_audio(audio_bytes, audio.filename or "voice.webm")
        print(f"🎤 User: {transcript}")

        ai_response = generate_response(
            transcript=transcript,
            voice_step=voice_step,
            user_name=user_name,
            user_genre=genre,
            user_career=career,
            waiting_for_confirmation=waiting_for_confirmation,
            confirmation_type=confirmation_type
        )
        print(f"🤖 AI: {ai_response}")

        audio_data = await text_to_speech_base64(ai_response)

        return {
            "success": True,
            "transcript": transcript,
            "response": ai_response,
            "audio": audio_data,
            "voice": EDGE_TTS_VOICE
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TTSRequest(BaseModel):
    text: str


@app.post("/api/tts")
async def tts_endpoint(request: TTSRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text empty.")
    audio_data = await text_to_speech_base64(request.text.strip())
    return {"success": True, "audio": audio_data, "voice": EDGE_TTS_VOICE}


@app.get("/")
async def root():
    return {
        "success": True,
        "message": "MOO MOO Voice API (Mr. Moo - Manglish)",
        "voice": EDGE_TTS_VOICE,
        "llm_model": LLM_MODEL
    }


# ============================================================
# RUN - Support local & cloud (Render/Railway)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)