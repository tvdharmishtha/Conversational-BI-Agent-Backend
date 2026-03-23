from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import shutil
import os
import pandas as pd
import speech_recognition as sr
import io
import base64
from analyzer import analyze_query, get_column_info, suggest_currency
from groq import Groq
from dotenv import load_dotenv
from gtts import gTTS

load_dotenv()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "data/current.xlsx"
df = None  # global dataframe


class QueryRequest(BaseModel):
    question: str


class ColumnsResponse(BaseModel):
    columns: list
    column_types: dict


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    global df

    os.makedirs("data", exist_ok=True)

    with open(DATA_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = pd.read_excel(DATA_PATH)

    # Get column information
    column_info = get_column_info(df)
    
    # Get currency suggestion
    currency_info = suggest_currency(df)

    return {
        "message": "File uploaded successfully",
        "columns": column_info["columns"],
        "column_types": column_info["column_types"],
        "row_count": len(df),
        "currency": currency_info
    }


@app.get("/columns")
def get_columns():
    global df

    if df is None:
        return {"error": "No file uploaded", "columns": [], "column_types": {}}

    column_info = get_column_info(df)

    return {
        "columns": column_info["columns"],
        "column_types": column_info["column_types"],
        "row_count": len(df),
        "preview": df.head(5).to_dict(orient='records')
    }


@app.post("/ask")
def ask(req: QueryRequest):
    global df

    if df is None:
        return {"error": "No file uploaded"}

    return analyze_query(req.question, df)


# Voice AI Endpoints
class AudioTextRequest(BaseModel):
    audio_data: str  # Base64 encoded audio


class TextToSpeechRequest(BaseModel):
    text: str


@app.post("/voice/transcribe")
def transcribe_audio(req: AudioTextRequest):
    """Transcribe audio from base64 encoded string using Google Speech Recognition"""
    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(req.audio_data)
        
        # Save to temporary file
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        
        # Use speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile("temp_audio.wav") as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        
        # Clean up
        os.remove("temp_audio.wav")
        
        return {
            "success": True,
            "text": text,
            "message": "Audio transcribed successfully"
        }
    except sr.UnknownValueError:
        return {
            "success": False,
            "text": "",
            "message": "Could not understand audio. Please speak clearly."
        }
    except sr.RequestError as e:
        return {
            "success": False,
            "text": "",
            "message": f"Speech recognition service error: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "message": f"Error processing audio: {str(e)}"
        }


@app.post("/voice/transcribe-file")
async def transcribe_audio_file(file: UploadFile = File(...)):
    """Transcribe audio file using Groq Whisper API for better accuracy"""
    try:
        # Save uploaded file temporarily
        audio_path = f"temp_{file.filename}"
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Transcribe using Groq Whisper
        with open(audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="json",
                language="en"
            )
        
        # Clean up
        os.remove(audio_path)
        
        return {
            "success": True,
            "text": transcription.text,
            "message": "Audio file transcribed successfully"
        }
    except Exception as e:
        # Clean up on error
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return {
            "success": False,
            "text": "",
            "message": f"Error transcribing audio file: {str(e)}"
        }


@app.post("/voice/listen")
def listen_from_microphone():
    """Real-time speech recognition from microphone (for testing)"""
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            return {
                "success": True,
                "message": "Microphone ready. Please speak now..."
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Microphone error: {str(e)}"
        }


@app.get("/voice/status")
def voice_status():
    """Check voice recognition availability"""
    try:
        # Check if we have microphone access
        mics = sr.Microphone.list_microphone_names()
        return {
            "available": True,
            "microphones": mics,
            "message": "Voice recognition is ready"
        }
    except Exception as e:
        return {
            "available": False,
            "microphones": [],
            "message": f"Voice recognition not available: {str(e)}"
        }


class TextToSpeechRequest(BaseModel):
    text: str
    lang: str = "en"


@app.post("/voice/speak")
def text_to_speech(req: TextToSpeechRequest):
    """Convert text to speech using Google TTS"""
    try:
        tts = gTTS(text=req.text, lang=req.lang)
        
        # Save to bytes buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Return as base64
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return {
            "success": True,
            "audio_data": audio_base64,
            "message": "Text converted to speech successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"TTS error: {str(e)}"
        }


@app.post("/voice/speak-stream")
def text_to_speech_stream(req: TextToSpeechRequest):
    """Stream text to speech directly as audio file"""
    try:
        tts = gTTS(text=req.text, lang=req.lang)
        
        # Save to bytes buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return Response(
            content=audio_buffer.read(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except Exception as e:
        return {"error": f"TTS error: {str(e)}"}


class VoiceCommandRequest(BaseModel):
    question: str
    df_data: dict = None  # Optional: send current dataframe summary


@app.post("/voice/command")
def voice_command(req: VoiceCommandRequest):
    """Process voice command and return AI response"""
    global df
    
    try:
        # If no dataframe loaded, return error
        if df is None:
            return {
                "success": False,
                "message": "No data loaded. Please upload a file first.",
                "requires_upload": True
            }
        
        # Process the question through the analyzer
        result = analyze_query(req.question, df)
        
        return {
            "success": True,
            "insight": result.get("insight", ""),
            "charts": result.get("charts", []),
            "message": "Voice command processed successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error processing command: {str(e)}"
        }
