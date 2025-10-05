import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
import speech_recognition as sr
import yt_dlp
import re
import asyncio

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY") # Not directly used in this version, but kept for consistency

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create a temporary directory for uploads and processed files
UPLOAD_DIR = "temp_uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "transcribed_text": None})

@app.post("/transcribe", response_class=HTMLResponse)
async def transcribe_audio_video(request: Request, file: UploadFile = File(None), video_url: str = Form(None)):
    transcribed_text = ""
    error_message = None
    temp_file_path = None
    audio_file_path = None

    try:
        if file:
            # Handle file upload
            temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Handle file upload
            temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"Uploaded file saved to: {temp_file_path}")
            
            # Convert to audio if it's a video or ensure it's a compatible audio format
            audio_file_path = convert_to_wav(temp_file_path)
            if not audio_file_path:
                raise HTTPException(status_code=400, detail="Could not convert file to audio.")
            print(f"Converted audio file path: {audio_file_path}")

        elif video_url:
            # Handle video URL
            if not is_valid_url(video_url):
                raise HTTPException(status_code=400, detail="Invalid URL provided.")
            print(f"Attempting to download audio from URL: {video_url}")
            
            audio_file_path = await download_audio_from_url(video_url)
            if not audio_file_path:
                raise HTTPException(status_code=400, detail="Could not download audio from the provided URL.")
            print(f"Downloaded audio file path from URL: {audio_file_path}")
        else:
            raise HTTPException(status_code=400, detail="No file or video URL provided.")

        # Transcribe the audio
        print(f"Starting transcription for: {audio_file_path}")
        transcribed_text = transcribe_audio(audio_file_path)
        print(f"Transcription result: {transcribed_text}")

    except HTTPException as e:
        error_message = e.detail
        print(f"HTTPException: {error_message}")
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(f"Unexpected error: {e}")
    finally:
        # Clean up temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")
        if audio_file_path and os.path.exists(audio_file_path):
            os.remove(audio_file_path)
            print(f"Cleaned up audio file: {audio_file_path}")
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "transcribed_text": transcribed_text,
            "error_message": error_message
        })

def convert_to_wav(file_path: str) -> str:
    """Converts an audio/video file to WAV format."""
    print(f"Attempting to convert {file_path} to WAV.")
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        output_wav_path = os.path.join(UPLOAD_DIR, f"{os.path.basename(file_path).split('.')[0]}.wav")

        if file_extension in ['.mp3', '.wav', '.ogg', '.flac', '.aac']:
            audio = AudioSegment.from_file(file_path, format=file_extension[1:])
        elif file_extension in ['.mp4', '.mov', '.avi', '.mkv']:
            audio = AudioSegment.from_file(file_path, format=file_extension[1:])
        else:
            print(f"Unsupported file format: {file_extension}")
            return None
        
        audio.export(output_wav_path, format="wav")
        print(f"Successfully converted {file_path} to {output_wav_path}")
        return output_wav_path
    except Exception as e:
        print(f"Error converting file {file_path} to WAV: {e}")
        return None

async def download_audio_from_url(url: str) -> str:
    """Downloads audio from a video URL using yt-dlp and converts it to WAV."""
    output_path = os.path.join(UPLOAD_DIR, "%(id)s.%(ext)s")
    wav_output_path = None
    print(f"Starting audio download from URL: {url}")
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': output_path,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info)
            print(f"yt-dlp downloaded file: {downloaded_file}")
            # yt-dlp might add .wav or other extensions, so we need to find the actual file
            base_name = os.path.splitext(downloaded_file)[0]
            # Search for the .wav file created by FFmpegExtractAudio
            for f in os.listdir(UPLOAD_DIR):
                if f.startswith(os.path.basename(base_name)) and f.endswith('.wav'):
                    wav_output_path = os.path.join(UPLOAD_DIR, f)
                    break
            
            if not wav_output_path or not os.path.exists(wav_output_path):
                # Fallback if direct .wav not found, try to convert the downloaded file
                print(f"Direct WAV not found, attempting conversion from: {downloaded_file}")
                wav_output_path = convert_to_wav(downloaded_file)
                if os.path.exists(downloaded_file):
                    os.remove(downloaded_file) # Clean up original downloaded file
            
            print(f"Final WAV audio path after download/conversion: {wav_output_path}")
            return wav_output_path
    except Exception as e:
        print(f"Error downloading audio from URL {url}: {e}")
        return None

def transcribe_audio(audio_file_path: str) -> str:
    """Transcribes audio from a WAV file using Google Speech Recognition."""
    print(f"Attempting to transcribe audio from: {audio_file_path}")
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = r.record(source)
        transcription = r.recognize_google(audio)
        print(f"Google Speech Recognition result: {transcription}")
        return transcription
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return f"Could not request results from Google Speech Recognition service; {e}"
    except Exception as e:
        print(f"Error during transcription for {audio_file_path}: {e}")
        return f"Error during transcription: {e}"

def is_valid_url(url: str) -> bool:
    """Checks if the provided string is a valid URL."""
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain...
        r'localhost|' # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None
