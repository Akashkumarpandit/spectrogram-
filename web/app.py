import os
import io
import tempfile
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf

from src.predict import DeepfakeDetector
from synthetic_data_generator import generate_synthetic_audio


app = FastAPI(title="SpectraGuard: AASIST Voice Anti-Spoofing & Deepfake Detection", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Detector (singleton)
CHECKPOINT_PATH = "./checkpoints/best_model.pth"
if not os.path.isfile(CHECKPOINT_PATH):
    CHECKPOINT_PATH = None

detector = DeepfakeDetector(checkpoint_path=CHECKPOINT_PATH)

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "AASIST",
        "device": str(detector.device),
        "dataset_compatibility": ["ASVspoof 2019 (LA/DF)", "ASVspoof 2021 (LA/DF)"],
    }


@app.post("/api/predict")
async def predict_audio(file: UploadFile = File(...)):
    """Analyze uploaded audio for voice spoofing / AI deepfake markers."""
    valid_exts = [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".webm"]
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in valid_exts and ext != "":
        # Still attempt to parse
        pass

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file provided.")

        suffix = ext if ext else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Run inference
        result = detector.predict(tmp_path)

        # Extract waveform points and acoustic comparison data
        try:
            from src.utils.audio_io import robust_load_audio
            from src.utils.acoustic_analysis import extract_spectral_biometrics
            waveform, sr = robust_load_audio(tmp_path, target_sr=16000)
            audio_data = waveform.cpu().numpy()
            duration = round(len(audio_data) / sr, 2)

            # Downsample to 200 points for sleek SVG/Canvas rendering
            step = max(1, len(audio_data) // 200)
            downsampled_waveform = audio_data[::step][:200].tolist()

            # Extract detailed acoustic biometric differences
            acoustic_data = extract_spectral_biometrics(audio_data, sample_rate=sr)
        except Exception as e:
            duration = 4.0
            downsampled_waveform = []
            acoustic_data = {}

        # Cleanup temp file
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)

        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "duration_sec": duration,
            "prediction": result,
            "waveform_preview": downsampled_waveform,
            "acoustic_analysis": acoustic_data,
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/sample/{sample_type}")
async def get_sample_audio(sample_type: str):
    """Generate or retrieve a sample audio file (bonafide or spoof) for instant testing."""
    is_spoof = (sample_type.lower() == "spoof" or sample_type.lower() == "deepfake")
    sr = 16000
    audio = generate_synthetic_audio(duration_sec=3.5, sr=sr, is_spoof=is_spoof)

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, audio, sr)

    filename = "sample_deepfake_synthetic.wav" if is_spoof else "sample_bonafide_human.wav"
    return FileResponse(temp_file.name, media_type="audio/wav", filename=filename)


# Mount static files for modern web frontend
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run AASIST Deepfake Detection Web UI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    args = parser.parse_args()

    print(f"\n🚀 Launching AASIST Voice Anti-Spoofing Web Server on http://localhost:{args.port}")
    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=False)
