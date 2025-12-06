#!/usr/bin/env python3
"""
FastAPI ML Inference Service for Cough Classification
Provides an API endpoint for real-time cough prediction (dry vs wet)
"""

import os
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import librosa
import numpy as np
import joblib
import json
import base64
import tempfile
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="Cough Classification API",
    description="ML-powered API for classifying cough sounds as dry or wet",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scaler
model = None
scaler = None
config = None

# Request model
class PredictionRequest(BaseModel):
    audio: str  # Base64-encoded audio data

# Response model
class PredictionResponse(BaseModel):
    predicted_cough_type: str  # "dry" or "wet"
    confidence_score: float  # 0-100
    message: Optional[str] = None

def load_models():
    """Load trained model, scaler, and configuration"""
    global model, scaler, config
    
    model_dir = "/home/ubuntu/coughsense_project/ml_training/trained_model"
    
    try:
        # Load model
        model_path = os.path.join(model_dir, "cough_classifier_model.pkl")
        model = joblib.load(model_path)
        print(f"✓ Model loaded from: {model_path}")
        
        # Load scaler
        scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
        scaler = joblib.load(scaler_path)
        print(f"✓ Scaler loaded from: {scaler_path}")
        
        # Load configuration
        config_path = os.path.join(model_dir, "model_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Configuration loaded from: {config_path}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False

def extract_audio_features(audio_path, target_duration=5.0, sr=22050):
    """
    Extract audio features from a cough recording
    Same as training script
    """
    try:
        # Load audio file
        audio, sample_rate = librosa.load(audio_path, sr=sr, duration=target_duration, res_type='kaiser_fast')
        
        # Pad or trim to fixed duration
        target_length = int(target_duration * sr)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        features = []
        
        # 1. MFCC features (13 coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()
        mfcc_std = np.std(mfcc, axis=1).tolist()
        mfcc_min = np.min(mfcc, axis=1).tolist()
        mfcc_max = np.max(mfcc, axis=1).tolist()
        features.extend(mfcc_mean)
        features.extend(mfcc_std)
        features.extend(mfcc_min)
        features.extend(mfcc_max)
        
        # 2. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(float(np.mean(spectral_centroid)))
        features.append(float(np.std(spectral_centroid)))
        features.append(float(np.min(spectral_centroid)))
        features.append(float(np.max(spectral_centroid)))
        
        # 3. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features.append(float(np.mean(spectral_bandwidth)))
        features.append(float(np.std(spectral_bandwidth)))
        features.append(float(np.min(spectral_bandwidth)))
        features.append(float(np.max(spectral_bandwidth)))
        
        # 4. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(float(np.mean(spectral_rolloff)))
        features.append(float(np.std(spectral_rolloff)))
        features.append(float(np.min(spectral_rolloff)))
        features.append(float(np.max(spectral_rolloff)))
        
        # 5. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(float(np.mean(zcr)))
        features.append(float(np.std(zcr)))
        features.append(float(np.min(zcr)))
        features.append(float(np.max(zcr)))
        
        # 6. Chroma Features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.append(float(np.mean(chroma)))
        features.append(float(np.std(chroma)))
        features.append(float(np.min(chroma)))
        features.append(float(np.max(chroma)))
        
        # 7. Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features.append(float(np.mean(spectral_contrast)))
        features.append(float(np.std(spectral_contrast)))
        features.append(float(np.min(spectral_contrast)))
        features.append(float(np.max(spectral_contrast)))
        
        # 8. Tonnetz (Tonal Centroid Features)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        features.append(float(np.mean(tonnetz)))
        features.append(float(np.std(tonnetz)))
        features.append(float(np.min(tonnetz)))
        features.append(float(np.max(tonnetz)))
        
        # 9. RMS Energy
        rms = librosa.feature.rms(y=audio)
        features.append(float(np.mean(rms)))
        features.append(float(np.std(rms)))
        features.append(float(np.min(rms)))
        features.append(float(np.max(rms)))
        
        # 10. Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features.append(float(tempo))
        
        return np.array(features)
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("="*60)
    print("Starting Cough Classification ML Service")
    print("="*60)
    success = load_models()
    if not success:
        print("WARNING: Models not loaded. Service will not function correctly.")
    print("="*60)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Cough Classification ML API",
        "status": "running",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "config_loaded": config is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_cough(request: PredictionRequest):
    """
    Predict cough type from base64-encoded audio
    
    Args:
        request: PredictionRequest with base64-encoded audio
        
    Returns:
        PredictionResponse with predicted_cough_type and confidence_score
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    try:
        # Decode base64 audio
        audio_data = base64.b64decode(request.audio)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        try:
            # Extract features
            features = extract_audio_features(temp_path)
            
            if features is None:
                raise HTTPException(status_code=400, detail="Failed to extract audio features")
            
            # Reshape and scale features
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            prediction_proba = model.predict_proba(features_scaled)[0]
            
            # Get predicted class and confidence
            cough_type = config['label_mapping'][str(prediction)]
            confidence = float(prediction_proba[prediction] * 100)  # Convert to percentage
            
            return PredictionResponse(
                predicted_cough_type=cough_type,
                confidence_score=round(confidence, 2),
                message="Prediction completed successfully"
            )
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("Starting FastAPI server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
