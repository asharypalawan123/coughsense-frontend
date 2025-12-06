#!/usr/bin/env python3
"""
Train model with correct feature dimensions
The feature extraction function produces 85 features, not 109
"""

import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Import the actual feature extraction to verify feature count
import sys
import librosa
import tempfile
import soundfile as sf

def extract_audio_features_actual(audio_path, target_duration=5.0, sr=22050):
    """
    Actual feature extraction matching ml_inference_service.py
    Returns 85 features (not 109)
    """
    try:
        audio, sample_rate = librosa.load(audio_path, sr=sr, duration=target_duration, res_type='kaiser_fast')
        
        target_length = int(target_duration * sr)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        features = []
        
        # 1. MFCC (52 features: 13 coefficients x 4 stats)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        features.extend(np.min(mfcc, axis=1))
        features.extend(np.max(mfcc, axis=1))
        
        # 2-9. Other features (4 stats each = 32 features)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid),
                        np.min(spectral_centroid), np.max(spectral_centroid)])
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                        np.min(spectral_bandwidth), np.max(spectral_bandwidth)])
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff),
                        np.min(spectral_rolloff), np.max(spectral_rolloff)])
        
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.extend([np.mean(zcr), np.std(zcr), np.min(zcr), np.max(zcr)])
        
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.extend([np.mean(chroma), np.std(chroma), np.min(chroma), np.max(chroma)])
        
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features.extend([np.mean(spectral_contrast), np.std(spectral_contrast),
                        np.min(spectral_contrast), np.max(spectral_contrast)])
        
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        features.extend([np.mean(tonnetz), np.std(tonnetz), np.min(tonnetz), np.max(tonnetz)])
        
        rms = librosa.feature.rms(y=audio)
        features.extend([np.mean(rms), np.std(rms), np.min(rms), np.max(rms)])
        
        # 10. Tempo (1 feature)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features.append(tempo)
        
        return np.array(features)
    except:
        return None

def create_synthetic_cough_features(n_samples=800, n_features=85):
    """
    Create synthetic cough features with CORRECT dimension (85 features)
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    # Create DRY cough samples (label=0, 75%)
    for i in range(int(n_samples * 0.75)):
        features = np.random.randn(n_features)
        
        # MFCC features (0-51): higher mean for dry coughs
        features[0:13] = np.random.normal(10, 5, 13)  # MFCC mean
        features[13:26] = np.abs(np.random.normal(5, 2, 13))  # MFCC std
        
        # Spectral Centroid (52-55): higher for dry (brighter)
        features[52] = np.random.normal(2500, 400)
        
        # Spectral Bandwidth (56-59): lower for dry (sharper)
        features[56] = np.random.normal(1800, 300)
        
        # Zero Crossing Rate (64-67): higher for dry (more abrupt)
        features[64] = np.random.normal(0.15, 0.03)
        
        X.append(features)
        y.append(0)  # dry
    
    # Create WET cough samples (label=1, 25%)
    for i in range(int(n_samples * 0.25)):
        features = np.random.randn(n_features)
        
        # MFCC features: lower mean for wet coughs
        features[0:13] = np.random.normal(8, 4, 13)
        features[13:26] = np.abs(np.random.normal(4, 2, 13))
        
        # Spectral Centroid: lower for wet (darker)
        features[52] = np.random.normal(2000, 350)
        
        # Spectral Bandwidth: higher for wet (more complex)
        features[56] = np.random.normal(2200, 400)
        
        # Zero Crossing Rate: lower for wet (more continuous)
        features[64] = np.random.normal(0.10, 0.02)
        
        X.append(features)
        y.append(1)  # wet
    
    return np.array(X), np.array(y)

print("="*60)
print("TRAINING CORRECTED MODEL (85 features)")
print("="*60)

# Verify actual feature count
print("\nVerifying feature extraction dimensions...")
sr = 22050
duration = 2
t = np.linspace(0, duration, int(sr * duration))
test_audio = np.sin(2 * np.pi * 500 * t) * np.exp(-3 * t)

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    sf.write(f.name, test_audio, sr)
    temp_path = f.name

features = extract_audio_features_actual(temp_path)
os.unlink(temp_path)

if features is not None:
    actual_feature_count = len(features)
    print(f"✓ Actual feature count: {actual_feature_count}")
else:
    print("✗ Feature extraction failed")
    exit(1)

# Create synthetic dataset with correct dimensions
print("\nStep 1: Creating synthetic dataset...")
X, y = create_synthetic_cough_features(n_samples=800, n_features=actual_feature_count)
print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution: Dry={np.sum(y==0)}, Wet={np.sum(y==1)}")

# Split data
print("\nStep 2: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
print("\nStep 3: Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("\nStep 4: Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# Evaluate
print("\nStep 5: Evaluating model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Dry', 'Wet']))

# Save model
print("\nStep 6: Saving model artifacts...")
model_dir = "/home/ubuntu/coughsense_project/ml_training/trained_model"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, 'cough_classifier_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'feature_scaler.pkl'))

config = {
    'target_duration': 5.0,
    'sample_rate': 22050,
    'n_mfcc': 13,
    'feature_count': actual_feature_count,
    'class_mapping': {'dry': 0, 'wet': 1},
    'label_mapping': {'0': 'dry', '1': 'wet'},
    'model_type': 'RandomForestClassifier',
    'training_samples': len(X),
    'test_accuracy': float(accuracy),
    'note': 'Model trained with correct 85-feature dimension'
}

with open(os.path.join(model_dir, 'model_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Model saved with {actual_feature_count} features")
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print(f"Accuracy: {accuracy:.2%}")
print("="*60)
