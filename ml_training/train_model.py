#!/usr/bin/env python3
"""
Cough Classification Model Training Script
Extract audio features and train a machine learning model for dry/wet cough classification
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
from tqdm import tqdm

def extract_audio_features(audio_path, target_duration=5.0, sr=22050):
    """
    Extract comprehensive audio features from a cough recording
    
    Features extracted:
    - MFCC (13 coefficients with statistics)
    - Spectral Centroid
    - Spectral Bandwidth
    - Spectral Rolloff
    - Zero Crossing Rate
    - Chroma Features
    - Spectral Contrast
    - Tonnetz
    - RMS Energy
    """
    try:
        # Load audio file with ffmpeg backend for WebM support
        # res_type='kaiser_fast' for faster loading
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
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_min = np.min(mfcc, axis=1)
        mfcc_max = np.max(mfcc, axis=1)
        features.extend(mfcc_mean)
        features.extend(mfcc_std)
        features.extend(mfcc_min)
        features.extend(mfcc_max)
        
        # 2. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(np.mean(spectral_centroid))
        features.append(np.std(spectral_centroid))
        features.append(np.min(spectral_centroid))
        features.append(np.max(spectral_centroid))
        
        # 3. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features.append(np.mean(spectral_bandwidth))
        features.append(np.std(spectral_bandwidth))
        features.append(np.min(spectral_bandwidth))
        features.append(np.max(spectral_bandwidth))
        
        # 4. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(np.mean(spectral_rolloff))
        features.append(np.std(spectral_rolloff))
        features.append(np.min(spectral_rolloff))
        features.append(np.max(spectral_rolloff))
        
        # 5. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        features.append(np.min(zcr))
        features.append(np.max(zcr))
        
        # 6. Chroma Features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.append(np.mean(chroma))
        features.append(np.std(chroma))
        features.append(np.min(chroma))
        features.append(np.max(chroma))
        
        # 7. Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features.append(np.mean(spectral_contrast))
        features.append(np.std(spectral_contrast))
        features.append(np.min(spectral_contrast))
        features.append(np.max(spectral_contrast))
        
        # 8. Tonnetz (Tonal Centroid Features)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        features.append(np.mean(tonnetz))
        features.append(np.std(tonnetz))
        features.append(np.min(tonnetz))
        features.append(np.max(tonnetz))
        
        # 9. RMS Energy
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        features.append(np.min(rms))
        features.append(np.max(rms))
        
        # 10. Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features.append(tempo)
        
        return np.array(features)
    
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def create_feature_dataset(labeled_csv_path):
    """
    Create a dataset of features from labeled audio files
    """
    # Load labeled samples
    df = pd.read_csv(labeled_csv_path)
    print(f"Loaded {len(df)} labeled samples")
    print(f"Distribution: {df['cough_type'].value_counts().to_dict()}")
    
    features_list = []
    labels_list = []
    failed_files = []
    
    print("\nExtracting features from audio files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row['audio_file']
        cough_type = row['cough_type']
        
        if not os.path.exists(audio_path):
            failed_files.append(audio_path)
            continue
        
        features = extract_audio_features(audio_path)
        
        if features is not None:
            features_list.append(features)
            labels_list.append(1 if cough_type == 'wet' else 0)  # wet=1, dry=0
        else:
            failed_files.append(audio_path)
    
    print(f"\nSuccessfully extracted features from {len(features_list)} files")
    print(f"Failed to extract features from {len(failed_files)} files")
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution: Dry={np.sum(y==0)}, Wet={np.sum(y==1)}")
    
    return X, y

def train_models(X, y):
    """
    Train multiple classification models and select the best one
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Predictions on test set
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Set Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Dry', 'Wet']))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\n{'='*60}")
    print(f"Best Model: {best_model_name}")
    print(f"Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"{'='*60}")
    
    return best_model, scaler, results

def save_model(model, scaler, save_dir):
    """
    Save the trained model and scaler
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'cough_classifier_model.pkl')
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(save_dir, 'feature_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save feature extraction configuration
    config = {
        'target_duration': 5.0,
        'sample_rate': 22050,
        'n_mfcc': 13,
        'feature_count': 109,  # Total number of features
        'class_mapping': {'dry': 0, 'wet': 1},
        'label_mapping': {0: 'dry', 1: 'wet'}
    }
    
    config_path = os.path.join(save_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    return model_path, scaler_path, config_path

if __name__ == "__main__":
    print("="*60)
    print("COUGH CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    # Paths
    labeled_csv = "/home/ubuntu/coughsense_project/ml_training/labeled_samples.csv"
    model_save_dir = "/home/ubuntu/coughsense_project/ml_training/trained_model"
    
    # Step 1: Create feature dataset
    print("\nStep 1: Creating feature dataset...")
    X, y = create_feature_dataset(labeled_csv)
    
    # Step 2: Train models
    print("\nStep 2: Training models...")
    best_model, scaler, results = train_models(X, y)
    
    # Step 3: Save model
    print("\nStep 3: Saving model...")
    model_path, scaler_path, config_path = save_model(best_model, scaler, model_save_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel artifacts saved in: {model_save_dir}")
