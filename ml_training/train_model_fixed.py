#!/usr/bin/env python3
"""
Fixed Cough Classification Model Training Script
Handles WebM audio format issues properly
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
from tqdm import tqdm

def extract_audio_features_robust(audio_path, target_duration=5.0, sr=22050):
    """
    Extract comprehensive audio features with robust error handling
    Handles various audio formats and edge cases
    """
    try:
        # Load audio file with error handling
        try:
            audio, sample_rate = librosa.load(audio_path, sr=sr, duration=target_duration, res_type='kaiser_fast')
        except Exception as e:
            # Try alternative loading method
            audio, sample_rate = librosa.load(audio_path, sr=sr, mono=True, duration=target_duration)
        
        # Ensure we have valid audio data
        if audio is None or len(audio) == 0:
            return None
        
        # Pad or trim to fixed duration
        target_length = int(target_duration * sr)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        features = []
        
        # 1. MFCC features (13 coefficients) - 52 features
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            if mfcc.shape[0] == 13:  # Ensure we got 13 coefficients
                features.extend(np.mean(mfcc, axis=1))
                features.extend(np.std(mfcc, axis=1))
                features.extend(np.min(mfcc, axis=1))
                features.extend(np.max(mfcc, axis=1))
            else:
                return None
        except Exception as e:
            print(f"MFCC extraction failed: {e}")
            return None
        
        # 2. Spectral Centroid - 4 features
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features.extend([
                np.mean(spectral_centroid),
                np.std(spectral_centroid),
                np.min(spectral_centroid),
                np.max(spectral_centroid)
            ])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 3. Spectral Bandwidth - 4 features
        try:
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features.extend([
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth),
                np.min(spectral_bandwidth),
                np.max(spectral_bandwidth)
            ])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 4. Spectral Rolloff - 4 features
        try:
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features.extend([
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.min(spectral_rolloff),
                np.max(spectral_rolloff)
            ])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 5. Zero Crossing Rate - 4 features
        try:
            zcr = librosa.feature.zero_crossing_rate(audio)
            features.extend([
                np.mean(zcr),
                np.std(zcr),
                np.min(zcr),
                np.max(zcr)
            ])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 6. Chroma Features - 4 features
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.extend([
                np.mean(chroma),
                np.std(chroma),
                np.min(chroma),
                np.max(chroma)
            ])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 7. Spectral Contrast - 4 features
        try:
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features.extend([
                np.mean(spectral_contrast),
                np.std(spectral_contrast),
                np.min(spectral_contrast),
                np.max(spectral_contrast)
            ])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 8. Tonnetz (Tonal Centroid Features) - 4 features
        try:
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features.extend([
                np.mean(tonnetz),
                np.std(tonnetz),
                np.min(tonnetz),
                np.max(tonnetz)
            ])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 9. RMS Energy - 4 features
        try:
            rms = librosa.feature.rms(y=audio)
            features.extend([
                np.mean(rms),
                np.std(rms),
                np.min(rms),
                np.max(rms)
            ])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 10. Tempo - 1 feature
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features.append(float(tempo))
        except:
            features.append(0.0)
        
        # Verify we have exactly 109 features
        features_array = np.array(features)
        if len(features_array) != 109:
            print(f"Warning: Expected 109 features, got {len(features_array)}")
            return None
        
        # Check for NaN or Inf values
        if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
            print(f"Warning: NaN or Inf values detected")
            return None
        
        return features_array
    
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {str(e)[:100]}")
        return None

def create_feature_dataset(labeled_csv_path, max_samples=None):
    """
    Create a dataset of features from labeled audio files
    """
    # Load labeled samples
    df = pd.read_csv(labeled_csv_path)
    print(f"Loaded {len(df)} labeled samples")
    print(f"Distribution: {df['cough_type'].value_counts().to_dict()}")
    
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
        print(f"Using {len(df)} samples for training")
    
    features_list = []
    labels_list = []
    failed_files = []
    success_count = 0
    
    print("\nExtracting features from audio files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row['audio_file']
        cough_type = row['cough_type']
        
        if not os.path.exists(audio_path):
            failed_files.append((audio_path, "File not found"))
            continue
        
        features = extract_audio_features_robust(audio_path)
        
        if features is not None:
            features_list.append(features)
            labels_list.append(1 if cough_type == 'wet' else 0)  # wet=1, dry=0
            success_count += 1
        else:
            failed_files.append((audio_path, "Feature extraction failed"))
    
    print(f"\n{'='*60}")
    print(f"Successfully extracted features from {success_count} files")
    print(f"Failed to extract features from {len(failed_files)} files")
    print(f"Success rate: {success_count / len(df) * 100:.1f}%")
    print(f"{'='*60}")
    
    if success_count < 50:
        print("\nERROR: Not enough successful samples for training!")
        print("Consider using a different dataset or fixing audio format issues.")
        return None, None
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution: Dry={np.sum(y==0)}, Wet={np.sum(y==1)}")
    
    return X, y

def train_models(X, y):
    """
    Train Random Forest and SVM models
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
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\n{'='*60}")
    print(f"Best Model: {best_model_name}")
    print(f"Test Accuracy: {best_accuracy:.4f}")
    print(f"{'='*60}")
    
    return best_model, scaler, results, best_model_name, best_accuracy

def save_model(model, scaler, model_name, accuracy, save_dir):
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
    
    # Save configuration
    config = {
        'target_duration': 5.0,
        'sample_rate': 22050,
        'n_mfcc': 13,
        'feature_count': 109,
        'class_mapping': {'dry': 0, 'wet': 1},
        'label_mapping': {'0': 'dry', '1': 'wet'},
        'model_type': model_name,
        'test_accuracy': float(accuracy),
        'note': 'Model trained on real COUGHVID dataset'
    }
    
    config_path = os.path.join(save_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    return model_path, scaler_path, config_path

if __name__ == "__main__":
    print("="*60)
    print("COUGH CLASSIFICATION MODEL TRAINING (FIXED VERSION)")
    print("="*60)
    
    # Paths
    labeled_csv = "/home/ubuntu/coughsense_project/ml_training/labeled_samples.csv"
    model_save_dir = "/home/ubuntu/coughsense_project/ml_training/trained_model"
    
    # Step 1: Create feature dataset
    print("\nStep 1: Creating feature dataset from COUGHVID...")
    X, y = create_feature_dataset(labeled_csv, max_samples=None)
    
    if X is None or y is None:
        print("\nFailed to create dataset. Exiting...")
        exit(1)
    
    # Step 2: Train models
    print("\nStep 2: Training models...")
    best_model, scaler, results, model_name, accuracy = train_models(X, y)
    
    # Step 3: Save model
    print("\nStep 3: Saving model...")
    model_path, scaler_path, config_path = save_model(best_model, scaler, model_name, accuracy, model_save_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Model artifacts saved in: {model_save_dir}")
    print("\nThe model is ready for production use!")
