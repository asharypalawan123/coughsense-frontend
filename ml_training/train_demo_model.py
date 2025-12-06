#!/usr/bin/env python3
"""
Create a demo trained model with synthetic but realistic data
This demonstrates the complete pipeline while addressing dataset audio format issues
"""

import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def create_synthetic_cough_features(n_samples=800, n_features=109):
    """
    Create synthetic cough features that mimic real audio features
    Based on actual characteristics of dry vs wet coughs
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    # Create DRY cough samples (label=0)
    # Dry coughs typically have:
    # - Higher zero-crossing rate (more abrupt)
    # - Lower spectral bandwidth (sharper)
    # - Higher spectral centroid (brighter)
    for i in range(int(n_samples * 0.75)):  # 75% dry
        features = np.random.randn(n_features)
        
        # Adjust specific feature ranges to mimic dry cough
        features[0:13] = np.random.normal(10, 5, 13)  # MFCC mean
        features[72] = np.random.normal(0.15, 0.03)    # ZCR mean (higher)
        features[56] = np.random.normal(2500, 400)     # Spectral centroid (higher)
        features[60] = np.random.normal(1800, 300)     # Spectral bandwidth (lower)
        
        X.append(features)
        y.append(0)  # dry
    
    # Create WET cough samples (label=1)
    # Wet coughs typically have:
    # - Lower zero-crossing rate (more continuous)
    # - Higher spectral bandwidth (more complex)
    # - Lower spectral centroid (darker)
    for i in range(int(n_samples * 0.25)):  # 25% wet
        features = np.random.randn(n_features)
        
        # Adjust specific feature ranges to mimic wet cough
        features[0:13] = np.random.normal(8, 4, 13)    # MFCC mean (lower)
        features[72] = np.random.normal(0.10, 0.02)    # ZCR mean (lower)
        features[56] = np.random.normal(2000, 350)     # Spectral centroid (lower)
        features[60] = np.random.normal(2200, 400)     # Spectral bandwidth (higher)
        
        X.append(features)
        y.append(1)  # wet
    
    return np.array(X), np.array(y)

def train_demo_model():
    """Train a demo model with synthetic data"""
    
    print("="*60)
    print("DEMO COUGH CLASSIFICATION MODEL TRAINING")
    print("(Using synthetic data due to dataset audio format issues)")
    print("="*60)
    
    # Create synthetic dataset
    print("\nStep 1: Creating synthetic feature dataset...")
    X, y = create_synthetic_cough_features(n_samples=800)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution: Dry={np.sum(y==0)}, Wet={np.sum(y==1)}")
    
    # Split data
    print("\nStep 2: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
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
    print("✓ Model trained successfully")
    
    # Evaluate
    print("\nStep 5: Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Dry', 'Wet']))
    
    # Save model
    print("\nStep 6: Saving model artifacts...")
    model_dir = "/home/ubuntu/coughsense_project/ml_training/trained_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'cough_classifier_model.pkl')
    joblib.dump(model, model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to: {scaler_path}")
    
    # Save configuration
    config = {
        'target_duration': 5.0,
        'sample_rate': 22050,
        'n_mfcc': 13,
        'feature_count': 109,
        'class_mapping': {'dry': 0, 'wet': 1},
        'label_mapping': {0: 'dry', 1: 'wet'},
        'model_type': 'RandomForestClassifier',
        'training_samples': len(X),
        'test_accuracy': float(accuracy),
        'note': 'Demo model trained with synthetic data for integration testing'
    }
    
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Configuration saved to: {config_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel artifacts saved in: {model_dir}")
    print(f"Test Accuracy: {accuracy:.2%}")
    print("\nNOTE: This is a demo model for integration testing.")
    print("In production, train with actual labeled cough audio data.")
    
    return model, scaler, config

if __name__ == "__main__":
    train_demo_model()
