#!/usr/bin/env python3
"""
COUGHVID Dataset Preparation Script
Extract and prepare labeled cough samples for ML training
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_labeled_samples(dataset_dir):
    """
    Extract all samples with expert labels from the COUGHVID dataset
    """
    dataset_path = Path(dataset_dir)
    labeled_data = []
    
    # Get all JSON files
    json_files = list(dataset_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in total")
    
    # Extract files with expert labels
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if expert labels exist
            expert_label_key = None
            for key in data.keys():
                if key.startswith('expert_labels'):
                    expert_label_key = key
                    break
            
            if expert_label_key and 'cough_type' in data[expert_label_key]:
                expert_labels = data[expert_label_key]
                cough_type = expert_labels.get('cough_type')
                
                # Only include dry or wet (exclude 'cant_tell' or other values)
                if cough_type in ['dry', 'wet']:
                    # Get the corresponding audio file
                    audio_file = json_file.with_suffix('.webm')
                    
                    if audio_file.exists():
                        labeled_data.append({
                            'uuid': json_file.stem,
                            'audio_file': str(audio_file),
                            'cough_type': cough_type,
                            'quality': expert_labels.get('quality', 'unknown'),
                            'severity': expert_labels.get('severity', 'unknown'),
                            'diagnosis': expert_labels.get('diagnosis', 'unknown'),
                            'dyspnea': expert_labels.get('dyspnea', 'False'),
                            'wheezing': expert_labels.get('wheezing', 'False'),
                            'congestion': expert_labels.get('congestion', 'False'),
                            'age': data.get('age', ''),
                            'gender': data.get('gender', ''),
                            'status': data.get('status', ''),
                            'cough_detected': data.get('cough_detected', '')
                        })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(labeled_data)
    print(f"\n{'='*60}")
    print(f"Labeled samples extracted: {len(df)}")
    print(f"\nCough type distribution:")
    print(df['cough_type'].value_counts())
    print(f"\nQuality distribution:")
    print(df['quality'].value_counts())
    print(f"{'='*60}\n")
    
    return df

if __name__ == "__main__":
    # Set paths
    dataset_dir = "/home/ubuntu/coughsense_project/ml_training/public_dataset"
    output_file = "/home/ubuntu/coughsense_project/ml_training/labeled_samples.csv"
    
    # Extract labeled samples
    df = extract_labeled_samples(dataset_dir)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Labeled samples saved to: {output_file}")
    
    # Display sample
    print("\nSample data:")
    print(df.head())
