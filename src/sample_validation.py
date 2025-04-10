import os
import random
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def sample_validation_files(train_features_dir, train_labels_dir, val_features_dir, val_labels_dir, num_samples=60):
    # Get all files from training features
    feature_files = [f for f in os.listdir(train_features_dir) if f.endswith('.csv') and "sent" not in f]
    logging.info(f"Found {len(feature_files)} feature files in training directory")
    
    # Adjust number of samples if necessary
    actual_samples = min(num_samples, len(feature_files))
    if actual_samples < num_samples:
        logging.warning(f"Requested {num_samples} samples but only {len(feature_files)} files available. Using {actual_samples} samples.")
    
    # Randomly sample files
    val_files = random.sample(feature_files, actual_samples)
    
    # Create validation directories if they don't exist
    os.makedirs(val_features_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Move sampled files to validation directories
    moved_files = 0
    for file in val_files:
        # Move feature file
        src_feature = os.path.join(train_features_dir, file)
        dst_feature = os.path.join(val_features_dir, file)
        shutil.move(src_feature, dst_feature)
        
        # Move corresponding label file
        label_file = file.replace('_features.csv', '.csv')
        src_label = os.path.join(train_labels_dir, label_file)
        dst_label = os.path.join(val_labels_dir, label_file)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
            moved_files += 1
        else:
            logging.warning(f"Label file not found for {file}: {src_label}")
    
    logging.info(f"Successfully moved {moved_files} pairs of files to validation directories")

if __name__ == '__main__':
    base_dir = '/pasteur/appa/homes/bsow/ACSR/ACSR/data'
    
    train_features_dir = os.path.join(base_dir, 'training_features')
    train_labels_dir = os.path.join(base_dir, 'train_labels')
    val_features_dir = os.path.join(base_dir, 'val_features')
    val_labels_dir = os.path.join(base_dir, 'val_labels')
    
    logging.info(f"Training features directory: {train_features_dir}")
    logging.info(f"Training labels directory: {train_labels_dir}")
    logging.info(f"Validation features directory: {val_features_dir}")
    logging.info(f"Validation labels directory: {val_labels_dir}")
    
    sample_validation_files(
        train_features_dir=train_features_dir,
        train_labels_dir=train_labels_dir,
        val_features_dir=val_features_dir,
        val_labels_dir=val_labels_dir,
        num_samples=60
    )