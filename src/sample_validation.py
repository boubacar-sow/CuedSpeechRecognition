import os
import random
import shutil

def move_validation_samples():
    # Define paths
    training_features = "/pasteur/appa/homes/bsow/ACSR_github/data/training_features"
    train_labels = "/pasteur/appa/homes/bsow/ACSR_github/data/train_labels"
    val_features = "/pasteur/appa/homes/bsow/ACSR_github/data/val_features"
    val_labels = "/pasteur/appa/homes/bsow/ACSR_github/data/val_labels"

    # Create validation directories if needed
    os.makedirs(val_features, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)

    # Get all feature files
    feature_files = [f for f in os.listdir(training_features) 
                    if f.endswith("_features.csv") and 'sent' not in f]
    
    # Randomly select 60 files
    selected = random.sample(feature_files, 60)

    # Move files and corresponding labels
    for fname in selected:
        # Move feature file
        src_feature = os.path.join(training_features, fname)
        dst_feature = os.path.join(val_features, fname)
        shutil.move(src_feature, dst_feature)
        
        # Process label file
        base_name = fname.replace("_features.csv", "")
        label_name = f"{base_name}.csv"
        src_label = os.path.join(train_labels, label_name)
        dst_label = os.path.join(val_labels, label_name)
        
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
        else:
            print(f"Warning: Label {label_name} not found for {fname}")

    print(f"Successfully moved {len(selected)} samples to validation set")

if __name__ == "__main__":
    move_validation_samples()