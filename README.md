# ACSR: Automated Cued Speech Recognition

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the implementation and resources for our paper: "Automated Recognition of French Cued Speech Using Joint Multimodal Learning". It presents a novel approach to decode French Cued Speech (CS) by jointly processing hand shapes, hand positions, and lip movements using deep learning techniques.

Cued Speech is a visual communication system that combines hand shapes and positions near the face with lip movements to make spoken language visually accessible, particularly for individuals with hearing impairments.

## Features

- **Multimodal Processing**: Joint analysis of hand shapes, hand positions, and lip movements
- **Three-Stream Fusion Architecture**: Specialized neural network encoders for each modality
- **Joint CTC-Attention Model**: Combined CTC and attention mechanisms for improved sequence prediction
- **Phoneme and Syllable Recognition**: Support for both phoneme-level and syllable-level decoding

## Requirements

- Python >= 3.11
- PyTorch
- Poetry (for dependency management)
- CUDA-compatible GPU (recommended)

Full dependencies are listed in the `pyproject.toml` file.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/acsr.git
   cd acsr
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Dataset

Our model is trained on a French Cued Speech dataset. Due to privacy constraints, the raw data is not publicly available, but processed features can be provided upon request.

The data pipeline expects the following structure:
- Hand shape features
- Hand position features
- Lip movement features
- Corresponding phonetic or syllabic transcriptions

## Usage

### Training

```bash
python src/train.py \
    --mode phoneme \
    --train_features_dir data/train/features \
    --train_labels_dir data/train/labels \
    --val_features_dir data/val/features \
    --val_labels_dir data/val/labels \
    --vocab_path data/vocab.txt \
    --batch_size 32 \
    --n_epochs 100 \
    --learning_rate 0.001 \
    --hidden_dim 256 \
    --n_layers 3 \
    --checkpoint_dir checkpoints
```

### Testing

```bash
python src/test.py \
    --mode phoneme \
    --test_features_dir data/test/features \
    --test_labels_dir data/test/labels \
    --vocab_path data/vocab.txt \
    --checkpoint_path checkpoints/best_model.pt
```

### Decoding

```bash
python src/decoding.py \
    --input_video path/to/video.mp4 \
    --checkpoint_path checkpoints/best_model.pt \
    --vocab_path data/vocab.txt \
    --output_file decoded_transcript.txt
```

## Model Architecture

Our system employs a three-stream fusion architecture:

1. **Hand Shape Encoder**: Processes hand configuration features
2. **Hand Position Encoder**: Analyzes spatial positioning relative to facial landmarks
3. **Lip Movement Encoder**: Extracts visual cues from lip reading

These streams are combined through a fusion mechanism before being processed by a joint CTC-Attention decoder that produces the final output sequence.

## Results

Our model achieves the following performance metrics on the test set:

| Metric | Phoneme Mode | Syllable Mode |
|--------|--------------|---------------|
| CER    | -            | xx.x%         |
| SER    | -            | xx.x%         |
| GER    | -            | xx.x%         |

## Citation

If you use this code or refer to our work in your research, please cite our paper:

```bibtex
@article{sow2024automated,
  title={Automated Recognition of French Cued Speech Using Joint Multimodal Learning},
  author={Sow, Boubacar},
  journal={},
  year={2024},
  publisher={}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Institut Pasteur](https://www.pasteur.fr/) for infrastructure support
- Funding agencies and grants that supported this research
- Open-source libraries and tools used in this implementation

## Contact

For questions or collaborations, please contact:
- Boubacar Sow - [boubasow.pro@gmail.com]
