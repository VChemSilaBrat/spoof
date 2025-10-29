# ASVspoof 2019 Voice Anti-Spoofing with LightCNN

Deep learning system for voice anti-spoofing detection using LightCNN architecture on the ASVspoof 2019 Logical Access (LA) dataset.

## ⚠️ Project Status

This is a research implementation of LightCNN for the ASVspoof 2019 challenge. The code is fully functional and ready for training and evaluation.

## 📋 Overview

This project implements a countermeasure (CM) system for detecting spoofed audio samples using the LightCNN architecture with Max-Feature-Map (MFM) activation layers. The system is trained and evaluated on the ASVspoof 2019 LA dataset to distinguish between bonafide (genuine) and spoofed voice recordings.

### Key Features

- **LightCNN Architecture**: Lightweight CNN with MFM layers for efficient feature learning
- **STFT Features**: Short-Time Fourier Transform for audio preprocessing
- **EER Metric**: Equal Error Rate as the primary evaluation metric
- **Experiment Tracking**: Integration with Comet.ml for monitoring training progress
- **Modular Design**: Clean, maintainable code structure with separate modules

## 🏗️ Architecture

### Model Details

- **Input**: STFT spectrograms (1 x 863 x 600)
- **Layers**: 4 convolutional blocks with MFM activation
- **Regularization**: Batch normalization + 0.75 dropout
- **Output**: Binary classification (bonafide vs spoof)

### STFT Parameters

- FFT size: 1724
- Hop length: 172
- Window length: 1724
- Sampling rate: 16 kHz

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/VChemSilaBrat/spoof
cd LightCNN_ASVspoof2019

# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Download the ASVspoof 2019 LA dataset:

```bash
# Download from https://datashare.ed.ac.uk/handle/10283/3336
# Extract to 'LA' directory in project root
```

Expected structure:

```
LA/
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
├── ASVspoof2019_LA_eval/flac/
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

### Comet.ml Setup

Set up experiment tracking:

```bash
# Set environment variables
export COMET_API_KEY="your_api_key"
export COMET_WORKSPACE="your_workspace"
```

### Training

```bash
python3 train.py
```

### Evaluation

```bash
python3 evaluate.py
```

## 📊 Results

The model achieves competitive performance on the ASVspoof 2019 LA dataset. Training metrics are logged to Comet.ml for visualization and analysis.

### 🔗 Experiment Links

- **Training Experiment**: [View on Comet.ml](https://www.comet.com/api/experiment/redirect?experimentKey=a4c2a80fe21d4eddaad5fb31a7f09843)
- **Trained Model**: [Download from Comet.ml](https://www.comet.com/api/experiment/redirect?experimentKey=7608afaa9cf04ee4ab77b6d5c948d6f3) (Go to Assets tab)

## 📁 Project Structure

```
.
├── src/
│   ├── model/              # Model architectures
│   │   └── lightcnn_original.py
│   ├── datasets/           # Data loading and preprocessing
│   │   ├── asvspoof_dataset.py
│   │   ├── data_utils.py
│   │   └── dataloader_utils.py
│   ├── trainer/            # Training and evaluation
│   │   ├── asvspoof_trainer.py
│   │   └── evaluator.py
│   ├── metrics/            # Evaluation metrics
│   │   └── eer_utils.py
│   ├── logger/             # Experiment logging
│   │   └── cometml.py
│   └── utils/              # Utility functions
│       └── __init__.py
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── config.yaml             # Configuration file
└── requirements.txt        # Dependencies

```

## 🔧 Configuration

Edit `config.yaml` to customize:

- Model hyperparameters
- Training settings
- Dataset paths
- Logging options

## 📝 Citation

If you use this code, please cite:

```bibtex
@inproceedings{asvspoof2019,
  title={ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection},
  author={...},
  booktitle={Proc. Interspeech},
  year={2019}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.
