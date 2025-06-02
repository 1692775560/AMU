# AMU: Melanoma Immunotherapy Response Prediction

![Melanoma Prediction](https://img.shields.io/badge/Deep%20Learning-Melanoma%20Prediction-blue)
![PaddlePaddle](https://img.shields.io/badge/Framework-PaddlePaddle%202.3.2-brightgreen)
![Status](https://img.shields.io/badge/Status-Research-yellow)

## Overview

This repository contains the implementation of Attention-based mRNA Transformer (AMU), a novel deep learning architecture that accurately predicts melanoma response to immune checkpoint inhibitor therapy using mRNA expression data. AMU combines a transformer encoder with a convolutional network to effectively model gene interactions and extract predictive signatures from transcriptomic profiles.

## Key Features

- **Novel Architecture**: Transformer-based approach for gene expression data analysis
- **Accurate Prediction**: State-of-the-art performance in predicting immunotherapy response
- **Interpretable Results**: Model interpretation through SHAP values and gene embedding visualization
- **Batch Effect Correction**: Robust pipeline for handling multi-source data
- **Comprehensive Evaluation**: Comparison with 5 other machine learning approaches

## Repository Structure

```
├── data/
│   ├── raw/                     # Original dataset files (four.csv, logfourupsample.csv)
│   ├── processed/               # Processed and batch-corrected data
│   └── results/                 # Model prediction results
├── models/
│   ├── amu.py                   # AMU model architecture
│   └── baseline_models.py       # Implementation of comparison models
├── scripts/
│   ├── preprocess.py            # Data preprocessing pipeline
│   ├── evaluate_amu.py          # AMU model evaluation
│   ├── evaluate_multiple_models.py  # Comparative model evaluation
│   └── generate_figures.py      # Script for generating publication figures
├── figures/                     # Generated figures
├── notebooks/                   # Jupyter notebooks for analysis
├── requirements.txt             # Package dependencies
└── README.md                    # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/username/AMU-melanoma-prediction.git
cd AMU-melanoma-prediction

# Create a conda environment (recommended)
conda create -n amu_env python=3.8
conda activate amu_env

# Install dependencies
pip install -r requirements.txt
```

## Key Dependencies

- PaddlePaddle 2.3.2
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn
- SciPy

## Dataset

The study used four datasets with mRNA expression profiles from melanoma patients treated with immune checkpoint inhibitors:

1. GSE78220 (Riaz et al.)
2. GSE91061 (Hugo et al.)
3. GSE165278 (Liu et al.)
4. Independent dataset from Liu et al. (PMID:31792460)

We selected 160 genes based on their relevance to melanoma biology and immune response. The data preprocessing pipeline includes:

1. Quality control
2. Normalization
3. Batch effect correction using ComBat algorithm
4. Feature selection
5. Final normalization and integration

## Usage

### Data Preprocessing

```bash
# Preprocess raw RNA-seq data
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed

# Apply batch effect correction
python scripts/preprocess.py --correct_batch --input_dir data/processed --output_dir data/processed/batch_corrected
```

### Model Training and Evaluation

```bash
# Train and evaluate AMU model
python scripts/evaluate_amu.py --data_path data/processed/batch_corrected_data.csv --output_dir results/amu

# Compare multiple models
python scripts/evaluate_multiple_models.py --data_path data/processed/batch_corrected_data.csv --output_dir results/comparison
```

### Generate Figures

```bash
# Generate all figures for publication
python scripts/generate_figures.py --results_dir results/ --output_dir figures/
```

## Model Architecture

AMU consists of two main components:

1. **Transformer Encoder**:
   - 20-dimensional gene embedding layer
   - 8 multi-head attention mechanisms
   - 8 repeated transformer encoder layers

2. **Convolutional Network**:
   - Sequential architecture: Convolution → Dropout → Batch Normalization → ReLU → Adaptive Max Pooling
   - Final SoftMax activation with cross-entropy loss

The model contains 83,462 trainable parameters and uses the Adam optimizer with a two-step learning rate decay schedule.

## Results

AMU outperformed five other machine learning models (SVM, XGBoost, Random Forest, MLP, CNN) in predicting melanoma response to immunotherapy:

- Validation set: AUC of 0.953, mAP of 0.972
- Testing set: AUC of 0.672, mAP of 0.800

Model interpretation revealed the importance of the TNF-TNFRSF1A axis and lymphocyte proliferation pathway in determining treatment response.

## Citation

If you use this code or find our work useful for your research, please cite our paper:

```
@article{yin2023attention,
  title={Attention-based mRNA Transformer Accurately Predicts Melanoma Immune Checkpoint Inhibitor Response},
  author={Yin, Yi and Zhang, Tao and Wang, Ziming and Li, Dong},
  journal={[Journal Name]},
  year={2023},
  volume={},
  number={},
  pages={}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

We thank the Baidu PaddlePaddle team for providing free online GPU resources and training courses. We also acknowledge the authors of the original datasets used in this study.

## Contact

For questions or collaborations, please open an issue on this repository or contact the corresponding author at [email@institution.edu].
