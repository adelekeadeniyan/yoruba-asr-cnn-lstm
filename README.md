# Hybrid CNN-LSTM Framework for Yoruba Automatic Speech Recognition

Official implementation accompanying the research paper: **"Benchmarking Hybrid CNN-LSTM Frameworks for Yoruba Automatic Speech Recognition"**.

## Abstract
This repository provides the complete, reproducible PyTorch source code for training and evaluating an end-to-end hybrid Convolutional Neural Network and Long Short-Term Memory (CNN-LSTM) framework on low-resource tonal African speech data using Connectionist Temporal Classification (CTC) loss. It addresses critical structural and morphological challenges inherent to low-resource tonal languages without requiring heavy pre-trained self-supervised teacher models.

## Repository Structure
* `model.py`: Defines the dual-stride convolutional and bidirectional LSTM architecture.
* `dataset.py`: Handles streaming dataset retrieval from Hugging Face and on-the-fly Mel-spectrogram extraction.
* `train.py`: Manages the training loop, gradient scaling, and CTC loss alignment.
* `benchmark_hardware.py`: Quantifies model parameters, memory footprint, and edge inference latency.
* `evaluate_tones.py`: Implements the targeted tonal error analysis.
* `augment_ablation.py`: Executes the feature-level perturbation and SpecAugment ablation study.
* `qualitative_analysis.py`: Provides alignment comparisons for qualitative transcription evaluation.

## Research & Novelty Implementation
This repository contains the full implementation for the experiments detailed in Section 4 (*Results and Findings*) of the manuscript:

| LaTeX Section | Purpose | Associated Script |
| :--- | :--- | :--- |
| **4.4 Hardware Efficiency** | Measures edge deployment suitability | `benchmark_hardware.py` |
| **4.5 Tonal Error Analysis** | Evaluates recognition of Yorùbá diacritics | `evaluate_tones.py` |
| **4.6 Ablation Study** | Tests robustness of feature masking | `augment_ablation.py` |
| **4.7 Qualitative Analysis** | Logs alignment and decoding accuracy | `qualitative_analysis.py` |

## Data Acquisition & Authentication
The dataset is streamed dynamically from the Hugging Face repository (`naijavoices/naijavoices-dataset`).

1. **Hugging Face Token:** Generate a user access token from your Hugging Face account settings.
2. **Configuration:** 
   * **Notebook Environments:** Save your token as a secret named `streaming-token`. The system uses `UserSecretsClient()` to authenticate the session automatically.
   * **Local Environment:** Run `huggingface-cli login` in your terminal to authenticate your session globally.

## Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/adelekeadeniyan/yoruba-asr-cnn-lstm.git](https://github.com/adelekeadeniyan/yoruba-asr-cnn-lstm.git)
cd yoruba-asr-cnn-lstm
pip install -r requirements.txt
