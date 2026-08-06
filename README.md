# Hybrid CNN-LSTM Framework for Yoruba Automatic Speech Recognition

Official implementation accompanying the research paper: **"Benchmarking Hybrid CNN-LSTM Frameworks for Yoruba Automatic Speech Recognition"**.

## Abstract
This repository provides the complete, reproducible PyTorch source code for training and evaluating an end-to-end hybrid Convolutional Neural Network and Long Short-Term Memory (CNN-LSTM) framework on low-resource tonal African speech data using Connectionist Temporal Classification (CTC) loss. It addresses critical structural and morphological challenges inherent to low-resource tonal languages without requiring heavy pre-trained self-supervised teacher models.

## Repository Structure
* `model.py`: Defines the dual-stride convolutional and bidirectional LSTM architecture.
* `dataset.py`: Handles streaming dataset retrieval from Hugging Face and on-the-fly Mel-spectrogram extraction.
* `train.py`: Manages the training loop, gradient scaling, and CTC loss alignment.
* `benchmark_hardware.py`: Quantifies model parameters, memory footprint, and edge inference latency to verify device efficiency.
* `requirements.txt`: Lists essential library dependencies.

## Data Acquisition & Hugging Face Authentication
The dataset is streamed dynamically from the repository without requiring full local disk storage. 

1. **Dataset Source:** The pipeline pulls audio arrays and transcripts from the `naijavoices/naijavoices-dataset` under the `yoruba-batch-1` configuration[cite: 1].
2. **Hugging Face Token:** Because the dataset requires authentication, you must generate a user access token from your Hugging Face account[cite: 1]. 
3. **Configuration:** 
   * For interactive or notebook environments, save your token as a secret named `streaming-token` so the session can securely authenticate via `UserSecretsClient()`[cite: 1].
   * For local terminal usage, authenticate via the CLI using `huggingface-cli login`.

## Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/YOUR-USERNAME/yoruba-asr-cnn-lstm.git](https://github.com/YOUR-USERNAME/yoruba-asr-cnn-lstm.git)
cd yoruba-asr-cnn-lstm
pip install -r requirements.txt
