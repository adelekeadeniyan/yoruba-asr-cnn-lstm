# Hybrid CNN-LSTM Framework for Yoruba Automatic Speech Recognition

Official implementation accompanying the research paper: **"Benchmarking Hybrid CNN-LSTM Frameworks for Yoruba Automatic Speech Recognition"**.

## Abstract
This repository provides the complete, reproducible PyTorch source code for training and evaluating an end-to-end hybrid Convolutional Neural Network and Long Short-Term Memory (CNN-LSTM) framework on low-resource tonal African speech data using Connectionist Temporal Classification (CTC) loss. It addresses critical structural and morphological challenges inherent to low-resource tonal languages without requiring heavy pre-trained self-supervised teacher models.

## Repository Structure
* `model.py`: Defines the dual-stride convolutional and bidirectional LSTM architecture.
* `dataset.py`: Handles streaming dataset retrieval from Hugging Face and on-the-fly Mel-spectrogram extraction.
* `train.py`: Manages the training loop, gradient scaling, and CTC loss alignment.
* `benchmark_hardware.py`: Quantifies model parameters, memory footprint, and edge inference latency to verify device efficiency.

## Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/YOUR-USERNAME/yoruba-asr-cnn-lstm.git](https://github.com/YOUR-USERNAME/yoruba-asr-cnn-lstm.git)
cd yoruba-asr-cnn-lstm
pip install -r requirements.txt
