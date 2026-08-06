import re
import librosa
import numpy as np
from datasets import load_dataset, Audio

# Define a comprehensive list of valid characters for the vocabulary
# Index 0 represents the blank token required for CTC loss calculations
vocab_list = [
    "<blank>", " ", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", 
    "m", "n", "o", "p", "r", "s", "t", "u", "w", "y", "ẹ", "ọ", "ṣ", 
    "á", "à", "é", "è", "í", "ì", "ó", "ò", "ú", "ù", "ẹ́", "ẹ̀", "ọ́", "ọ̀", 
    ".", ","
]
vocab_dict = {char: index for index, char in enumerate(vocab_list)}

def get_streaming_dataset():
    # Stream the dataset directly from the repository
    dataset = load_dataset(
        "naijavoices/naijavoices-dataset", 
        "yoruba-batch-1", 
        split="train",
        streaming=True
    )
    # Cast the audio column to resample the data down to 16 kHz on the fly
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset

def extract_mel_spectrogram(batch):
    audio_array = batch["audio"]["array"]
    sampling_rate = batch["audio"]["sampling_rate"]
    
    # Extract the Mel spectrogram from the resampled audio signal
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array, 
        sr=sampling_rate, 
        n_mels=80, 
        n_fft=1024, 
        hop_length=256
    )
    
    # Convert the power spectrogram to a decibel scale for model stability
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    batch["input_features"] = mel_spec_db
    return batch

def tokenize_transcript(batch):
    transcript = batch["text"].lower()
    transcript = re.sub(r"[^a-zẹọṣáàéèíìóòúùẹ́ẹ̀ọ́ọ̀\.\,\s]", "", transcript)
    
    sequence = []
    for char in transcript:
        if char in vocab_dict:
            sequence.append(vocab_dict[char])
            
    batch["labels"] = sequence
    return batch
