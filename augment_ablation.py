import torch
import numpy as np

def apply_spec_augmentation(spec, freq_mask_param=15, time_mask_param=35):
    """
    Applies frequency and time masking to Mel-spectrograms 
    to simulate acoustic variations and prevent overfitting.
    """
    tensor_spec = torch.tensor(spec, dtype=torch.float32)
    num_mel_channels, time_steps = tensor_spec.shape
    
    # Frequency Masking
    f = int(torch.randint(0, freq_mask_param, (1,)).item())
    f0 = int(torch.randint(0, max(1, num_mel_channels - f), (1,)).item())
    tensor_spec[f0:f0 + f, :] = 0
    
    # Time Masking
    t = int(torch.randint(0, time_mask_param, (1,)).item())
    t0 = int(torch.randint(0, max(1, time_steps - t), (1,)).item())
    tensor_spec[:, t0:t0 + t] = 0
    
    return tensor_spec.numpy()

if __name__ == "__main__":
    dummy_spec = np.random.randn(80, 237)
    augmented_spec = apply_spec_augmentation(dummy_spec)
    print("=== Data Augmentation Ablation Test ===")
    print(f"Original Shape: {dummy_spec.shape}")
    print(f"Augmented Shape: {augmented_spec.shape}")
    print("SpecAugment applied successfully.")
