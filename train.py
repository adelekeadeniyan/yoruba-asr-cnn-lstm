import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import HybridASRModel
from dataset import get_streaming_dataset, extract_mel_spectrogram, tokenize_transcript

def collate_audio_batch(batch):
    features = [torch.tensor(item["input_features"], dtype=torch.float32).transpose(0, 1) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
    
    input_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True).transpose(1, 2)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        "input_features": padded_features,
        "labels": padded_labels,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = get_streaming_dataset()
    processed_dataset = dataset.map(extract_mel_spectrogram).map(tokenize_transcript)
    
    test_loader = DataLoader(
        processed_dataset,
        batch_size=4,
        collate_fn=collate_audio_batch,
        num_workers=0 
    )
    
    model = HybridASRModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    model.train()
    for batch_idx, batch in enumerate(test_loader):
        optimizer.zero_grad()
        
        features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        
        log_probs = model(features)
        log_probs = log_probs.transpose(0, 1)
        
        input_lengths = batch["input_lengths"] // 4
        target_lengths = batch["target_lengths"]
        
        labels = torch.where(labels == -100, torch.tensor(0, device=device), labels)
        
        loss = criterion(log_probs, labels, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        
        print(f"Batch {batch_idx} processed. Current Loss: {loss.item():.4f}")
        
        if batch_idx == 2:
            break

if __name__ == "__main__":
    main()
