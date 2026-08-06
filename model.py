import torch
import torch.nn as nn

class HybridASRModel(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, vocab_size=46):
        super(HybridASRModel, self).__init__()
        
        # Convolutional layers to extract acoustic features and downsample time
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Recurrent layers for context modeling across the time sequence
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.1
        )
        
        # Projection layer to map hidden states to character probabilities
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = self.cnn(x)
        
        # Restore the time-first orientation required by PyTorch recurrent layers
        x = x.transpose(1, 2) 
        x, _ = self.lstm(x)
        x = self.classifier(x)
        
        return nn.functional.log_softmax(x, dim=-1)
