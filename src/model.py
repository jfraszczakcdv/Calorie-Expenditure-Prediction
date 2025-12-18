import torch
import torch.nn as nn

class CaloriePredictor(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, dropout_rate=0.0):
        super(CaloriePredictor, self).__init__()
        
        self.net = nn.Sequential(
            # Warstwa wejściowa -> Ukryta 1
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Opcjonalny Dropout
            
            # Warstwa Ukryta 1 -> Ukryta 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Opcjonalny Dropout
            
            # Warstwa wyjściowa (1 neuron - przewidywana liczba kalorii)
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)