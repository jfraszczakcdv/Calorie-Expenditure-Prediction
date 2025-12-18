import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.model import CaloriePredictor

# --- KONFIGURACJA ---
SEED = 42
EPOCHS = 100
BATCH_SIZE = 32

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_and_process_data(filepath):
    df = pd.read_csv(filepath)
    # Kodowanie płci: male=0, female=1
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex']
    X = df[features].values.astype(np.float32)
    y = df['Calories'].values.astype(np.float32).reshape(-1, 1)
    
    return X, y

def train_experiment(learning_rate, dropout_rate, hidden_dim, X_train, X_val, y_train, y_val):
    set_seed(SEED) # Reprodukowalność
    
    model = CaloriePredictor(input_dim=7, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    criterion = nn.MSELoss() # Błąd średniokwadratowy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Konwersja na tensory
    X_t = torch.tensor(X_train)
    y_t = torch.tensor(y_train)
    X_v = torch.tensor(X_val)
    y_v = torch.tensor(y_val)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_t)
        loss = criterion(output, y_t)
        loss.backward()
        optimizer.step()
        
        # Walidacja
        model.eval()
        with torch.no_grad():
            val_out = model(X_v)
            val_loss = criterion(val_out, y_v)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Zapisz model tymczasowo
            torch.save(model.state_dict(), f'outputs/model_h{hidden_dim}_lr{learning_rate}_d{dropout_rate}.pth')

    return best_val_loss.item()

# --- GŁÓWNA PĘTLA ---
if __name__ == "__main__":
    X, y = load_and_process_data('data/train.csv')
    
    # Skalowanie danych (bardzo ważne dla sieci neuronowych!)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Podział: 80% trening, 20% walidacja
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    print(f"Dane podzielone: Trening={len(X_train)}, Walidacja={len(X_val)}")
    
    # Eksperymenty
    lrs = [0.01, 0.001]
    dropouts = [0.0, 0.2]
    hidden_dims = [32, 64, 128] # Różne architektury (rozmiary warstwy ukrytej)
    
    print("\n--- WYNIKI EKSPERYMENTÓW (MSE Loss na zbiorze walidacyjnym) ---")
    print(f"{'Hidden':<10} | {'LR':<10} | {'Dropout':<10} | {'Val Loss':<10}")
    print("-" * 50)
    
    best_loss = float('inf')
    best_config = None

    for hidden in hidden_dims:
        for lr in lrs:
            for drop in dropouts:
                loss = train_experiment(lr, drop, hidden, X_train, X_val, y_train, y_val)
                print(f"{hidden:<10} | {lr:<10} | {drop:<10} | {loss:.4f}")
                
                if loss < best_loss:
                    best_loss = loss
                    best_config = (hidden, lr, drop)

    print(f"\nNajlepsza konfiguracja: Hidden={best_config[0]}, LR={best_config[1]}, Dropout={best_config[2]} (Loss: {best_loss:.4f})")
    
    # Kopiujemy najlepszy model do best_model.pth
    import shutil
    import os
    best_model_run = f'outputs/model_h{best_config[0]}_lr{best_config[1]}_d{best_config[2]}.pth'
    if os.path.exists(best_model_run):
        shutil.copy(best_model_run, 'outputs/best_model.pth')
        print("Zapisano najlepszy model do outputs/best_model.pth")

    # Zapisujemy scaler, żeby użyć go przy predykcji
    import joblib
    joblib.dump(scaler, 'outputs/scaler.pkl')