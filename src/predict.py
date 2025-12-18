import pandas as pd
import numpy as np
import torch
import joblib
from src.model import CaloriePredictor

def predict():
    # 1. Wczytaj model i scaler
    model = CaloriePredictor(input_dim=7, hidden_dim=64, dropout_rate=0.0) # Parametry jak w treningu
    model.load_state_dict(torch.load('outputs/best_model.pth'))
    model.eval()
    
    scaler = joblib.load('outputs/scaler.pkl')
    
    # 2. Wczytaj dane testowe
    test_df = pd.read_csv('data/test.csv')
    ids = test_df['id'] # Zachowaj ID do pliku wynikowego
    
    # 3. Przetwórz dane tak samo jak treningowe
    test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
    features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex']
    X_test = test_df[features].values.astype(np.float32)
    
    # 4. Skalowanie
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Predykcja
    with torch.no_grad():
        inputs = torch.tensor(X_test_scaled)
        predictions = model(inputs).numpy().flatten()
    
    # 6. Zapisz wynik
    submission = pd.DataFrame({'id': ids, 'Calories': predictions})
    submission.to_csv('outputs/submission.csv', index=False)
    print("Wygenerowano plik outputs/submission.csv gotowy do wysłania!")

if __name__ == "__main__":
    predict()