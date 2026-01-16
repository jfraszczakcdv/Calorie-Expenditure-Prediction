import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')

# Импортируем модель из существующего файла
from src.model import NeuralNetwork

def load_model(model_path):
    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Путь к модели
    model_path = Path("outputs/2026-01-16/16-18-59/best_model.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"Loading model from: {model_path}")
    
    # Загружаем модель
    model = load_model(model_path)
    model.to(device)
    
    # Загружаем тестовые данные
    test_data = pd.read_csv("data/test.csv")
    print(f"Test data shape: {test_data.shape}")
    
    # Предобработка: кодируем Sex (male=1, female=0)
    test_data["Sex"] = test_data["Sex"].map({"male": 1, "female": 0})
    
    # Убираем колонку id
    test_features = test_data.drop(columns=["id"]).values.astype(np.float32)
    print(f"Features shape: {test_features.shape}")
    
    # Конвертируем в тензор
    test_tensor = torch.from_numpy(test_features)
    
    # Делаем предсказания
    with torch.no_grad():
        test_tensor = test_tensor.to(device)
        predictions = model(test_tensor).cpu().numpy().flatten()
    
    # Создаём submission файл
    submission = pd.DataFrame({
        "id": test_data["id"],
        "Calories": predictions
    })
    
    submission_path = "data/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")

if __name__ == "__main__":
    main()
