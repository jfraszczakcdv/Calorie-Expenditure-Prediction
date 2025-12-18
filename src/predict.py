import torch
import pandas as pd

from src.model import NeuralNetwork
from src.workout_dataset import load_data

MODEL_PATH = "outputs/2025-12-18/12-53-52/best_model.pth"
TEST_CSV = "data/test.csv"
OUT_CSV = "submission.csv"


def main():
    # Wczytaj dane testowe (bez kolumny Calories)
    test_df = pd.read_csv(TEST_CSV)
    test_ids = test_df["id"].values

    # Użyj tego samego przetwarzania co w load_data, tylko bez targetu
    data = test_df.copy()
    data["Sex"] = data["Sex"].map({"male": 1, "female": 0}).astype(float)
    features = data.drop(columns=["id"])
    x = torch.tensor(features.values, dtype=torch.float32)

    # Model
    model = NeuralNetwork()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        preds = model(x).squeeze().numpy()

    submission = pd.DataFrame({"id": test_ids, "Calories": preds})
    submission.to_csv(OUT_CSV, index=False)
    print(f"Zapisano {OUT_CSV}")


if __name__ == "__main__":
    main()
