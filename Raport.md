# Raport Końcowy - Predykcja Spalania Kalorii

## 1. Cel Zadania
Celem projektu było stworzenie modelu regresji przewidującego liczbę spalonych kalorii na podstawie parametrów treningu (czas trwania, tętno) oraz cech fizycznych użytkownika (wiek, waga, wzrost).

## 2. Analiza Danych (EDA)
Dane wejściowe obejmowały 7 cech: `Sex`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`.
Na podstawie analizy korelacji stwierdzono, że:
- Najsilniejszy wpływ na zmienną docelową (`Calories`) mają: **Czas trwania (Duration)** oraz **Tętno (Heart Rate)**.
- Zmienne te wykazują niemal liniową zależność ze spalonymi kaloriami.

## 3. Architektura Modelu
Zastosowano sieć neuronową typu **MLP (Multi-Layer Perceptron)** zaimplementowaną w bibliotece PyTorch.
Szczegóły architektury (`src/model.py`):
- **Warstwa wejściowa**: 7 neuronów (odpowiadających liczbie cech).
- **Warstwy ukryte**:
  - Pierwsza: 64 neurony (aktywacja ReLU) + Dropout.
  - Druga: 32 neurony (aktywacja ReLU) + Dropout.
- **Warstwa wyjściowa**: 1 neuron (przewidywana wartość kalorii).

Zastosowanie mechanizmu **Dropout** miało na celu zapobieganie przeuczeniu się modelu (overfitting).

## 4. Wyniki Eksperymentów
Proces uczenia przeprowadzono przez 30 epok. Na podstawie logów treningowych (`train.log`) zaobserwowano stabilny spadek funkcji błędu.

**Osiągnięte wyniki:**
- **Początkowy błąd (Epoka 1):** RMSLE: 0.17, Loss: 0.51.
- **Końcowy błąd (Epoka 21-30):** RMSLE: **0.07**, Loss: **0.07**.

Model osiągnął zbieżność (convergence) w okolicy 20. epoki, co potwierdza poprawny dobór hiperparametrów.

## 5. Wnioski
Model skutecznie nauczył się zależności w danych. Niska wartość błędu RMSLE (0.07) sugeruje wysoką dokładność predykcji. Architektura z dwiema warstwami ukrytymi okazała się wystarczająca dla tego problemu regresji.