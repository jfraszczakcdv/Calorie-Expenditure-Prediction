# Raport z analizy danych i realizacji projektu

## 1. Dane
- Zbiór treningowy: 750 000 wierszy, 9 kolumn (7 cech + id + target).
- Zbiór testowy: 250 000 wierszy, 8 kolumn (brak target).
- Zmienna docelowa: Calories (kalorie).

## 2. Przetworzenie danych
- Kolumna `Sex` przekształcona na binarną (male=1, female=0).
- Brak brakujących wartości.
- Wszystkie cechy numeryczne poza id.

## 3. Wizualizacje
- Rozkład Calories zbliżony do normalnego.
- Najwyższa korelacja z `Duration`, `Heart_Rate`, `Body_Temp`.

## 4. Model
- Architektura: 7 → 8 → 4 → 1 (aktywacja ReLU).
- Trening: 21 epok, RMSLE spadła z 1.76 do 0.09.
- Optymalizator: Adam, funkcja straty: MSE.

## 5. Wyniki
- Wygenerowano predykcje dla 250 000 wierszy testowych.
- Plik `submission.csv` przesłany na Kaggle.

## 6. Dalsze kroki
- Porównanie architektur (więcej warstw, dropout).
- Strojenie hiperparametrów (learning rate, batch size).
- Walidacja krzyżowa.
