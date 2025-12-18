# Raport Końcowy - Predykcja Spalania Kalorii

## 1. Cel Zadania
Celem projektu było stworzenie modelu regresji przewidującego liczbę spalonych kalorii na podstawie parametrów treningu (czas trwania, tętno) oraz cech fizycznych użytkownika (wiek, waga, wzrost).

## 2. Analiza Danych (EDA)
Dane wejściowe obejmowały 7 cech: `Sex`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`.

### Wnioski z analizy:
- Najsilniejszy wpływ na zmienną docelową (`Calories`) mają: **Czas trwania (Duration)** oraz **Tętno (Heart Rate)**.
- Zmienne te wykazują niemal liniową zależność ze spalonymi kaloriami.

### Wizualizacje:
**Rozkład zmiennej docelowej (Calories):**
![Rozkład zmiennej docelowej](outputs/plots/target_dist_calories.png)

**Macierz korelacji:**
![Macierz korelacji](outputs/plots/correlation_matrix.png)

**Zależność: Duration vs Calories:**
![Duration vs Calories](outputs/plots/scatter_Duration_calories.png)

**Zależność: Heart Rate vs Calories:**
![Heart Rate vs Calories](outputs/plots/scatter_Heart_Rate_calories.png)

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

---

## 6. Część Teoretyczna (Zadanie PDF)

### Zadanie 1: Obliczenie pochodnych
Dla podanej sieci (wejście [2, 3], target 5, wagi=1.0, bias=1.0):
- **Wagi wyjściowe**: $\frac{\partial L}{\partial w_{out}} = 96$
- **Bias wyjściowy**: $\frac{\partial L}{\partial b_{out}} = 16$
- **Wagi warstwy ukrytej**:
  - Dla $x_1$ (wagi 1,1 i 2,1): $\frac{\partial L}{\partial w_{x1}} = 32$
  - Dla $x_2$ (wagi 1,2 i 2,2): $\frac{\partial L}{\partial w_{x2}} = 48$
- **Biasy ukryte**: $\frac{\partial L}{\partial b_{h}} = 16$

**Pytanie**: Czy sieć jest w stanie się uczyć, gdy wszystkie parametry zostaną zainicjalizowane z wartością 0.0?
**Odpowiedź**: **Nie**.
Przy inicjalizacji zerami:
1. Propagacja w przód daje wyjście 0.
2. Gradient dla wag wyjściowych wynosi 0 (bo aktywacja h=0).
3. Gradient nie propaguje się do niższych warstw.
4. Nawet przy inicjalizacji stałą inną niż zero, występuje problem symetrii (neurony uczą się tego samego).

### Zadanie 2: Pytania teoretyczne

**1. Kiedy używać sieci neuronowych vs programy imperatywne?**
Sieci neuronowe stosujemy, gdy:
- Dane są nieustrukturyzowane (obrazy, dźwięk, tekst).
- Zależności są zbyt złożone, by opisać je ręcznymi regułami if/else (np. rozpoznawanie kota).
- Problem wymaga generalizacji na nowe, niewidziane dane (czego sztywne reguły nie potrafią).

**2. Rola funkcji aktywacji**
Funkcje aktywacji (np. ReLU, Sigmoid) wprowadzają **nieliniowość**.
Bez nich, niezależnie od liczby warstw, sieć byłaby matematycznie równoważna pojedynczej transformacji liniowej (regresji), co uniemożliwiłoby rozwiązywanie złożonych, nieliniowych problemów.

**3. Rola Dropout'u**
Dropout to technika **regularyzacji**. Polega na losowym wyłączaniu neuronów podczas treningu.
Zapobiega to **przeuczeniu (overfitting)**, zmuszając sieć do tworzenia nadmiarowych reprezentacji i niepoleganiu na pojedynczych, silnych cechach.