# Odpowiedzi teoretyczne

## 1. Pochodne parametrów (w, b) dla architektury:
- Wejście: [x1, x2] = [2, 3]
- Warstwa ukryta: 2 neurony, ReLU
- Wyjście: 1 neuron (bez funkcji aktywacji)
- Funkcja straty: MSE
- Wartość docelowa: y = 5

**Kroki:**
1. Przebieg do przodu:
   z1 = w11*x1 + w12*x2 + b1
   a1 = ReLU(z1)
   z2 = w21*x1 + w22*x2 + b2
   a2 = ReLU(z2)
   y_pred = w3_1*a1 + w3_2*a2 + b3

2. MSE loss: L = (y_pred - y)^2

3. Wsteczna propagacja (gradienty):
   dL/dy_pred = 2*(y_pred - y)
   dL/dw3_1 = dL/dy_pred * a1
   dL/dw3_2 = dL/dy_pred * a2
   dL/db3 = dL/dy_pred
   ... i analogicznie dla wag warstwy ukrytej.

## 2. Inicjalizacja zerami:
Jeśli wszystkie parametry = 0.0:
- Wszystkie aktywacje ReLU będą zerowe (jeśli bias=0).
- Gradienty będą zerowe → uczenie niemożliwe.

Jeśli wszystkie parametry = 1.0:
- Aktywacje będą niezerowe.
- Gradienty będą niezerowe → uczenie możliwe.

## 3. Do jakich zadań nadają się sieci neuronowe?
Sieci neuronowe nadają się do zadań ze złożonymi nieliniowymi zależnościami:
- Klasyfikacja obrazów, NLP, systemy rekomendacyjne.
Dlaczego nie można napisać programu składającego się z samych if'ów:
- Liczba reguł rośnie wykładniczo ze wzrostem złożoności danych.
- Niemożliwe jest ręczne uchwycenie wszystkich zależności w danych (np. piksele w obrazie).

## 4. Rola funkcji aktywacji:
Bez funkcji aktywacji wielowarstwowa sieć degeneruje się do modelu liniowego (złożenie operacji liniowych = jedna operacja liniowa).
ReLU, sigmoid, tanh wprowadzają nieliniowość, umożliwiając modelowanie złożonych funkcji.

## 5. Rola dropout:
Dropout — metoda regularyzacji:
- Podczas treningu losowo "wyłącza" część neuronów.
- Zapobiega przeuczeniu, zmuszając sieć do większej odporności.
- Poprawia generalizację.
