import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Wczytanie danych
df = pd.read_csv('data/train.csv')

# Zamiana płci na liczby dla korelacji
df['Sex_Code'] = df['Sex'].map({'male': 0, 'female': 1})

# 1. Macierz korelacji
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Macierz Korelacji")
plt.savefig('outputs/correlation_matrix.png')
print("Zapisano macierz korelacji.")

# 2. Zależność Kalorii od Czasu Trwania
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Duration', y='Calories', hue='Sex', data=df, alpha=0.6)
plt.title("Spalone Kalorie vs Czas Trwania")
plt.savefig('outputs/calories_vs_duration.png')
print("Zapisano wykres punktowy.")