import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

def basic_info(train, test):
    print("=== TRAIN ===")
    print(f"Shape: {train.shape}")
    print(f"Columns: {train.columns.tolist()}")
    print("\n=== TEST ===")
    print(f"Shape: {test.shape}")
    print(f"Columns: {test.columns.tolist()}")
    print("\nTrain dtypes:")
    print(train.dtypes)

def target_distribution(train):
    if 'Calories' in train.columns:  # целевая переменная называется 'Calories'
        plt.figure(figsize=(10, 6))
        train['Calories'].hist(bins=50)
        plt.title('Distribution of target (Calories)')
        plt.xlabel('Calories')
        plt.ylabel('Frequency')
        plt.savefig('data/target_distribution.png')
        print("Saved: data/target_distribution.png")

def correlation_analysis(train):
    # Оставляем только числовые колонки для корреляции
    numeric_cols = train.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        corr = train[numeric_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation matrix (numeric only)')
        plt.tight_layout()
        plt.savefig('data/correlation_matrix.png')
        print("Saved: data/correlation_matrix.png")
    else:
        print("Not enough numeric columns for correlation matrix.")

if __name__ == "__main__":
    train, test = load_data()
    basic_info(train, test)
    target_distribution(train)
    correlation_analysis(train)
