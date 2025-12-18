import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set style
sns.set(style="whitegrid")
OUTPUT_DIR = "outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv('data/train.csv')
    return df

def plot_target_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Calories'], kde=True, color='blue')
    plt.title('Rozkład zmiennej docelowej (Calories)')
    plt.xlabel('Spalone Kalorie')
    plt.ylabel('Liczebność')
    plt.savefig(f'{OUTPUT_DIR}/target_dist_calories.png')
    plt.close()
    print(f"Saved {OUTPUT_DIR}/target_dist_calories.png")

def plot_correlation_matrix(df):
    plt.figure(figsize=(10, 8))
    # Select only numeric columns manually to avoid errors if non-numeric exist (like Gender before mapping)
    numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
    # If Sex is string, map it temporarily just for corr or ignore it if not mapped yet
    # Checking if Sex is string or mapped in raw data. Usually raw is string.
    if df['Sex'].dtype == 'O':
        df_corr = df.copy()
        df_corr['Sex'] = df_corr['Sex'].map({'male': 0, 'female': 1})
        numeric_cols.append('Sex')
        corr = df_corr[numeric_cols].corr()
    else:
         corr = df[numeric_cols].corr()
         
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Macierz korelacji')
    plt.savefig(f'{OUTPUT_DIR}/correlation_matrix.png')
    plt.close()
    print(f"Saved {OUTPUT_DIR}/correlation_matrix.png")

def plot_scatter_features(df):
    features = ['Duration', 'Heart_Rate', 'Body_Temp']
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[feature], y=df['Calories'], alpha=0.5)
        plt.title(f'Zależność: {feature} vs Calories')
        plt.xlabel(feature)
        plt.ylabel('Calories')
        plt.savefig(f'{OUTPUT_DIR}/scatter_{feature}_calories.png')
        plt.close()
        print(f"Saved {OUTPUT_DIR}/scatter_{feature}_calories.png")

if __name__ == "__main__":
    print("Generowanie wykresów...")
    df = load_data()
    plot_target_distribution(df)
    plot_correlation_matrix(df)
    plot_scatter_features(df)
    print("Zakończono.")
