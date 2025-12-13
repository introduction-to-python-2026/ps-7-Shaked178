import kagglehub
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("mattop/leading-causes-of-death-in-the-united-states")
files_in_dataset = os.listdir(path)
data_file_name = None

for f_name in files_in_dataset:
    if f_name.lower().endswith(".csv"):
        data_file_name = f_name
        break

if data_file_name:
    file_path = data_file_name
    from kagglehub import KaggleDatasetAdapter

    df = kagglehub.load_dataset(
      KaggleDatasetAdapter.PANDAS,
      "mattop/leading-causes-of-death-in-the-united-states",
      file_path,
    )
    print("Data loaded successfully.")
    
    features_of_interest = ['Year', 'Deaths', 'Population', 'Crude Rate']
    selected_features_df = df[features_of_interest]
    print(f"Selected features: {features_of_interest}")
    
    print("Calculating and saving correlation heatmap...")
    correlation_matrix = selected_features_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5
    )
    plt.title('Correlation Matrix Heatmap')
    
    figure_file_name = 'correlation_heatmap.png'
    plt.savefig(figure_file_name)
    print(f"Successfully saved the figure as {figure_file_name}")

else:
    print("Error: No CSV file found.")
