import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

file_name = "cleaned_merged_heart_dataset.csv"

df = pd.read_csv("./data/"+file_name)

#Corrélation heatmap
def matriceCorrelation(data):
    numeric_df = data.select_dtypes(include=[np.number]) #s'assurer que l'on a des données numériques
    plt.figure(figsize=(12, 8))
    df_corr=numeric_df.corr()
    color = sns.diverging_palette(10, 130, s=90, l=80, center="light", as_cmap=True)
    sns.heatmap(df_corr,annot=True, cmap=color)
    plt.title("Matrice de corréralation")
    plt.show()


matriceCorrelation(df)

