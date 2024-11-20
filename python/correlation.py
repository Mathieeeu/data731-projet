import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

file_name = "cleaned_merged_heart_dataset.csv"

df = pd.read_csv("./data/"+file_name)

#Corrélation heatmap
def matriceCorrelation(data):
    numeric_df = data.select_dtypes(include=[np.number]) #s'assurer que l'on a des données numériques
    df_corr=numeric_df.corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_corr,annot=True, cmap="magma")
    plt.title("Matrice de corréralation")
    plt.show()


matriceCorrelation(df)

def nivCorrAvecTarget(data):
    data_corr = abs(data.corr())
    data_corr_target = data_corr['target']
    data = data_corr_target.drop('target')
    sorted_id = data.argsort()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_id)))
    plt.figure(figsize=(12, 8))
    plt.barh(data.keys()[sorted_id],data[sorted_id], color = colors)
    plt.xlabel('Corrélation')
    plt.ylabel('Paramètres')
    plt.title('Niveau de corrélation par rapport à target')
    plt.show()
    
nivCorrAvecTarget(df)

#Corrélation de deux variables par rapport au target du coup (3 variables)
#Ex : acp+bage = target
# Z = corr(cp,target)