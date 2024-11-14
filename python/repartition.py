import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

file_name = "cleaned_merged_heart_dataset.csv"

df = pd.read_csv("./data/"+file_name)

def repartition_maladie(data):
    # data_target=data['target']
    # print(data['target'])
    # x = data_target.count(0)
    # y= data_target.count(1)
    # plt.pie(np.array([x,y]), labels=["Risque faible","Risque élévé"])
    sns.countplot(x='target', data=data)
    plt.title("Distribution des maladies cardiaques")
    plt.show()


repartition_maladie(df)

# Filtrer les lignes où 'target' vaut 1
df_target_1 = df[df['target'] == 1]

