import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def repartition_maladie(data):
    # data_target=data['target']
    # print(data['target'])
    # x = data_target.count(0)
    # y= data_target.count(1)
    # plt.pie(np.array([x,y]), labels=["Risque faible","Risque élévé"])
    sns.countplot(x='target', data=data)
    plt.title("Distribution des maladies cardiaques")
    plt.show()


#repartition_maladie(df)

def repartitionParCaractere(data1, data2, feature, nb_barres):

    count, bins, ignored = plt.hist([data1[feature],data2[feature]], nb_barres , histtype="bar", color=['lightgreen','teal'],edgecolor='black', density=True, label=["Risque élevé", "Rsique faible"])
    #count, bins, ignored = plt.hist(data2[feature], nb_barres , histtype="barstacked", color='teal',edgecolor='black', density=True)
    plt.legend(prop={'size': 10})
    plt.title(f"Répartition du nombre de personnes à risque selon {feature}")
    plt.show()



file_name = "cleaned_merged_heart_dataset.csv"

df = pd.read_csv("./data/"+file_name)

nb_barres = 24

# Filtrer les lignes où 'target' vaut 1
df_target_1 = df[df['target'] == 1]
df_target_0 = df[df['target'] == 0]

repartitionParCaractere(df_target_1, df_target_0, "age", nb_barres)
#repartitionParCaractere(df, "age", nb_barres)
repartitionParCaractere(df_target_1,df_target_0, "sex", 2)
repartitionParCaractere(df_target_1,df_target_0, "cp", 5)
repartitionParCaractere(df_target_1,df_target_0, "trestbps", nb_barres)
repartitionParCaractere(df_target_1,df_target_0, "chol", nb_barres)
repartitionParCaractere(df_target_1,df_target_0, "fbs", 2)
repartitionParCaractere(df_target_1,df_target_0, "restecg", 3)
repartitionParCaractere(df_target_1,df_target_0, "thalachh", nb_barres)
repartitionParCaractere(df_target_1,df_target_0, "exang", 2)
repartitionParCaractere(df_target_1, df_target_0,"oldpeak", nb_barres)
repartitionParCaractere(df_target_1, df_target_0,"slope", 4)
repartitionParCaractere(df_target_1,df_target_0, "ca", 5)
repartitionParCaractere(df_target_1,df_target_0, "thal", 7)


