import numpy as np
from math import e
import pandas as pd   
from scipy.stats import entropy

#Calcul entropy avec une librairie
#value, count = np.unique(column, return_counts=True) 
#print(entropy(count, base=2))

def pandas_entropy(column, base=None):
    
    return ()

def entropy(data, base=None):
    dic_entropy ={}
    for feature in (data.columns.tolist()):
        vc = pd.Series(data[feature]).value_counts(normalize=True, sort=False)
        base = e if base is None else base 
        dic_entropy[feature] = -(vc * np.log(vc)/np.log(base)).sum()
    dict_sorted = dict(sorted(dic_entropy.items(), key=lambda item: item[1]))
    return dict_sorted

from collections import Counter

# Entropie conditionnelle
def entropy_conditionnelle(data, target):
    """"Quelle est l'incertitude restante dans target si nous connaissons les valeurs de given?"
    
    Si permet de prédire totalement, résultat proche de 0
    Aucune info apporter par given sur target, résultat élévé
    """
    dic_entropy ={}
    for feature in (data.columns.tolist()):
        total_entropy = 0
        for _,partie in df.groupby(feature):
            #calcul de la probabilité de chaque valeur sur le dataset
            prob_valeur = len(partie) / len(df)
            target_probs = partie[target].value_counts(normalize=True)
            entropy_partie = -(target_probs * np.log2(target_probs)).sum()
            #Pondération
            total_entropy += prob_valeur * entropy_partie
            dic_entropy[feature] = total_entropy
    dict_sorted = dict(sorted(dic_entropy.items(), key=lambda item: item[1]))
    return dict_sorted


file_name = "cleaned_merged_heart_dataset.csv"

df = pd.read_csv("./data/"+file_name)

print(entropy(df, base=2))
print(entropy_conditionnelle(df,"target"))
