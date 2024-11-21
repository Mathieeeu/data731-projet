import numpy as np
from math import e
import pandas as pd   
from scipy.stats import entropy
import matplotlib.pyplot as plt

#Calcul entropy avec une librairie
#value, count = np.unique(column, return_counts=True) 
#print(entropy(count, base=2))

def entropie(data, base=None):
    vc = pd.Series(data).value_counts(normalize=True, sort=False)
    base = e if base is None else base 
    return -(vc * np.log(vc)/np.log(base)).sum()

def entropie_totale(data, base=None):
    """Entropie élevée : Beaucoup de diversité dans une colonne, indiquant qu'elle couvre une large gamme de catégories.
        Entropie faible : La colonne est dominée par une ou deux catégories.
    """
    dic_entropie ={}
    for feature in (data.columns.tolist()):
        dic_entropie[feature] = entropie(data[feature],base)
    dict_sorted = dict(sorted(dic_entropie.items(), key=lambda item: item[1]))
   
    #Graphique
    colors = plt.cm.viridis(np.linspace(0, 1, len(dict_sorted)))
    plt.figure(figsize=(10, 7))
    plt.barh(list(dict_sorted.keys()), list(dict_sorted.values()), color=colors)
    plt.xlabel("Valeur de l'entropie")
    plt.ylabel('Paramètres')
    plt.title('Entropie')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return dict_sorted


# Entropie conditionnelle
def entropie_conditionnelle(data, target):
    """"Quelle est l'incertitude restante dans target si nous connaissons les valeurs de given?"
    
    Si permet de prédire totalement, résultat proche de 0
    Aucune info apporter par given sur target, résultat élévé
    """
    dic_entropie ={}
    for feature in (data.columns.tolist()):
        if feature == target:
            continue
        total_entropie = 0
        for _,partie in df.groupby(feature):
            #calcul de la probabilité de chaque valeur sur le dataset
            prob_valeur = len(partie) / len(df)
            target_probs = partie[target].value_counts(normalize=True)
            entropie_partie = -(target_probs * np.log2(target_probs)).sum()
            #Pondération
            total_entropie += prob_valeur * entropie_partie
            dic_entropie[feature] = total_entropie
    dict_sorted = dict(sorted(dic_entropie.items(), key=lambda item: item[1]))
    
    #Graphique
    colors = plt.cm.viridis(np.linspace(0, 1, len(dict_sorted)))
    plt.figure(figsize=(10, 7))
    plt.barh(list(dict_sorted.keys()), list(dict_sorted.values()), color=colors)
    plt.xlabel("Valeur de l'entropie")
    plt.ylabel('Paramètres')
    plt.title('Entropie Conditionnelle')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlim((0.7,1.001))
    plt.tight_layout()
    plt.show()
    
    return dict_sorted

def entropie_croisee(X, Y):
    total_entropie = 0
    crosstab = pd.crosstab(X, Y, normalize='index')
    for _, row in crosstab.iterrows():
        entropie = -(row * np.log(row + 1e-9)).sum()
    # for _, partie in data.groupby(feature):
    #     X = partie[feature].value_counts(normalize=True, sort=False)
    #     Y = partie[target].value_counts(normalize=True, sort=False)
    #     entropie = -(X * np.log(Y + 1e-9)).sum()  # Ajout d'une petite valeur pour éviter log(0)
        total_entropie += entropie
        
    return total_entropie

def entropie_croisee_totale(data, target):
    dic_entropie = {}
    for feature in data.columns.tolist():
        if feature == target:
            continue
        total_entropie = entropie_croisee(data[feature],data[target])
        dic_entropie[feature] = total_entropie
     
    dict_sorted = dict(sorted(dic_entropie.items(), key=lambda item: item[1]))
       
    #Graphique
    colors = plt.cm.viridis(np.linspace(0, 1, len(dict_sorted)))
    plt.figure(figsize=(10, 7))
    plt.barh(list(dict_sorted.keys()), list(dict_sorted.values()), color=colors)
    plt.xlabel("Valeur de l'entropie")
    plt.ylabel('Paramètres')
    plt.title('Entropie Croisée')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return dic_entropie

def entropie_relative(data,feature,target):
    """
    Mesure à quel point une distribution X s'écarte d'une distribution de référence Y
    E(X,Y)-E(X)
    """
    entropie_relative = entropie_croisee(data[feature],data[target]) - entropie(data[feature],2)
    
    # X = data[feature].value_counts(normalize=True, sort=False)
    # Y = reference_distribution.value_counts(normalize=True, sort=False)
    # # Éviter log(0) avec un epsilon
    # epsilon = 1e-15
    # X = np.clip(X, epsilon, 1)
    # Y = np.clip(Y, epsilon, 1)
    
    # # Calcul de la divergence KL
    # entropie_relative = np.sum(X * np.log(X / Y))
    
    return entropie_relative

def entropie_relative_totale(data, target):
    dic_entropie = {}
    for feature in data.columns.tolist():
        if feature == target:
            continue
        total_entropie = entropie_relative(data,feature,target)
        dic_entropie[feature] = total_entropie
        
    dict_sorted = dict(sorted(dic_entropie.items(), key=lambda item: item[1]))
       
    #Graphique
    colors = plt.cm.viridis(np.linspace(0, 1, len(dict_sorted)))
    plt.figure(figsize=(10, 7))
    plt.barh(list(dict_sorted.keys()), list(dict_sorted.values()), color=colors)
    plt.xlabel("Valeur de l'entropie")
    plt.ylabel('Paramètres')
    plt.title('Entropie Relative')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return dic_entropie


file_name = "cleaned_merged_heart_dataset.csv"

df = pd.read_csv("./data/"+file_name)

#print(entropie_totale(df, base=2))

#print(entropie_conditionnelle(df,"target"))

print(entropie_croisee_totale(df, "target"))

print(entropie_relative_totale(df, "target"))