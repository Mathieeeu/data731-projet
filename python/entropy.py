import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt

def entropie(data):
    vc = pd.Series(data).value_counts(normalize=True, sort=False)
    return -(vc * np.log(vc)).sum()

def entropie_totale(data):
    """
    Entropie élevée : Beaucoup de diversité dans une colonne, indiquant qu'elle couvre une large gamme de catégories.
    Entropie faible : La colonne est dominée par une ou deux catégories.
    """
    dic_entropie ={}
    for feature in (data.columns.tolist()):
        dic_entropie[feature] = entropie(data[feature])
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
        for _,partie in data.groupby(feature):
            #calcul de la probabilité de chaque valeur sur le dataset
            prob_valeur = len(partie) / len(data)
            target_probs = partie[target].value_counts(normalize=True)
            entropie_partie = -(target_probs * np.log(target_probs)).sum()
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
    plt.xlim((0.5,0.7))
    plt.tight_layout()
    plt.show()
    
    return dict_sorted


def entropie_croisee(data,column, target):
    """
    Mesure à quel point une distribution X est bonne pour représenter une autre distribution Y
    Notion de qualité
    Plus l'entropie est faible plus Y est proche de X
    """
    total_entropie = 0
    # Distribution conditionnelle P(x | target)
    distribution = pd.crosstab(data[column], data[target], normalize='columns')
    distribution_target = data[target].value_counts(normalize=True)
    # Calcul de l'entropie croisée
    entropie_croisee = -(distribution * np.log(distribution + 1e-9)).sum(axis=0)
    total_entropie = np.dot(entropie_croisee, distribution_target.values)
    return total_entropie.sum()

def entropie_croisee_totale(data, target):
    dic_entropie = {}
    for feature in data.columns:
        if feature == target:
            continue
        dic_entropie[feature] = entropie_croisee(data,feature,target)
     
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
    Notion d'écart
    """
    entropie_relative = entropie_croisee(data, feature) - entropie(data[feature])    
    return entropie_relative

def entropie_relative_totale(data, target):
    dict_entropie = {}
    for feature in data.columns.tolist():
        if feature == target:
            continue
        dict_entropie[feature] = entropie_relative(data,feature,target)
        
    dict_sorted = dict(sorted(dict_entropie.items(), key=lambda item: item[1]))
       
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
    
    return dict_sorted


file_name = "cleaned_merged_heart_dataset.csv"

df = pd.read_csv("./data/"+file_name)

print(entropie_totale(df))

print(entropie_conditionnelle(df,"target"))

print(entropie_croisee_totale(df, "target"))

#print(entropie_relative_totale(df, "target"))