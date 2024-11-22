# Modèle de prédictions qui utilise toutes les entrées du dataset pour prédire la cible selon un ensemble de paramètres.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import tree

file_name = "cleaned_merged_heart_dataset.csv"
df = pd.read_csv("./data/"+file_name)

# Séparation des variables et de la cible
variables = df.drop('target', axis=1)
results = df['target']

# Régression linéaire
regression = linear_model.LinearRegression()
regression.fit(variables, results)

# Affichage des coefficients de la régression linéaire
print("Coefficients de la régression linéaire :")
print(pd.DataFrame({
    'variable': variables.columns,
    'coefficient': regression.coef_
}))
print("\n\n")

# Arbre de décision
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(variables, results)

# Affichage de l'importance des variables dans l'arbre de décision
print("Importance des variables dans l'arbre de décision :")
print(pd.DataFrame({
    'variable': variables.columns,
    'importance': decision_tree.feature_importances_
}))
print("\n\n")


# # Graphe de target en fonction de l'index de la ligne (entre les index index_min et index_max)
# index_min = 0
# index_max = 100
# plt.plot(range(index_min, index_max), results[index_min:index_max])

# # Prédiction de la cible (avec regressive) pour les entrées de la ligne index_min à index_max
# predictions = regression.predict(variables[index_min:index_max])
# plt.plot(range(index_min, index_max), predictions)


# plt.xlabel('Entrée n°')
# plt.ylabel('Malade (1=vrai)')
# plt.show()

# Prédiction d'une personne saisie manuellement 
person = pd.DataFrame({
    'age':22,
    'sex':0,
    'cp':0,
    'trestbps':140,
    'chol':200,
    'fbs':1,
    'restecg':0,
    'thalachh':200,
    'exang':0,
    'oldpeak':0,
    'slope':1,
    'ca':2,
    'thal':1
}, index=[0])

print(f"Prédiction de la cible pour la personne : {regression.predict(person)[0]}, {decision_tree.predict(person)[0]}")

# Graphique de la moyenne de target en fonction d'une clé avec les données réelles puis avec les prédictions
keys = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
#key = keys[0]



# Fonction pour générer un subplot pour chaque clé
def comparer_predictions_realité(key):
    grouped = df.groupby(key).mean()
    plt.plot(grouped.index, grouped['target'])

    # Création d'un dataframe avec toutes les valeurs prédites pour chaque valeur de la clé
    df_predictions = df.copy()
    df_predictions.drop('target', axis=1, inplace=True)
    df_predictions['prediction'] = regression.predict(df_predictions)
    grouped_predictions = df_predictions.groupby(key).mean()
    plt.plot(grouped_predictions.index, grouped_predictions['prediction'])  
    plt.ylabel(f'target/{key}')

# Affichage du graphique
plt.figure(figsize=(20, 20))
for i, key in enumerate(keys):
    plt.subplot(4, 4, i+1)
    comparer_predictions_realité(key)
plt.legend(['Réel', 'Prédiction'])
plt.show()