import pandas as pd
import matplotlib.pyplot as plt

# Chemin du fichier CSV
file_path = './data/raw_merged_heart_dataset.csv'

# Charger le fichier CSV en utilisant l'encodage ISO-8859-1 et la virgule comme délimiteur
df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=',')

# Assurez-vous que la colonne 'thalachh' est bien numérique (conversion si nécessaire)
df['thalachh'] = pd.to_numeric(df['thalachh'], errors='coerce')

df['trestbps'] = pd.to_numeric(df['trestbps'], errors='coerce')


# on vire les non malade du dataframe 
df_targ=df[df['target']==1]

# Variables globales
trancheAge = 5  # Taille de la tranche d'âge
tranche_t = 5    # Taille de la tranche de 'thalachh'

def graphique_age():
    # Créer des tranches d'âge
    df_targ['tranche_age'] = (df_targ['age'] // trancheAge) * trancheAge
    df_groupe = df_targ.groupby('tranche_age').agg({'target': 'sum'})
    
    # Affichage du DataFrame agrégé pour vérification
    print(df_groupe)

    # Configuration du graphique
    plt.figure(figsize=(10, 6))
    
    # Tracer le nombre moyen de patients par tranche d'âge
    plt.bar(df_groupe.index, df_groupe['target'], width=2, color='Navy', label=f"Nombre moyen de patients par tranche d'âge ({trancheAge} ans)")
    
    # Ajouter les étiquettes et le titre
    plt.xlabel(f"Tranche d'âge ({trancheAge} ans)")
    plt.ylabel("Nombre moyen de patients")
    plt.title(f"Nombre moyen de patients par tranche d'âge de {trancheAge} ans")
    plt.legend()
    plt.show()

    return None


def graphique_cp():
    # Je regroupe les données par 'cp' en calculant la moyenne de 'target'
    df_groupe = df.groupby('cp').agg({'target': 'mean'})

    # Affichage du DataFrame agrégé pour vérification
    print(df_groupe)

    # Configuration du graphique
    plt.figure(figsize=(10, 6))
    
    # Tracer le nombre moyen de patients par type de CP
    plt.bar(df_groupe.index, df_groupe['target'], width=0.8, color='Gold', label="Type de CP")

    # Ajouter les étiquettes et le titre
    plt.xlabel("Type de CP")
    plt.ylabel("Nombre moyen de patients")
    plt.title("Nombre moyen de patients par type de CP")
    plt.legend()
    plt.show()

    return None


def graphique_thalach():
    # Créer des tranches de thalachh
    df['groupement_t'] = (df['thalachh'] // tranche_t) * tranche_t
    
    df_groupe = df.groupby('groupement_t').agg({'target': 'mean'})
    
    # Affichage du DataFrame agrégé pour vérification
    print(df_groupe)

    # Configuration du graphique
    plt.figure(figsize=(10, 6))
    
    # Tracer le nombre moyen de patients par tranche de thalachh
    plt.bar(df_groupe.index, df_groupe['target'], width=1, color='Plum', label=f"Nombre moyen de patients par tranche de thalachh ({tranche_t})")
    
    # Ajouter les étiquettes et le titre
    plt.xlabel(f"Tranche de thalachh ({tranche_t})")
    plt.ylabel("Nombre moyen de patients")
    plt.title(f"Nombre moyen de patients par tranche de thalachh ({tranche_t})")
    plt.legend()
    plt.show()

    return None 


def graphique_trestbps():
    # Créer des tranches de thalachh
    df['groupement_t'] = (df['trestbps'] // tranche_t) * tranche_t
    
    df_groupe = df.groupby('groupement_t').agg({'target': 'mean'})
    
    # Affichage du DataFrame agrégé pour vérification
    print(df_groupe)

    # Configuration du graphique
    plt.figure(figsize=(10, 6))
    
    # Tracer le nombre moyen de patients par tranche de thalachh
    plt.bar(df_groupe.index, df_groupe['target'], width=1, color='crimson', label=f"Nombre moyen de patients par tranche de trestbps ({tranche_t})")
    
    # Ajouter les étiquettes et le titre
    plt.xlabel(f"Tranche de trestbps ({tranche_t})")
    plt.ylabel("Nombre moyen de patients")
    plt.title(f"Nombre moyen de patients par tranche de trestbps({tranche_t})")
    plt.legend()
    plt.show()
    return None 


# Exécuter des fonctions
graphique_age()
graphique_cp()
graphique_thalach()
graphique_trestbps()

print(df)
