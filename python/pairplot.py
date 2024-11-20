import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def PairPlot(df):
    """
    Pair plot des variables du dataframe (attention ça rame fort)
    """
    # Ajout d'une nouvell colonne target pour le hue
    df['legend\ntarget:'] = df['target'].map({0: 0, 1: 1})

    sns.pairplot(df, hue='legend\ntarget:', height=1)
    plt.show()

    df.drop('legend\ntarget:', axis=1, inplace=True)

filename = "cleaned_merged_heart_dataset.csv"
df = pd.read_csv('./data/' + filename)

print(df.index(0))

# matrice_correlation(df)
# PairPlot(df)