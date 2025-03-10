import pandas as pd

def preprocess_data(data):
    """Prétraitement des données avant la prédiction."""
    
    # Conversion des variables catégorielles en variables numériques
    data['SEXE'] = data['SEXE'].map({'Homme': 1, 'Femme': 0})
    data['Cardiopathie'] = data['Cardiopathie'].map({'Oui': 1, 'Non': 0})
    data['Ulceregastrique'] = data['Ulceregastrique'].map({'Oui': 1, 'Non': 0})
    data['Douleurepigastrique'] = data['Douleurepigastrique'].map({'Oui': 1, 'Non': 0})
    data['Ulcerobourgeonnant'] = data['Ulcerobourgeonnant'].map({'Oui': 1, 'Non': 0})
    data['Denitrution'] = data['Denitrution'].map({'Oui': 1, 'Non': 0})
    data['Tabac'] = data['Tabac'].map({'Oui': 1, 'Non': 0})
    data['Mucineux'] = data['Mucineux'].map({'Oui': 1, 'Non': 0})
    data['Infiltrant'] = data['Infiltrant'].map({'Oui': 1, 'Non': 0})
    data['Stenosant'] = data['Stenosant'].map({'Oui': 1, 'Non': 0})
    data['Metastases'] = data['Metastases'].map({'Oui': 1, 'Non': 0})
    data['Adenopathie'] = data['Adenopathie'].map({'Oui': 1, 'Non': 0})

    # Remplacer les valeurs NaN par la médiane (au cas où certaines valeurs sont manquantes)
    data.fillna(data.median(), inplace=True)

    return data
