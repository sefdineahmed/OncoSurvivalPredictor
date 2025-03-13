import pandas as pd


def preprocess_data(data):
    # Exemple de prétraitement pour les variables binaires
    data['Cardiopathie'] = data['Cardiopathie'].map({'Oui': 1, 'Non': 0})
    data['Ulceregastrique'] = data['Ulceregastrique'].map({'Oui': 1, 'Non': 0})
    data['Douleurepigastrique'] = data['Douleurepigastrique'].map({'Oui': 1, 'Non': 0})
    data['Ulcero-bourgeonnant'] = data['Ulcero-bourgeonnant'].map({'Oui': 1, 'Non': 0})  # Correction ici
    data['Denitrution'] = data['Denitrution'].map({'Oui': 1, 'Non': 0})
    data['Tabac'] = data['Tabac'].map({'Oui': 1, 'Non': 0})
    data['Mucineux'] = data['Mucineux'].map({'Oui': 1, 'Non': 0})
    data['Infiltrant'] = data['Infiltrant'].map({'Oui': 1, 'Non': 0})
    data['Stenosant'] = data['Stenosant'].map({'Oui': 1, 'Non': 0})
    data['Metastases'] = data['Metastases'].map({'Oui': 1, 'Non': 0})
    data['Adenopathie'] = data['Adenopathie'].map({'Oui': 1, 'Non': 0})
    return data
