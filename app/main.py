
import streamlit as st
import pandas as pd
import joblib
import os
import tensorflow as tf
import numpy as np  # Assurez-vous que numpy est importé pour éviter l'erreur 'np' non défini

from preprocessing import preprocess_data  # Assurez-vous que ce module existe et est correctement importé

# -------------------------------------------------------------
# Fonctions utilitaires
# -------------------------------------------------------------

def load_model(model_path):
    """
    Charge un modèle pré-entraîné à partir d'un fichier.
    Prend en charge les modèles Keras (.keras, .h5) et joblib.
    """
    _, ext = os.path.splitext(model_path)
    if ext in ['.keras', '.h5']:
        # Pour les modèles Keras, on suppose que la fonction de perte 'cox_loss'
        # est enregistrée via @tf.keras.saving.register_keras_serializable()
        from model_utils import cox_loss  # Assurez-vous que model_utils.py est dans le PYTHONPATH
        return tf.keras.models.load_model(model_path, custom_objects={"cox_loss": cox_loss})
    else:
        return joblib.load(model_path)


def predict_survival(model, data):
    """
    Effectue une prédiction de survie selon le type de modèle.
    """
    # Cas pour les modèles comme CoxPHFitter
    if hasattr(model, "predict_median"):  # Cox Proportionnel
        pred = model.predict_median(data)
        return pred  # Retourne directement la médiane sans tenter d'indexer un tableau
    
    # Cas pour les modèles comme RSF, GBST qui retournent des tableaux numpy
    elif hasattr(model, "predict"):  
        prediction = model.predict(data)
        if isinstance(prediction, np.ndarray):  # Vérifie si c'est un tableau numpy
            return prediction[0]  # Prédiction dans le premier élément (même si c'est un tableau)
        else:
            return prediction  # Si ce n'est pas un tableau, renvoyez directement la prédiction
    
    # Cas pour DeepSurv (qui renvoie une sortie de type numpy.ndarray)
    elif hasattr(model, "predict"):
        prediction = model.predict(data)
        return prediction[0][0]  # DeepSurv renvoie un tableau 2D, donc nous prenons la première valeur

    else:
        raise ValueError(f"Le modèle {model} ne supporte pas la prédiction de survie.")


def clean_prediction(prediction, model_name):
    """
    Fonction pour nettoyer et ajuster les résultats des prédictions pour les afficher correctement.
    """
    if model_name == "COXPH":
        # CoxProportionnel retourne une valeur unique, on vérifie et renvoie un nombre de mois
        return max(prediction, 0)  # Prédiction ne peut pas être négative
    
    elif model_name == "RSF" or model_name == "GBST":
        # Si les résultats sont des probabilités, on doit ajuster pour les mois
        return max(prediction, 0)  # Ajuster pour ne pas retourner une valeur négative
    
    elif model_name == "DEEPSURV":
        # DeepSurv retourne un tableau, nous devons prendre la valeur en premier élément
        prediction = prediction[0]
        return max(prediction, 1)  # Eviter les valeurs négatives, minimum 1 jour (ou 1 mois si nécessaire)
    
    else:
        return prediction  # En cas d'erreur de modèle, renvoyer tel quel


# -------------------------------------------------------------
# Variables de l'étude et formulaire de saisie
# -------------------------------------------------------------

# Initialisation des champs du formulaire
initial_features = {
    'AGE': {"label": "Âge", "type": "number", "min": 18, "max": 120, "default": 50},
    'Cardiopathie': {"label": "Cardiopathie", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
    'Ulceregastrique': {"label": "Ulcère gastrique", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
    'Douleurepigastrique': {"label": "Douleur épigastrique", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
    'Ulcero-bourgeonnant': {"label": "Ulcero-bourgeonnant", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
    'Denitrution': {"label": "Dénutrition", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
    'Tabac': {"label": "Tabac", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
    'Mucineux': {"label": "Mucineux", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
    'Infiltrant': {"label": "Infiltrant", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
    'Stenosant': {"label": "Sténosant", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
    'Metastases': {"label": "Métastases", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
    'Adenopathie': {"label": "Adénopathie", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
}


# -------------------------------------------------------------
# Chargement des modèles
# -------------------------------------------------------------

def load_all_models():
    """
    Charge tous les modèles nécessaires.
    """
    models = {
        'coxph': load_model('models/coxph.pkl'),
        'rsf': load_model('models/rsf.pkl'),
        'gbst': load_model('models/gbst.pkl'),
        'deepsurv': load_model('models/deepsurv.keras')
    }
    return models


# -------------------------------------------------------------
# Interface utilisateur Streamlit
# -------------------------------------------------------------

# Titre de l'application
st.title("Prédiction du Temps de Survie des Patients Atteints de Cancer Gastrique")

# Section du formulaire de saisie des données du patient
st.header("Entrez les informations du patient")
patient_inputs = {}
for key, config in initial_features.items():
    if config["type"] == "number":
        patient_inputs[key] = st.number_input(
            config["label"], min_value=config.get("min", 0),
            max_value=config.get("max", 120),
            value=config.get("default", 50)
        )
    elif config["type"] == "selectbox":
        patient_inputs[key] = st.selectbox(
            config["label"],
            options=config["options"],
            index=config["options"].index(config.get("default", config["options"][0]))
        )

# Création d'un DataFrame avec les données saisies
patient_data = pd.DataFrame([patient_inputs])

# Prétraitement des données
patient_data = preprocess_data(patient_data)

# Chargement des modèles
models = load_all_models()

# Bouton pour lancer la prédiction
if st.button("Prédire le temps de survie"):
    st.subheader("Résultats des modèles")

    for model_name, model in models.items():
        try:
            pred = predict_survival(model, patient_data)
            # Nettoyage de la prédiction pour une sortie correcte
            cleaned_pred = clean_prediction(pred, model_name)
            st.write(f"{model_name.upper()} : {cleaned_pred} mois")
        except Exception as e:
            st.error(f"Erreur avec le modèle {model_name}: {e}")

# Option d'enregistrement des données (exemple, à adapter selon votre besoin)
if st.button("Enregistrer les données du patient"):
    patient_data.to_csv("patient_data.csv", mode='a', index=False)
    st.success("Données enregistrées avec succès !")
