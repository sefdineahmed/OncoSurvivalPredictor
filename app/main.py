import streamlit as st
import pandas as pd
import joblib
import os
import tensorflow as tf
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
        return model.predict_median(data).values[0]
    elif hasattr(model, "predict_survival_function"):  # Cox Proportionnel avec fonction de survie
        survival_function = model.predict_survival_function(data)
        return survival_function.iloc[0].index[0]  # Renvoie la première prédiction (temps médian)
    
    # Cas pour les autres modèles comme Random Survival Forest (RSF), Gradient Boosting Survival (GBST), DeepSurv
    elif hasattr(model, "predict"):  
        return model.predict(data)[0]  # Prédiction directe (cette ligne peut être adaptée selon votre modèle)
    
    else:
        raise ValueError(f"Le modèle {model} ne supporte pas la prédiction de survie.")


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
            st.write(f"{model_name.upper()} : {pred} mois")
        except Exception as e:
            st.error(f"Erreur avec le modèle {model_name}: {e}")

# Option d'enregistrement des données (exemple, à adapter selon votre besoin)
if st.button("Enregistrer les données du patient"):
    patient_data.to_csv("patient_data.csv", mode='a', index=False)
    st.success("Données enregistrées avec succès !")
