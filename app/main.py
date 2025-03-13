import streamlit as st
import pandas as pd
import joblib
import os
import tensorflow as tf

from preprocessing import preprocess_data

# Fonction de chargement des modèles (Keras ou joblib)
def load_model(model_path):
    _, ext = os.path.splitext(model_path)
    if ext in ['.keras', '.h5']:
        # Pour les modèles Keras, on suppose que la fonction de perte 'cox_loss'
        # est enregistrée via @tf.keras.saving.register_keras_serializable()
        from model_utils import cox_loss  # Assurez-vous que model_utils.py est dans le PYTHONPATH
        return tf.keras.models.load_model(model_path, custom_objects={"cox_loss": cox_loss})
    else:
        return joblib.load(model_path)

# Fonction de prédiction générique pour les modèles de survie
def predict_survival(model, data):
    # Vous pouvez adapter cette fonction selon le type de sortie de chaque modèle
    # Par exemple, pour Cox, on pourrait retourner la médiane prédite,
    # tandis que pour d'autres modèles on renvoie la prédiction directe.
    # Ici, on suppose que model.predict(data) retourne une valeur lisible.
    return model.predict(data)

# Variables de l'étude (initiales)
# Ces variables peuvent être étendues facilement en ajoutant de nouveaux champs dans le formulaire
initial_features = {
    'AGE': {"label": "Âge", "type": "number", "min": 18, "max": 120, "default": 50},
    'SEXE': {"label": "Sexe", "type": "selectbox", "options": ["Homme", "Femme"], "default": "Homme"},
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

# Chargement des modèles
cph_model    = load_model('models/coxph.pkl')
rsf_model    = load_model('models/rsf.pkl')
gbst_model   = load_model('models/gbst.pkl')
deep_model   = load_model('models/deepsurv.keras')

# Titre de l'application
st.title("Prédiction du Temps de Survie des Patients Atteints de Cancer Gastrique")

# Section du formulaire de saisie
st.header("Entrez les informations du patient")
patient_inputs = {}
for key, config in initial_features.items():
    if config["type"] == "number":
        patient_inputs[key] = st.number_input(
            config["label"], min_value=config.get("min", 0),
            max_value=config.get("max", 1000),
            value=config.get("default", 0)
        )
    elif config["type"] == "selectbox":
        patient_inputs[key] = st.selectbox(
            config["label"],
            options=config["options"],
            index=config["options"].index(config.get("default", config["options"][0]))
        )
    # Vous pouvez ajouter d'autres types de champs si nécessaire

# Création d'un DataFrame avec les données saisies
patient_data = pd.DataFrame([patient_inputs])

# Prétraitement des données (adaptation nécessaire selon votre fonction preprocess_data)
patient_data = preprocess_data(patient_data)

# Bouton pour lancer la prédiction
if st.button("Prédire le temps de survie"):
    # Prédictions pour chaque modèle
    cph_pred    = predict_survival(cph_model, patient_data)
    rsf_pred    = predict_survival(rsf_model, patient_data)
    gbst_pred   = predict_survival(gbst_model, patient_data)
    deep_pred   = predict_survival(deep_model, patient_data)
    
    st.subheader("Résultats des modèles")
    st.write(f"Modèle Cox Proportionnel : {cph_pred} mois")
    st.write(f"Random Survival Forest : {rsf_pred} mois")
    st.write(f"Gradient Boosting Survival Tree : {gbst_pred} mois")
    st.write(f"Deep Survival Model : {deep_pred} mois")

# Option d'enregistrement des données (exemple, à adapter selon votre besoin)
if st.button("Enregistrer les données du patient"):
    # Ici, vous pouvez sauvegarder les données dans un fichier ou une base de données
    patient_data.to_csv("patient_data.csv", mode='a', index=False)
    st.success("Données enregistrées avec succès !")
