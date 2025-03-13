import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from preprocessing import preprocess_data

# Fonction de chargement des modèles (Keras ou joblib)
def load_model(model_path):
    _, ext = os.path.splitext(model_path)
    if ext in ['.keras', '.h5']:
        from model_utils import cox_loss  # Assurez-vous que model_utils.py est dans le PYTHONPATH
        return tf.keras.models.load_model(model_path, custom_objects={"cox_loss": cox_loss})
    else:
        return joblib.load(model_path)

# Fonction de prédiction générique pour les modèles de survie
def predict_survival(model, data):
    return model.predict(data)

# Variables initiales de l'étude
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

# Fonction pour ajouter de nouvelles variables (exemple)
def add_new_feature(new_feature_name, new_feature_details):
    initial_features[new_feature_name] = new_feature_details

# Chargement des modèles
cph_model    = load_model('models/coxph.pkl')
rsf_model    = load_model('models/rsf.pkl')
gbst_model   = load_model('models/gbst.pkl')
deep_model   = load_model('models/deepsurv.keras')

# Titre de l'application
st.title("Prédiction du Temps de Survie des Patients Atteints de Cancer Gastrique")

# Onglet de formulaire pour la saisie des informations
form_section = st.sidebar.selectbox("Choisir une section", ["Formulaire de saisie", "Tableau de bord", "Analyse des modèles", "Paramètres"])

if form_section == "Formulaire de saisie":
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

    # Création d'un DataFrame avec les données saisies
    patient_data = pd.DataFrame([patient_inputs])

    # Prétraitement des données (adaptation nécessaire selon votre fonction preprocess_data)
    patient_data = preprocess_data(patient_data)

    # Bouton pour lancer la prédiction
    if st.button("Prédire le temps de survie"):
        # Prédictions pour chaque modèle
        cph_pred = predict_survival(cph_model, patient_data)
        rsf_pred = predict_survival(rsf_model, patient_data)
        gbst_pred = predict_survival(gbst_model, patient_data)
        deep_pred = predict_survival(deep_model, patient_data)

        st.subheader("Résultats des modèles")
        st.write(f"Modèle Cox Proportionnel : {cph_pred} mois")
        st.write(f"Random Survival Forest : {rsf_pred} mois")
        st.write(f"Gradient Boosting Survival Tree : {gbst_pred} mois")
        st.write(f"Deep Survival Model : {deep_pred} mois")

elif form_section == "Tableau de bord":
    st.header("Tableau de bord des prédictions")
    # Vous pouvez ajouter des graphiques ou autres visualisations ici en fonction des données prédites

elif form_section == "Analyse des modèles":
    st.header("Analyse des modèles")
    # Ajoutez des outils d'analyse comparative ici

elif form_section == "Paramètres":
    st.header("Paramètres de l'application")
    st.write("Ajoutez de nouvelles variables et ajustez les modèles ici.")
    # Permettre à l'utilisateur d'ajouter de nouvelles variables à l'étude
    new_feature_name = st.text_input("Nom de la nouvelle variable")
    new_feature_label = st.text_input("Label de la nouvelle variable")
    new_feature_type = st.selectbox("Type de la nouvelle variable", ["number", "selectbox"])
    new_feature_options = st.text_area("Options de la variable (si applicable, séparées par des virgules)")
    
    if st.button("Ajouter une nouvelle variable"):
        new_feature_details = {
            "label": new_feature_label,
            "type": new_feature_type,
            "options": new_feature_options.split(",") if new_feature_type == "selectbox" else [],
            "default": new_feature_options.split(",")[0] if new_feature_type == "selectbox" else 0
        }
        add_new_feature(new_feature_name, new_feature_details)
        st.success(f"Nouvelle variable '{new_feature_name}' ajoutée avec succès!")

