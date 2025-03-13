import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from preprocessing import preprocess_data  # À adapter selon votre fonction de prétraitement

# Fonction de chargement des modèles
def load_model(model_path):
    _, ext = os.path.splitext(model_path)
    if ext in ['.keras', '.h5']:
        from model_utils import cox_loss  # Assurez-vous que le fichier model_utils.py est dans le PYTHONPATH
        return tf.keras.models.load_model(model_path, custom_objects={"cox_loss": cox_loss})
    else:
        return joblib.load(model_path)

# Fonction de prédiction générique
def predict_survival(model, data):
    return model.predict(data)

# Fonction pour ajouter de nouvelles variables
def add_new_variable(df, variable_name, variable_value):
    df[variable_name] = variable_value
    return df

# Chargement des modèles
cph_model = load_model('models/coxph.pkl')
rsf_model = load_model('models/rsf.pkl')
gbst_model = load_model('models/gbst.pkl')
deep_model = load_model('models/deepsurv.keras')

# Titre de l'application
st.title("Prédiction du Temps de Survie des Patients Atteints de Cancer Gastrique")

# Onglets
tabs = ["Formulaire", "Résultats", "Tableau de bord", "Ajouter une variable", "Aide"]
selected_tab = st.sidebar.radio("Sélectionnez une section", tabs)

if selected_tab == "Formulaire":
    st.header("Entrez les informations du patient")

    # Formulaire pour les données du patient
    patient_inputs = {}
    features = ['Cardiopathie', 'Ulceregastrique', 'Douleurepigastrique', 
                'Ulcero-bourgeonnant', 'Denitrution', 'Tabac', 
                'Mucineux', 'Infiltrant', 'Stenosant', 'Metastases', 'Adenopathie']

    for feature in features:
        patient_inputs[feature] = st.selectbox(
            f"Le patient a-t-il {feature} ?",
            options=["Oui", "Non"],
            index=0
        )

    patient_inputs['AGE'] = st.number_input("Âge du patient", min_value=0, max_value=120, value=50)

    # Créer un DataFrame avec les données saisies
    patient_data = pd.DataFrame([patient_inputs])

    # Prétraitement des données
    patient_data = preprocess_data(patient_data)

    # Bouton pour lancer la prédiction
    if st.button("Prédire le temps de survie"):
        # Prédiction pour chaque modèle
        cph_pred = predict_survival(cph_model, patient_data)
        rsf_pred = predict_survival(rsf_model, patient_data)
        gbst_pred = predict_survival(gbst_model, patient_data)
        deep_pred = predict_survival(deep_model, patient_data)
        
        st.subheader("Résultats des modèles")
        st.write(f"Modèle Cox Proportionnel : {cph_pred[0]} mois")
        st.write(f"Random Survival Forest : {rsf_pred[0]} mois")
        st.write(f"Gradient Boosting Survival Tree : {gbst_pred[0]} mois")
        st.write(f"Deep Survival Model : {deep_pred[0]} mois")

elif selected_tab == "Résultats":
    st.header("Visualisation des résultats")
    st.write("Cette section peut afficher des graphiques ou des tables récapitulatives des prédictions passées.")

elif selected_tab == "Tableau de bord":
    st.header("Tableau de bord des résultats")
    st.write("Afficher des statistiques et analyses de survie, avec des courbes de Kaplan-Meier, par exemple.")

elif selected_tab == "Ajouter une variable":
    st.header("Ajouter une nouvelle variable")
    new_variable_name = st.text_input("Nom de la nouvelle variable")
    new_variable_value = st.text_input("Valeur de la nouvelle variable")

    if st.button("Ajouter la variable"):
        if new_variable_name and new_variable_value:
            # Ajouter la nouvelle variable aux données
            patient_data = add_new_variable(patient_data, new_variable_name, new_variable_value)
            st.success(f"Variable {new_variable_name} ajoutée avec succès.")

elif selected_tab == "Aide":
    st.header("Aide à l'utilisation")
    st.write("Des informations détaillées sur l'application, le but de l'étude, et comment l'utiliser.")
