import streamlit as st
import pandas as pd
import joblib
from model_utils import load_model, predict_survival
from preprocessing import preprocess_data

# Charger les modèles
cph_model = load_model('models/coxph.pkl')
rsf_model = load_model('models/rsf.pkl')
gbst_model = load_model('models/gbst.pkl')
deep_model = load_model('models/deepsurv.keras')

# Titre de l'application
st.title("Prédiction du Temps de Survie des Patients Atteints de Cancer Gastrique")

# Saisie des données patient
st.header("Entrez les informations du patient")
age = st.number_input("Âge", min_value=18, max_value=120)
sexe = st.selectbox("Sexe", ["Homme", "Femme"])
cardiopathie = st.selectbox("Cardiopathie", ["Oui", "Non"])
ulceregastrique = st.selectbox("Ulcère gastrique", ["Oui", "Non"])
# Ajouter d'autres champs selon les variables disponibles...

# Prétraiter les données
patient_data = pd.DataFrame({
    'AGE': [age],
    'SEXE': [sexe],
    'Cardiopathie': [cardiopathie],
    'Ulceregastrique': [ulceregastrique],
    # Ajouter d'autres variables ici...
})

patient_data = preprocess_data(patient_data)

# Prédictions
if st.button("Prédire le temps de survie"):
    cph_pred = predict_survival(cph_model, patient_data)
    rsf_pred = predict_survival(rsf_model, patient_data)
    gbst_pred = predict_survival(gbst_model, patient_data)
    deep_pred = predict_survival(deep_model, patient_data)

    st.subheader("Résultats des modèles")
    st.write(f"Modèle Cox : {cph_pred} mois")
    st.write(f"Random Survival Forest : {rsf_pred} mois")
    st.write(f"Gradient Boosting Survival Tree : {gbst_pred} mois")
    st.write(f"Deep Survival Model : {deep_pred} mois")

# Enregistrer les données dans un fichier (optionnel)
if st.button("Enregistrer les données du patient"):
    # Code pour enregistrer les données dans un fichier ou une base de données
    pass
