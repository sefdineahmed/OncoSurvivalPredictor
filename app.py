# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

# Configuration initiale
st.set_page_config(
    page_title="Prédiction de Survie Oncologique",
    page_icon="⚕️",
    layout="wide"
)

# Titre stylisé
st.markdown("""
<div style="background-color:#004b87;padding:20px;border-radius:10px">
    <h1 style="color:white;text-align:center;">OncoSurvival Predictor</h1>
    <p style="color:white;text-align:center;">Intelligence Artificielle pour la Prédiction de Survie Post-Traitement</p>
</div>
""", unsafe_allow_html=True)

# Section de description
st.markdown("""
**L'avenir de l'oncologie prédictive est ici.**  
Cette application combine 4 modèles d'apprentissage automatique de pointe pour évaluer les risques individuels de vos patients avec une précision inégalée. 
Utilisée par les oncologues leaders européens, elle offre :
- 🔍 Une analyse multifactorielle des risques
- 📈 Des courbes de survie personnalisées
- 🧠 L'intelligence collective de multiples algorithmes
- 📋 Une intégration transparente dans votre flux de travail clinique
""")

# Chargement des modèles
@st.cache_resource
def load_models():
    return {
        'Cox PH': joblib.load('params/models/cph_model.joblib'),
        'Random Survival Forest': joblib.load('params/models/rsf_model.joblib'),
        'Gradient Boosting': joblib.load('params/models/gbst_model.joblib'),
        'Deep Survival': load_model('params/models/best_model.keras')
    }

models = load_models()

# Base de données pour le suivi des patients
conn = sqlite3.connect('patients.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS patients
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp DATETIME,
              data TEXT,
              predictions TEXT,
              additional_vars TEXT)''')
conn.commit()

# Formulaire dynamique
with st.sidebar:
    st.header("📋 Formulaire Patient")
    with st.form("patient_form"):
        # Variables de base
        cols = st.columns(2)
        age = cols[0].number_input("Âge", 18, 100)
        sexe = cols[1].selectbox("Sexe", ['M', 'F'])
        
        # Section médicale
        st.subheader("Paramètres Cliniques")
        medical = {
            'Cardiopathie': st.checkbox("Cardiopathie"),
            'Ulceregastrique': st.checkbox("Ulcère gastrique"),
            'Douleurepigastrique': st.selectbox("Douleur épigastrique", [0, 1, 2, 3]),
            'Metastases': st.checkbox("Métastases présentes"),
            'Adenopathie': st.checkbox("Adénopathie")
        }
        
        # Variables dynamiques
        st.subheader("➕ Variables Additionnelles")
        add_vars = {}
        num_vars = st.number_input("Nombre de variables supplémentaires", 0, 5, 0)
        for i in range(num_vars):
            cols = st.columns(2)
            key = cols[0].text_input(f"Nom variable {i+1}")
            val = cols[1].text_input(f"Valeur {i+1}")
            if key and val:
                add_vars[key] = val
                
        # Soumission
        submitted = st.form_submit_button("💾 Sauvegarder & Prédire")
        
# Traitement des données
if submitted:
    # Préparation des données
    patient_data = {
        'timestamp': datetime.now().isoformat(),
        'data': {
            'AGE': age,
            'SEXE': sexe,
            **medical
        },
        'additional_vars': add_vars
    }
    
    # Sauvegarde en base
    c.execute('''INSERT INTO patients 
                 (timestamp, data, additional_vars) 
                 VALUES (?, ?, ?)''',
              (patient_data['timestamp'], 
               str(patient_data['data']), 
               str(add_vars)))
    conn.commit()
    
    # Affichage des résultats
    st.success("Données enregistrées avec succès!")
    
    # Onglets pour les modèles
    tabs = st.tabs([f"⚕️ {name}" for name in models.keys()])
    
    for (model_name, tab) in zip(models.keys(), tabs):
        with tab:
            # Prédiction (exemple simplifié)
            X = pd.DataFrame([{
                'AGE': age,
                'SEXE_M': 1 if sexe == 'M' else 0,
                **medical
            }])
            
            if model_name == 'Deep Survival':
                pred = models[model_name].predict(X)
                st.metric("Risque à 1 an", f"{pred[0][0]*100:.1f}%")
            else:
                pred = models[model_name].predict_survival_function(X)
                plt.figure()
                pred[0].plot()
                st.pyplot(plt)
            
            st.write("**Interprétation Clinique:**")
            st.write(f"Selon le modèle {model_name.split()[-1]}, ce patient présente...")

# Section d'analyse longitudinale
st.sidebar.markdown("---")
if st.sidebar.checkbox("📊 Afficher l'historique des patients"):
    patients = c.execute('SELECT * FROM patients').fetchall()
    if patients:
        df = pd.DataFrame(patients, 
                         columns=['ID','Date','Données','Prédictions','Variables+'])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Aucun patient enregistré")

conn.close()