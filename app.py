import streamlit as st
import pandas as pd
import joblib
import yaml
import numpy as np
from tensorflow.keras.models import load_model
from utils.data_loader import load_config
from utils.model_predictor import predict_survival

# Configuration initiale
st.set_page_config(page_title="SurvivalMD", page_icon="⚕️", layout="wide")

# Chargement des configurations
variables_config = load_config('config/variables.yaml')
css_style = open('config/styles.css').read()
st.markdown(f'<style>{css_style}</style>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3712/3712068.png", width=100)
page = st.sidebar.radio("Navigation", ["📊 Prediction", "📥 Nouveau Patient", "ℹ️ À propos"])

# Chargement des modèles (cache pour performance)
@st.cache_resource
def load_models():
    return {
        'Cox': joblib.load('models/cph_model.joblib'),
        'RSF': joblib.load('models/rsf_model.joblib'),
        'GBST': joblib.load('models/gbst_model.joblib'),
        'DeepSurv': load_model('models/best_model.keras')
    }

models = load_models()

# Page de prédiction
if page == "📊 Prediction":
    st.title("🔮 SurvivalMD - Prédiction Intelligente de Survie")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Cox PH", "Random Survival Forest", "Gradient Boosting", "Deep Survival"])
    
    # Formulaire dynamique
    with st.form("patient_form"):
        cols = st.columns(3)
        inputs = {}
        for var in variables_config['variables']:
            with cols[int(var['column'])]:
                if var['type'] == 'numeric':
                    inputs[var['name']] = st.slider(var['label'], 
                                                   min_value=var['min'], 
                                                   max_value=var['max'],
                                                   value=var['default'])
                elif var['type'] == 'categorical':
                    inputs[var['name']] = st.selectbox(var['label'], var['categories'])
        
        submitted = st.form_submit_button("Calculer la Survie", use_container_width=True)
    
    if submitted:
        df_input = pd.DataFrame([inputs])
        
        with tab1:
            show_prediction('Cox', df_input)
        with tab2:
            show_prediction('RSF', df_input)
        with tab3:
            show_prediction('GBST', df_input)
        with tab4:
            show_deep_prediction(df_input)

# Page d'enregistrement des patients
elif page == "📥 Nouveau Patient":
    st.title("📥 Enregistrement des Nouveaux Patients")
    with st.form("new_patient_form"):
        # Formulaire dynamique étendu
        inputs = {}
        for var in variables_config['variables'] + variables_config['additional_vars']:
            if var['type'] == 'numeric':
                inputs[var['name']] = st.number_input(var['label'], 
                                                     min_value=var.get('min', 0), 
                                                     max_value=var.get('max', 100))
            elif var['type'] == 'categorical':
                inputs[var['name']] = st.selectbox(var['label'], var['categories'])
        
        if st.form_submit_button("Sauvegarder"):
            save_patient_data(inputs)
            st.success("Données sauvegardées avec succès!")

# Page À propos
else:
    st.title("⚕️ À Propos de SurvivalMD")
    st.markdown("""
    ## L'Intelligence Artificielle au Service de la Vie
    **SurvivalMD** combine quatre modèles d'apprentissage automatique de pointe pour prédire 
    la survie des patients après traitement avec une précision inégalée.
    
    ### Nos Atouts :
    - 🔍 Analyse multivariée intelligente
    - 🧠 Modèles constamment améliorés
    - 📈 Dashboard interactif temps réel
    - 🔒 Sauvegarde sécurisée des données
    
    ## Modèles Utilisés
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/1534/1534959.png", width=80)
        st.markdown("**Cox PH**\nModèle statistique classique")
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/1534/1534996.png", width=80)
        st.markdown("**RSF**\nForêts aléatoires survie")
    with col3:
        st.image("https://cdn-icons-png.flaticon.com/512/2694/2694993.png", width=80)
        st.markdown("**GBST**\nGradient boosting optimisé")
    with col4:
        st.image("https://cdn-icons-png.flaticon.com/512/1534/1534990.png", width=80)
        st.markdown("**DeepSurv**\nRéseau neuronal profond")

def show_prediction(model_name, data):
    try:
        prediction = predict_survival(models[model_name], data)
        st.metric(f"Survie Médiane Prédite ({model_name})", 
                 f"{prediction['median_survival']} mois")
        st.plotly_chart(prediction['curve'], use_container_width=True)
    except Exception as e:
        st.error(f"Erreur de prédiction: {str(e)}")

def show_deep_prediction(data):
    try:
        proba = models['DeepSurv'].predict(data.values)
        st.metric("Probabilité de Survie à 1 An", f"{proba[0][0]*100:.2f}%")
        st.progress(float(proba[0][0]))
    except Exception as e:
        st.error(f"Erreur Deep Learning: {str(e)}")

def save_patient_data(data):
    df = pd.DataFrame([data])
    today = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    df.to_csv(f"data/new_patients/patient_{today}.csv", index=False)
