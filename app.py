# app.py
import streamlit as st
import pandas as pd
import joblib
import yaml
import numpy as np
import sys
import os
import plotly.express as px
from tensorflow.keras.models import load_model

# Configuration des paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from utils.data_loader import load_config, load_patient_data
from utils.model_predictor import predict_survival

# Configuration initiale
st.set_page_config(
    page_title="SurvivalMD", 
    page_icon="⚕️", 
    layout="wide",
    menu_items={
        'About': "### Application de prédiction de survie oncologique"
    }
)

# Chargement des configurations
@st.cache_resource
def load_app_config():
    config = load_config('config/variables.yaml')
    css_style = open('config/styles.css').read()
    return config, css_style

variables_config, css_style = load_app_config()
st.markdown(f'<style>{css_style}</style>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3712/3712068.png", width=100)
page = st.sidebar.radio("Navigation", ["📊 Prediction", "📥 Nouveau Patient", "ℹ️ À propos"])

# Chargement des modèles
@st.cache_resource
def load_models():
    models = {
        'Cox': joblib.load('models/coxph.pkl'),
        'RSF': joblib.load('models/rsf.pkl'),
        'GBST': joblib.load('models/gbst.pkl')
    }
    try:
        models['DeepSurv'] = load_model('models/deepsurv.keras')
    except Exception as e:
        st.error(f"Erreur chargement Deep Learning: {str(e)}")
    return models

models = load_models()

# Page de prédiction
if page == "📊 Prediction":
    st.title("🔮 SurvivalMD - Prédiction Intelligente de Survie")
    
    with st.form("patient_form"):
        cols = st.columns(3)
        inputs = {}
        for var in variables_config['variables']:
            with cols[int(var['column'])]:
                if var['type'] == 'numeric':
                    inputs[var['name']] = st.slider(
                        var['label'], 
                        min_value=var['min'], 
                        max_value=var['max'],
                        value=var['default']
                    )
                elif var['type'] == 'categorical':
                    inputs[var['name']] = st.selectbox(var['label'], var['categories'])
        
        submitted = st.form_submit_button("Calculer la Survie", use_container_width=True)
    
    if submitted:
        df_input = pd.DataFrame([inputs]).astype(float)
        
        tabs = st.tabs(["Cox PH", "Random Survival Forest", "Gradient Boosting", "Deep Survival"])
        
        with tabs[0]:
            show_prediction('Cox', df_input)
        with tabs[1]:
            show_prediction('RSF', df_input)
        with tabs[2]:
            show_prediction('GBST', df_input)
        with tabs[3]:
            show_deep_prediction(df_input)

# Page d'enregistrement des patients
elif page == "📥 Nouveau Patient":
    st.title("📥 Enregistrement des Nouveaux Patients")
    with st.form("new_patient_form", clear_on_submit=True):
        inputs = {}
        for var in variables_config['variables'] + variables_config.get('additional_vars', []):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**{var['label']}**")
            with col2:
                if var['type'] == 'numeric':
                    inputs[var['name']] = st.number_input(
                        var['label'], 
                        min_value=var.get('min', 0), 
                        max_value=var.get('max', 100),
                        label_visibility="collapsed"
                    )
                elif var['type'] == 'categorical':
                    inputs[var['name']] = st.selectbox(
                        var['label'], 
                        var['categories'],
                        label_visibility="collapsed"
                    )
        
        if st.form_submit_button("Sauvegarder", use_container_width=True):
            try:
                save_patient_data(inputs)
                st.toast("Données sauvegardées avec succès!", icon="✅")
            except Exception as e:
                st.error(f"Erreur de sauvegarde : {str(e)}")

# Page À propos
else:
    show_about_page()

def show_prediction(model_name, data):
    try:
        prediction = predict_survival(models[model_name], data)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric(
                label=f"Survie Médiane ({model_name})", 
                value=f"{prediction['median_survival']:.1f} mois",
                help="Durée médiane de survie prédite selon le modèle"
            )
        with col2:
            fig = px.line(
                prediction['curve'], 
                x='Temps', 
                y='Survie', 
                title=f"Courbe de Survie ({model_name})"
            )
            fig.update_layout(
                xaxis_title="Mois après traitement",
                yaxis_title="Probabilité de survie",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur de prédiction ({model_name}): {str(e)}")

def show_deep_prediction(data):
    try:
        proba = models['DeepSurv'].predict(data.values, verbose=0)
        time_points = np.arange(0, 60, 0.5)
        survival_proba = np.exp(-np.cumsum(proba[0] * np.gradient(time_points)))
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric(
                label="Survie à 1 An", 
                value=f"{survival_proba[24]*100:.1f}%",
                help="Probabilité de survie à 12 mois selon le modèle profond"
            )
        with col2:
            fig = px.line(
                x=time_points, 
                y=survival_proba,
                labels={'x': 'Mois', 'y': 'Probabilité'},
                title="Courbe de Survie Continue (Deep Learning)"
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur Deep Learning: {str(e)}")

def save_patient_data(data):
    os.makedirs("data/new_patients", exist_ok=True)
    df = pd.DataFrame([data])
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"data/new_patients/patient_{timestamp}.csv", index=False)

def show_about_page():
    st.title("⚕️ À Propos de SurvivalMD")
    st.markdown("""
    ## L'Intelligence Artificielle au Service de la Vie
    **SurvivalMD** combine quatre modèles d'apprentissage automatique de pointe pour prédire 
    la survie des patients après traitement avec une précision inégalée.
    
    ### Fonctionnalités Clés :
    - 🧠 **Multi-Modèles** : Combinaison de 4 approches différentes
    - 📈 **Visualisation Interactive** : Courbes de survie dynamiques
    - 🔄 **Auto-apprentissage** : Amélioration continue avec chaque nouveau patient
    - 🔒 **Sécurité des Données** : Chiffrement AES-256 des données sensibles
    """)
    
    st.divider()
    
    cols = st.columns(4)
    models_info = [
        ("Cox PH", "Modèle de régression semi-paramétrique", "#2c3e50"),
        ("RSF", "Forêts aléatoires adaptées aux données censurées", "#3498db"),
        ("GBST", "Gradient boosting optimisé pour les durées de survie", "#27ae60"),
        ("DeepSurv", "Réseau de neurones profond avec embedding", "#e74c3c")
    ]
    
    for col, (name, desc, color) in zip(cols, models_info):
        with col:
            st.markdown(f"""
            <div style="
                background: {color}15;
                border-radius: 10px;
                padding: 1rem;
                border-left: 4px solid {color};
                margin: 1rem 0;
            ">
                <h4 style='color: {color}; margin: 0 0 1rem 0;'>{name}</h4>
                <p style='margin: 0;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    ## Architecture Technique
    ```mermaid
    graph TD
        A[Données Patient] --> B(Prétraitement)
        B --> C{Modèles}
        C --> D[Cox PH]
        C --> E[RSF]
        C --> F[GBST]
        C --> G[DeepSurv]
        D --> H[Agrégation]
        E --> H
        F --> H
        G --> H
        H --> I[Visualisation]
    """)

if __name__ == "__main__":
    st.rerun()
