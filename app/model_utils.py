import joblib
import numpy as np

def load_model(model_path):
    """Charge un modèle pré-entraîné à partir du fichier."""
    if model_path.endswith(".keras"):
        from tensorflow import keras
        return keras.models.load_model(model_path)
    return joblib.load(model_path)

def predict_survival(model, data):
    """Prédiction du temps de survie selon le modèle utilisé."""
    
    if hasattr(model, "predict_median"):  # CoxPHFitter
        return model.predict_median(data).values[0]
    
    elif hasattr(model, "predict_survival_function"):  # CoxPHFitter (autre option)
        survival_function = model.predict_survival_function(data)
        return np.median(survival_function.index)  # Renvoie une estimation du temps médian

    elif hasattr(model, "predict"):  # Random Survival Forest et autres modèles scikit-learn
        return model.predict(data)[0]
    
    else:
        raise ValueError("Le modèle fourni ne prend pas en charge la prédiction de survie.")
