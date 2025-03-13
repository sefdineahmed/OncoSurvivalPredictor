import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Fonction de perte personnalisée pour DeepSurv
def cox_loss(y_true, y_pred):
    """
    Fonction de perte de Cox pour l'entraînement du modèle DeepSurv.
    """
    event = tf.cast(y_true[:, 0], dtype=tf.float32)  # Événement (0 = censuré, 1 = décès)
    risk = y_pred[:, 0]  # Score de risque
    log_risk = tf.math.log(tf.cumsum(tf.exp(risk), reverse=True))
    loss = -tf.reduce_mean((risk - log_risk) * event)
    return loss

def load_model(model_path):
    """Charge un modèle pré-entraîné."""
    if model_path.endswith(".keras"):
        return keras.models.load_model(model_path, custom_objects={"cox_loss": cox_loss})
    return joblib.load(model_path)

def predict_survival(model, data):
    """Prédiction du temps de survie selon le modèle utilisé."""
    
    if hasattr(model, "predict_median"):  # CoxPHFitter (utilise `predict_median`)
        return model.predict_median(data).values[0]
    
    elif hasattr(model, "predict_survival_function"):  # CoxPHFitter (autre option)
        survival_function = model.predict_survival_function(data)
        return np.median(survival_function.index)  # Renvoie une estimation du temps médian

    elif hasattr(model, "predict"):  # RSF, GBST, et DeepSurv
        return model.predict(data)[0]
    
    else:
        raise ValueError("Le modèle fourni ne prend pas en charge la prédiction de survie.")
