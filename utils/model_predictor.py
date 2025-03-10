import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv

def predict_survival(model, data):
    """Convertit les données et fait la prédiction pour les modèles classiques"""
    try:
        # Préprocessing standard
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        
        if hasattr(model, 'predict_survival_function'):
            # Pour les modèles de survie
            surv_func = model.predict_survival_function(X_scaled)
            times = np.arange(1, 1000)
            proba = np.row_stack([fn(times) for fn in surv_func])
            return {
                'median_survival': np.median(times[np.argmax(proba < 0.5, axis=1)]),
                'curve': pd.DataFrame({'Temps': times, 'Survie': proba.mean(axis=0)})
            }
        else:
            # Pour les modèles de régression standard
            prediction = model.predict(X_scaled)
            return {
                'median_survival': np.median(prediction),
                'curve': pd.DataFrame({'Temps': prediction, 'Survie': np.linspace(1, 0, len(prediction))})
            }
    except Exception as e:
        raise RuntimeError(f"Erreur de prédiction: {str(e)}") from e
