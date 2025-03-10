import joblib

def load_model(model_path):
    """Charge un modèle pré-entraîné à partir du fichier."""
    return joblib.load(model_path)

def predict_survival(model, data):
    """Prédiction du temps de survie pour un patient donné."""
    return model.predict(data)
