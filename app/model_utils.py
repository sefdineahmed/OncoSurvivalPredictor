import os
import joblib
import tensorflow as tf

@tf.keras.saving.register_keras_serializable()
def cox_loss(y_true, y_pred):
    """
    Implémentation simplifiée de la loss de Cox.
    Remplacez cette implémentation par celle adaptée à votre problème.
    """
    # Exemple simplifié : à adapter pour votre cas d'utilisation
    return tf.reduce_mean(tf.square(y_true - y_pred))

def load_model(model_path):
    """
    Charge un modèle pré-entraîné depuis le fichier.
    Si l'extension est .keras ou .h5, le modèle est chargé via tf.keras.models.load_model()
    en utilisant custom_objects pour la fonction 'cox_loss'. Sinon, joblib.load() est utilisé.
    """
    _, ext = os.path.splitext(model_path)
    if ext in ['.keras', '.h5']:
        return tf.keras.models.load_model(model_path, custom_objects={"cox_loss": cox_loss})
    else:
        return joblib.load(model_path)

def predict_survival(model, data):
    """
    Prédit le temps de survie à partir des données fournies.
    """
    return model.predict(data)
