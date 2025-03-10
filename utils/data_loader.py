import yaml
import pandas as pd

def load_config(file_path):
    with open(file_path) as f:
        return yaml.safe_load(f)

def load_patient_data():
    return pd.concat([pd.read_csv(f) for f in glob.glob("data/new_patients/*.csv")])
