import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from django.conf import settings


def load_treated_data(filename='exo_dataset.csv'):
    """Carrega o CSV tratado utilizando o BASE_DIR do Django."""

    data_dir = os.path.join(settings.BASE_DIR, 'data')
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset não encontrado em: {filepath}")
    
    return pd.read_csv(filepath)


def get_features_and_target(target_col='label', filename='exo_dataset.csv'):
    """Lê o dataset e retorna X, y e a lista de features prontas para o ML."""
    base_data = load_treated_data(filename)

    feature_cols = [
        col for col in base_data.columns
        if col not in ['kepler_object_of_interest_name', target_col]
    ]

    x_koi = base_data[feature_cols].to_numpy()
    y_koi = base_data[target_col].to_numpy()

    return x_koi, y_koi


def process_features(x_koi, base_data):
    """Aplica log e padronização (StandardScaler) nas features, retornando os dados escalados."""
    nomes_features = [
        "orbital_period_days",
        "transit_duration_hours",
        "transit_depth_ppm",
        "planet_radius_earth",
        "insolation_flux_earth",
        "equilibrium_temperature_k",
        "impact_parameter",
        "transit_signal_to_noise",
        "stellar_effective_temperature_k",
        "stellar_surface_gravity",
        "stellar_radius_solar",
        "kepler_magnitude",
    ]

    x_koi_df = pd.DataFrame(x_koi, columns=nomes_features)

    colunas_log = [
        "orbital_period_days",
        "transit_depth_ppm",
        "planet_radius_earth",
        "insolation_flux_earth",
    ]

    x_koi_log = x_koi_df.copy()
    x_koi_log[colunas_log] = np.log1p(x_koi_log[colunas_log])

    scaler = StandardScaler()
    mask_known = (base_data["label"] != "CANDIDATE").to_numpy()
    
    x_known = x_koi_log[mask_known]
    x_candidates = x_koi_log[~mask_known]

    x_known_scaled = scaler.fit_transform(x_known)
    x_candidates_scaled = scaler.transform(x_candidates)

    return x_known_scaled, x_candidates_scaled


def process_target(y_koi, base_data):
    """Codifica o target (label) usando LabelEncoder, retornando os dados codificados."""
    label_encoder_target = LabelEncoder()
    mask_known = (base_data['label'] != 'CANDIDATE').to_numpy()
    
    y_koi_known = y_koi[mask_known]
    y_known_encoded = label_encoder_target.fit_transform(y_koi_known)

    return y_known_encoded


def persist_training_data(x_train, x_test, y_train, y_test, filename='exoplanets_split.pkl'):
    """Serializa e salva os dados de treino e teste tratados em binário."""
    data_dir = os.path.join(settings.BASE_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, filename)

    data = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
    }

    joblib.dump(data, filepath)
    return filepath