import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from feature_target_split_005 import x_koi, base_data, y_koi

# Nomes das 12 features, na mesma ordem em que foram montadas no
# feature_target_split_005 (x_koi é um numpy array, sem nomes de coluna,
# por isso precisamos reconstruir o DataFrame aqui para indexar por nome).
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

x_koi = pd.DataFrame(x_koi, columns=nomes_features)

colunas_log = [
    "orbital_period_days",
    "transit_depth_ppm",
    "planet_radius_earth",
    "insolation_flux_earth",
]

print("\n" + "=" * 60)
print("ASSIMETRIA (skew) ANTES DO LOG")
print("=" * 60)
print(x_koi[colunas_log].skew())

x_koi_log = x_koi.copy()
x_koi_log[colunas_log] = np.log1p(x_koi_log[colunas_log])

print("\n" + "=" * 60)
print("ASSIMETRIA (skew) DEPOIS DO LOG")
print("=" * 60)
print(x_koi_log[colunas_log].skew())

scaler = StandardScaler()

mask_known = (base_data["label"] != "CANDIDATE").to_numpy()
x_known = x_koi_log[mask_known]
x_candidates = x_koi_log[~mask_known]

x_known_scaled = scaler.fit_transform(x_known)
x_candidates_scaled = scaler.transform(x_candidates)

print("\n" + "=" * 60)
print("RESULTADO DA PADRONIZAÇÃO")
print("=" * 60)
print(x_known_scaled)
print(x_known.shape, x_candidates.shape)  # a soma dos dois deve bater com o total original
