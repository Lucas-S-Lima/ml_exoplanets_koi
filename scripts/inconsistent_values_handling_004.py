import pandas as pd
import numpy as np

base_data = pd.read_csv("data/cumulative_koi_renamed.csv")

print("Shape original:", base_data.shape)
print("Colunas disponíveis:", base_data.columns.tolist())

# Remover CANDIDATE com zeros implausíveis
mask_inconsistente = (base_data["label"] == "CANDIDATE") & (
    (base_data["transit_signal_to_noise"] == 0)
    | (base_data["transit_depth_ppm"] == 0)
    | (base_data["insolation_flux_earth"] == 0)
)
print(f"\nRegistros CANDIDATE com zero implausível: {mask_inconsistente.sum()}")

base_data = base_data.drop(base_data[mask_inconsistente].index)
print(f"Shape após remover: {base_data.shape}")

# Colunas numéricas que precisam tratamento
colunas_numericas_com_nulo = [
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

# Preencher nulos com mediana POR CLASSE (melhor que global)
print("\nPreenchendo nulos com mediana por classe...")
for col in colunas_numericas_com_nulo:
    base_data[col] = base_data.groupby("label")[col].transform(lambda x: x.fillna(x.median()))

print("\nNulos restantes:")
print(base_data.isnull().sum().sum(), "valores nulos no total")

# Verificar duplicatas (considerando kepler_object_of_interest_name como identificador único)
print(f"\nDuplicatas completas: {base_data.duplicated().sum()}")
print(
    f"Duplicatas por kepler_object_of_interest_name: {base_data.duplicated(subset=['kepler_object_of_interest_name']).sum()}"
)

print(f"\nShape final: {base_data.shape}")
print("\nDistribuição de classes:")
print(base_data["label"].value_counts())

# Verificar se kepler_object_of_interest_name está preservado
print(
    f"\nKepler_object_of_interest_name nulos: {base_data['kepler_object_of_interest_name'].isnull().sum()}"
)

base_data.to_csv("data/cumulative_koi_treated.csv", index=False)
print("\nArquivo salvo: data/cumulative_koi_treated.csv")
