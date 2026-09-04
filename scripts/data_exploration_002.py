import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Exploração dos dados
base_data = pd.read_csv("data/cumulative_koi_renamed.csv")

# Exploração das estatísticas
description = base_data.describe()

# Contagem de registros
registers = np.unique(base_data["label"], return_counts=True)

# Gráfico de contagem de registros
# counter = sns.countplot(x=base_data['label'])

# Histogramas
hist_stellar_surface_gravity = plt.hist(x=base_data["stellar_surface_gravity"])

hist_stellar_radius_solar = plt.hist(x=base_data["stellar_radius_solar"])

hist_equilibrium_temperature_k = plt.hist(x=base_data["equilibrium_temperature_k"])
plt.show()

# Amostra aleatória de N registros
sample_data = base_data.sample(n=100, random_state=42)

# Gráficos de dispersão
graphic = px.scatter_matrix(
    sample_data, dimensions=["orbital_period_days", "transit_duration_hours"], color="label"
)
graphic.update_traces(marker=dict(size=10, opacity=0.8))
# graphic.show()
