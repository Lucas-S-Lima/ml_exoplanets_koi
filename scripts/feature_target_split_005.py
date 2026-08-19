import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


base_data = pd.read_csv('data/cumulative_koi_treated.csv')

feature_cols = [
    col for col in base_data.columns
    if col not in ['kepler_object_of_interest_name', 'label']
]
target_col = 'label'

x_koi = base_data[feature_cols].to_numpy()
y_koi = base_data[target_col].to_numpy()

print(f"Features: {list(feature_cols)}")
print(f"Target: {target_col}")
print(x_koi.shape, y_koi.shape)

