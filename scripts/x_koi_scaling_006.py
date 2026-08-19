from sklearn.preprocessing import StandardScaler
from feature_target_split_005 import x_koi, base_data, y_koi

scaler = StandardScaler()

mask_known = base_data['label'] != 'CANDIDATE'
x_known = x_koi[mask_known]
x_candidates = x_koi[~mask_known]

x_known_scaled = scaler.fit_transform(x_known)     
x_candidates_scaled = scaler.transform(x_candidates) 

print(x_known_scaled)

print(x_known.shape, x_candidates.shape)    # a soma dos dois deve bater com o total original