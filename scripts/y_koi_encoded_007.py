import numpy as np
from sklearn.preprocessing import LabelEncoder
from feature_target_split_005 import y_koi, base_data

label_encoder_target = LabelEncoder()

mask_known = base_data["label"] != "CANDIDATE"
y_koi_known = y_koi[mask_known]
y_candidates = y_koi[~mask_known]

y_known_encoded = label_encoder_target.fit_transform(y_koi_known)

print(y_known_encoded)
print(np.unique(y_known_encoded))  # deve mostrar só [0 1] — se aparecer [0 1 2], o CANDIDATE entrou
print(label_encoder_target.classes_)
