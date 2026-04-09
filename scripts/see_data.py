import sys
import pickle
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

models_dir = project_root / 'outputs' / 'saved_models'
with open(models_dir / 'preprocessed_data_combined.pkl', 'rb') as f:
    data = pickle.load(f)
    
print("="*60)
print("DATASET STRUCTURE OVERVIEW")
print("="*60)
for key, value in data.items():
    if isinstance(value, np.ndarray):
        print(f"{key:<20} | Shape: {str(value.shape):<18} | Dtype: {value.dtype}")
    elif isinstance(value, list):
        print(f"{key:<20} | Type: List, Length: {len(value)}")
    else:
        print(f"{key:<20} | Type: {type(value).__name__}, Value: {value}")

print("\n" + "="*60)
print("HOW YOUR CFRP MATERIAL IS REPRESENTED (EXAMPLE)")
print("="*60)

first_specimen = data['specimen_ids_train'][0]
first_x = data['X_train'][0]
first_y = data['y_train'][0]

print(f"Specimen ID:         {first_specimen}")
print(f"Remaining Life (y):  {first_y:.4f} (Scaled RUL)")
print(f"\nTime-Series Sensor Matrix (X) Shape: {first_x.shape}")
print(f"This represents {first_x.shape[0]} continuous time steps across {first_x.shape[1]} sensor features.\n")

print(f"--- First 3 features of the first 5 time steps for this sequence ---")
print(first_x[:5, :3])


# PS C:\Users\Gagandeep Singh\OneDrive\Desktop\CFRP_Composit_Materials_Analysis> python scripts/see_data.py 
# ============================================================
# DATASET STRUCTURE OVERVIEW
# ============================================================
# X_train              | Shape: (183, 10, 16)      | Dtype: float64
# X_val                | Shape: (61, 10, 16)       | Dtype: float64
# X_test               | Shape: (62, 10, 16)       | Dtype: float64
# y_train              | Shape: (183,)             | Dtype: float64
# y_val                | Shape: (61,)              | Dtype: float64
# y_test               | Shape: (62,)              | Dtype: float64
# specimen_ids_train   | Shape: (183,)             | Dtype: <U3
# specimen_ids_val     | Shape: (61,)              | Dtype: <U3
# specimen_ids_test    | Shape: (62,)              | Dtype: <U3
# input_feature_names  | Type: List, Length: 16
# sequence_length      | Type: int, Value: 10
# n_features           | Type: int, Value: 16

# ============================================================
# HOW YOUR CFRP MATERIAL IS REPRESENTED (EXAMPLE)
# ============================================================
# Specimen ID:         S19
# Remaining Life (y):  0.5640 (Scaled RUL)

# Time-Series Sensor Matrix (X) Shape: (10, 16)
# This represents 10 continuous time steps across 16 sensor features.

# --- First 3 features of the first 5 time steps for this sequence ---
# [[ 1.57701886  0.252164   -1.00207343]
#  [ 1.68049986  0.46929194 -1.43824427]
#  [ 1.84816633  0.57395452 -1.43515971]
#  [ 0.02730005 -0.45563334 -0.23052713]
#  [ 1.14931487  0.13378236 -1.09297486]]