import pandas as pd

raw_path = "mlops_enzyme_stability/data/raw/"

# Load the raw data
df_train = pd.read_csv(f"{raw_path}train.csv", index_col="seq_id")
df_train_updates = pd.read_csv(f"{raw_path}train_updates_20220929.csv", index_col="seq_id")

# Remove rows with all features missing
all_features_nan = df_train_updates.isnull().all("columns")
drop_indices = df_train_updates[all_features_nan].index
df_train = df_train.drop(index=drop_indices)

# Correct transposed pH and tm values
swap_ph_tm_indices = df_train_updates[~all_features_nan].index
df_train.loc[swap_ph_tm_indices, ["pH", "tm"]] = df_train_updates.loc[swap_ph_tm_indices, ["pH", "tm"]]

# Save the updated training data
df_train.to_csv(f"{raw_path}train_fixed.csv")