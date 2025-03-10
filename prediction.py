import numpy as np
import pickle as cp
import pandas as pa
from sklearn.neighbors import KNeighborsRegressor as KNR

# Load processed data and selected features
with open("processed_data.pkl", "rb") as f:
    processed_data = cp.load(f)
evecs = processed_data["evecs"]

# Fix evecs index (remove gene names from MultiIndex)
evecs.index = evecs.index.get_level_values(0)

training_growth_rates = processed_data["sc_gr"]

with open("feature_selection_results_carrera-corr.pkl", "rb") as f:
    results = cp.load(f)
selected_features = results["selected_features"]

# Align training data with selected eigengenes
training_data = processed_data["sc_gncorr"][selected_features]

# Load new data and align columns with evecs' gene IDs
new_data = pa.read_csv("ExprsDatarc.csv", sep=',')
gene_ids = evecs.index.tolist()
new_data = new_data.reindex(columns=gene_ids, fill_value=0)

# Project onto selected eigengenes
new_data_transformed = new_data.dot(evecs[selected_features])

# Debug: Ensure non-zero transformed data
print("Transformed data (sample):\n", new_data_transformed.head())
print("Unique rows:", new_data_transformed.drop_duplicates().shape[0])

# Train and predict
knn_model = KNR(n_neighbors=7, weights="distance")
knn_model.fit(training_data, training_growth_rates)
predicted_growth = knn_model.predict(new_data_transformed)

# Save predictions
pa.DataFrame({
    "Condition": new_data.index,
    "Predicted_Growth_Rate": predicted_growth
}).to_csv("growth_rate_predictions.csv", index=False)