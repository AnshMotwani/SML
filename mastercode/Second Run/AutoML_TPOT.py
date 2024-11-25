from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

# Load the training and testing data
train_data = pd.read_excel("PCA_KM_with_New_Impact_Score_Till_2021.xlsx").fillna(0)
test_data = pd.read_excel("Filtered_PCA_KM_New_IS_2022.xlsx").fillna(0)

# Define features and target
features = [col for col in train_data.columns if col.startswith('PCA_') or col == "clusters"]
target = "Impact_Score_New"

X = train_data[features]
y = train_data[target]
X_test = test_data[features]
player_ids = test_data['PLAYER']

# Initialize TPOT AutoML
tpot = TPOTRegressor(
    generations=5,
    population_size=20,
    cv=3,
    random_state=42,
    verbosity=2
)

# Train the TPOT AutoML
print("Training TPOT AutoML...")
tpot.fit(X, y)

# Evaluate on training data
y_train_pred = tpot.predict(X)
print("\n=== TPOT AutoML Training Metrics ===")
print(f"Training MSE: {mean_squared_error(y, y_train_pred):.4f}")
print(f"Training MAE: {mean_absolute_error(y, y_train_pred):.4f}")
print(f"Training RMSE: {np.sqrt(mean_squared_error(y, y_train_pred)):.4f}")
print(f"Training R²: {r2_score(y, y_train_pred):.4f}")

# Predict on the test data
test_data['Predicted_Impact_Score'] = tpot.predict(X_test)

# Select top 12 players
top_12_players = test_data.nlargest(12, 'Predicted_Impact_Score')

print("\n=== TPOT AutoML Predicted Top 12 Players (2022) ===")
print(top_12_players[['PLAYER', 'Predicted_Impact_Score']])

# Additional metrics (same as XGBoost)
# ...