import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import numpy as np

# Load the training and testing data
train_data = pd.read_excel("PCA_KM_with_New_Impact_Score_Till_2021.xlsx").fillna(0)
test_data = pd.read_excel("Filtered_PCA_KM_New_IS_2022.xlsx").fillna(0)

# Define features and target
features = [col for col in train_data.columns if col.startswith('PCA_') or col == "clusters"]
target = "Impact_Score_New"

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
player_ids = test_data['PLAYER']

# Initialize LightGBM regressor
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# Train the model
print("Training LightGBM...")
lgb_model.fit(X_train, y_train)

# Evaluate on training data
y_train_pred = lgb_model.predict(X_train)
print("\n=== LightGBM Training Metrics ===")
print(f"Training MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"Training MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
print(f"Training RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print(f"Training R²: {r2_score(y_train, y_train_pred):.4f}")

# Predict on the test data
test_data['Predicted_Impact_Score'] = lgb_model.predict(X_test)

# Select top 12 players
top_12_players = test_data.nlargest(12, 'Predicted_Impact_Score')

print("\n=== LightGBM Predicted Top 12 Players (2022) ===")
print(top_12_players[['PLAYER', 'Predicted_Impact_Score']])

# Additional metrics (same as XGBoost)
# ...