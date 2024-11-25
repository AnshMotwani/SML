import pandas as pd
from xgboost import XGBRegressor
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

# Initialize XGBoost regressor
xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)

# Train the model
print("Training XGBoost...")
xgb_model.fit(X_train, y_train)

# Evaluate on training data
y_train_pred = xgb_model.predict(X_train)
print("\n=== XGBoost Training Metrics ===")
print(f"Training MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"Training MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
print(f"Training RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print(f"Training R²: {r2_score(y_train, y_train_pred):.4f}")

# Predict on the test data
test_data['Predicted_Impact_Score'] = xgb_model.predict(X_test)

# Select top 12 players
top_12_players = test_data.nlargest(12, 'Predicted_Impact_Score')

print("\n=== XGBoost Predicted Top 12 Players (2022) ===")
print(top_12_players[['PLAYER', 'Predicted_Impact_Score']])

# Ranking-based and custom metrics
if 'Impact_Score_New' in test_data.columns:  # Check if actual scores are available
    y_test = test_data['Impact_Score_New']
    y_test_pred = test_data['Predicted_Impact_Score']

    # Spearman’s Rank Correlation
    rank_corr, _ = spearmanr(y_test, y_test_pred)
    print(f"Spearman’s Rank Correlation: {rank_corr:.4f}")

    # Precision@12
    def precision_at_k(y_true, y_pred, k):
        top_k_pred = np.argsort(y_pred)[-k:]
        top_k_true = np.argsort(y_true)[-k:]
        relevant = len(set(top_k_pred).intersection(set(top_k_true)))
        return relevant / k

    precision = precision_at_k(y_test, y_test_pred, k=12)
    print(f"Precision@12: {precision:.4f}")

    # Overlap@12
    def overlap_at_k(y_true, y_pred, k):
        top_k_pred = np.argsort(y_pred)[-k:]
        top_k_true = np.argsort(y_true)[-k:]
        overlap = len(set(top_k_pred).intersection(set(top_k_true)))
        return overlap / k

    overlap = overlap_at_k(y_test, y_test_pred, k=12)
    print(f"Overlap@12: {overlap:.4f}")

    # Team Impact Error
    def team_impact_error(y_true, y_pred, k):
        top_k_pred = np.argsort(y_pred)[-k:]
        top_k_true = np.argsort(y_true)[-k:]
        return abs(np.sum(y_true.iloc[top_k_true]) - np.sum(y_pred.iloc[top_k_pred]))

    team_error = team_impact_error(y_test, y_test_pred, k=12)
    print(f"Team Impact Error: {team_error:.4f}")