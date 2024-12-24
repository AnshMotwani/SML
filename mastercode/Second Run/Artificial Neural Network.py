from google.colab import drive
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


drive.mount('/content/drive')


DRIVE_PATH = '/content/drive/My Drive/NBA_Data/'  
TRAIN_FILE = DRIVE_PATH + 'PCA_KM_with_New_Impact_Score_Till_2021.xlsx'
TEST_FILE = DRIVE_PATH + 'Filtered_PCA_KM_New_IS_2022.xlsx'

class NBADataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class NBAPredictor(nn.Module):
    def __init__(self, input_dim):
        super(NBAPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


train_data = pd.read_excel(TRAIN_FILE).fillna(0)
test_data = pd.read_excel(TEST_FILE).fillna(0)

features = [col for col in train_data.columns if col.startswith('PCA_') or col == "clusters"]
target = "Impact_Score_New"

X = train_data[features].values
y = train_data[target].values.reshape(-1, 1)
X_test = test_data[features].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


train_dataset = NBADataset(X_scaled, y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NBAPredictor(len(features)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)


print("Training Neural Network...")
epochs = 200
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), DRIVE_PATH + 'best_model.pth')

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')


model.load_state_dict(torch.load(DRIVE_PATH + 'best_model.pth'))
model.eval()


with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    y_train_pred = model(X_tensor).cpu().numpy()

print("\n=== Neural Network Training Metrics ===")
print(f"Training MSE: {mean_squared_error(y, y_train_pred):.4f}")
print(f"Training MAE: {mean_absolute_error(y, y_train_pred):.4f}")
print(f"Training RMSE: {np.sqrt(mean_squared_error(y, y_train_pred)):.4f}")
print(f"Training R²: {r2_score(y, y_train_pred):.4f}")


with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    test_predictions = model(X_test_tensor).cpu().numpy()

test_data['Predicted_Impact_Score'] = test_predictions


print("\n=== Neural Network Predicted Top 12 Players (2022) ===")
top_12_players = test_data.nlargest(12, 'Predicted_Impact_Score')
print(top_12_players[['PLAYER', 'Predicted_Impact_Score']])


test_data.to_csv(DRIVE_PATH + 'neural_network_predictions.csv', index=False)

