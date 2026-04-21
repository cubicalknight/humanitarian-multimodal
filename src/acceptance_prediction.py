# %%
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 1000
features = np.random.rand(n_samples, 2)

true_weights = np.array([1.5, 1.5])
true_logits = np.dot(features, true_weights) - 0.5
p_true = 1 / (1 + np.exp(-true_logits))
y_true = np.random.binomial(1, p_true).astype(np.float32)

c_true = 0.4
observed_mask = (y_true == 1) & (np.random.rand(n_samples) < c_true)
s_labels = observed_mask.astype(np.float32)

class CarrierAcceptProbability(nn.Module):
    def __init__(self, input_dim):
        super(CarrierAcceptProbability, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        pred = torch.sigmoid(self.linear(x))
        return pred
    
def train_step(X, realizations):
    m = CarrierAcceptProbability(X.shape[1])
    opt = optim.Adam(m.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for _ in range(1000):
        opt.zero_grad()
        loss = criterion(m(X).squeeze(), realizations)
        loss.backward()
        opt.step()
    return m

X_tensor = torch.tensor(features, dtype=torch.float32)
oracle_model = train_step(X_tensor, torch.tensor(y_true, dtype=torch.float32))
pu_model = train_step(X_tensor, torch.tensor(s_labels, dtype=torch.float32))

with torch.no_grad():
    c_est = pu_model(X_tensor[s_labels == 1]).mean().item()
    
    p_oracle = oracle_model(X_tensor).numpy().flatten()
    p_pu_raw = pu_model(X_tensor).numpy().flatten()
    p_pu_calibrated = np.clip(p_pu_raw / c_est, 0, 1)

print(f"True Labeling Frequency: {c_true:.2f}")
print(f"Estimated Labeling Frequency: {c_est:.2f}")

# if __name__ == "__main__":
#     n_inputs = 10  # Example input dimension
#     model = CarrierAcceptProbability(n_inputs)

#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     def train_step(features, realizations):
#         model.train()
#         optimizer.zero_grad()

#         p_hat = model(features).squeeze()
#         loss = criterion(p_hat, realizations)

#         loss.backward()
#         optimizer.step()

#         return loss.item()