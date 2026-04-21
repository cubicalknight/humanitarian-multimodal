# %%
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from data_processing import DataProcessing
from pathlib import Path

torch.manual_seed(42)

# %%
class ConfigClassifier:
    def __init__(self):
        self.dp = DataProcessing()
        self.data_dir = Path(__file__).resolve().parent.parent
        self.excel_path = Path(self.data_dir, "data/Airlink - UMichiagn - Data Collection - 9.8.2025.xlsx")
        
        self.batch_size = 64
        self.learning_rate = 0.001
        self.num_epochs = 50
        # self.model_save_path = Path(self.data_dir, "models/classifier_model.pth")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        self.input_data, self.target_data = self.dp.process_data(self.excel_path)
        self.input_dim = self.input_data.shape[1]
        self.num_classes = len(torch.unique(self.target_data))

    def split_data(self, split = 0.7):
        k = int(len(self.input_data) * split)
        indices = torch.randperm(len(self.input_data))[:k]
        self.x_train = self.input_data[indices]
        self.y_train = self.target_data[indices]
        self.x_test = self.input_data[~torch.isin(torch.arange(len(self.input_data)), indices)]
        self.y_test = self.target_data[~torch.isin(torch.arange(len(self.target_data)), indices)]

        self.tensor_train = torch.utils.data.TensorDataset(self.x_train, self.y_train.squeeze())
        
    
    def objective(self, trial, lr):
        batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32, 64])

        model = LogisticRegressionModel(input_dim=self.input_dim, num_classes=self.num_classes).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        data_loader = torch.utils.data.DataLoader(self.tensor_train, batch_size=batch_size, shuffle=True)

        optuna_epochs = 30
        for epoch in range(optuna_epochs):
            model.train()
            total_loss = 0.0
            
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = nn.CrossEntropyLoss()(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            model.eval()
            with torch.no_grad():
                logits = model(self.x_test.to(self.device))
                test_loss = nn.CrossEntropyLoss()(logits, self.y_test.to(self.device)).item()

            trial.report(test_loss, epoch)            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        return test_loss
            

    def train_model(self):
        # Placeholder for model training logic
        model = LogisticRegressionModel(input_dim=self.input_dim, num_classes=self.num_classes)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        data = torch.utils.data.TensorDataset(self.input_data, self.target_data.squeeze())
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True)

        # quick class weighting to address class imbalance
        class_counts = torch.bincount(self.target_data.squeeze())
        class_weights = len(self.target_data) / (len(class_counts) * class_counts.float())

        # NOTE explore focal loss as an alternative to cross-entropy with class weights for handling class imbalance

        losses = []
        for epoch in range(self.num_epochs):
            # model.train()
            total_loss = 0.0
            
            for batch_X, batch_y in data_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                # loss = MultiClassFocalLoss(gamma=0.25, weight=class_weights)(outputs, batch_y)
                loss = nn.CrossEntropyLoss(weight=class_weights)(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            avg_loss = total_loss / len(data_loader)
            losses.append(avg_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out


class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction=self.reduction)(inputs, targets)

        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


classifier = ConfigClassifier()
classifier.load_data()

x = classifier.input_data
y = classifier.target_data

uniq_classes = torch.unique(y)
num_classes = len(uniq_classes)

k = int(len(x) * 0.7)
indices = torch.randperm(len(x))[:k]
x_train = x[indices]
y_train = y[indices]

n = x.shape[1]
model = LogisticRegressionModel(input_dim=n, num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)

data = torch.utils.data.TensorDataset(x_train, y_train.squeeze())
data_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)

# quick class weighting to address class imbalance
class_counts = torch.bincount(y_train.squeeze())
class_weights = len(y_train) / (len(class_counts) * class_counts.float())

# NOTE explore focal loss as an alternative to cross-entropy with class weights for handling class imbalance

epochs = 150
losses = []
for epoch in range(epochs):
    total_loss = 0.0
    
    for batch_X, batch_y in data_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = MultiClassFocalLoss(gamma=0.25, weight=class_weights)(outputs, batch_y)
        # loss = nn.CrossEntropyLoss(weight=class_weights)(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# plot loss
sns.set_theme(style="darkgrid")
plt.figure(figsize=(8, 5))
sns.lineplot(x=range(1, epochs + 1), y=losses, color="#4c72b0")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# %%
# evalute model on test set
x_test = x[~torch.isin(torch.arange(len(x)), indices)]
y_test = y[~torch.isin(torch.arange(len(y)), indices)]

model.eval()
with torch.no_grad():
    logits = model(x_test)
    probs = torch.softmax(logits, dim=1)
    _, predicted = torch.max(logits.data, 1)

    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    y_true_np = y_test.squeeze().cpu().numpy()
    y_pred_np = predicted.cpu().numpy()
    probs_np = probs.cpu().numpy()

    classy = range(num_classes)
    plot_data = []

    for c in classy:
        # Create boolean masks using the numpy arrays
        is_class_c = (y_true_np == c)
        
        total_count = np.sum(is_class_c)
        correct_count = np.sum((y_pred_np == c) & is_class_c)
        
        plot_data.append({
            "Class": f"Class {c}",
            "Count": total_count,
            "Type": "Total Samples"
        })
        
        plot_data.append({
            "Class": f"Class {c}",
            "Count": correct_count,
            "Type": "Correct Predictions"
        })

    # Create a clean DataFrame for Seaborn
    df_plot = pl.DataFrame(plot_data)

    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    
    # Create grouped bar chart
    ax = sns.barplot(
        data=df_plot,
        x="Class",
        y="Count",
        hue="Type",
        palette=["#4c72b0", "#55a868"], # Blue for Total, Green for Correct
        alpha=0.9
    )

    # Add labels and title
    plt.title("Model Performance: Total vs Correct Predictions per Class", fontsize=14, pad=15)
    plt.xlabel("Class Label", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.legend(title="Metric")
    
    # Add number labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, padding=3) # type: ignore

    plt.tight_layout()
    plt.show()

    # --- PROBABILITY OUTPUT ---
    # Create a DataFrame showing Actual, Predicted, and Probabilities per class
    prob_cols = [f"Prob_Class_{int(c)}" for c in range(probs_np.shape[1])]
    df_results = pl.DataFrame(probs_np, schema=prob_cols)
    
    # Insert Actual and Predicted columns at the start
    df_results = df_results.with_columns([pl.Series('Predicted',y_pred_np), pl.Series('Actual', y_true_np)])

    print("\n--- Detailed Probabilities (First 5 Rows) ---")
    print(df_results.head())

    # Calculate overall accuracy
    acc = (y_pred_np == y_true_np).mean()
    print(f"\nOverall Test Accuracy: {acc:.2%}")

# %%