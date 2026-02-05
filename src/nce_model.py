# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import polars as pl
import optuna
import os
from dataclasses import dataclass

import data_processing as dp

from matplotlib import pyplot as plt
import seaborn as sns

# TODO revise constant variables and paths
# DIR = os.getcwd()

# %%
# @dataclass
class NoiseGeneration:
    def __init__(self, data: torch.Tensor, inflation: float = 1.5):
        mu = torch.mean(data, dim=0)
        cov = torch.from_numpy(np.cov(data.numpy(), rowvar=False)).float()

        # read about how to do this
        cov_numpy = np.cov(data.numpy(), rowvar=False)
        cov = torch.from_numpy(cov_numpy).float()

        epsilon = 1e-5 * torch.eye(cov.shape[0])
        cov_noise = (inflation ** 2) * (cov + epsilon)

        self.dist = torch.distributions.MultivariateNormal(mu, cov_noise)

    def sample(self, n: int) -> torch.Tensor:
        return self.dist.sample((n,))
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(x)

# @dataclass
class NoiseContrastiveEstimation(nn.Module):
    def __init__(self, model: nn.Module, noise_gen: NoiseGeneration, noise_multiplier: int = 1):
        super().__init__()
        self.model = model
        self.noise_gen = noise_gen
        self.noise_multiplier = noise_multiplier
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        noise_x = self.noise_gen.sample(batch_size * self.noise_multiplier)

        score_obs = self.model(x).squeeze()
        score_noise = self.model(noise_x).squeeze()

        log_q_obs = self.noise_gen.log_prob(x).detach()
        log_q_noise = self.noise_gen.log_prob(noise_x).detach()

        logits_obs = score_obs - log_q_obs
        logits_noise = score_noise - log_q_noise

        loss_obs = self.bce(logits_obs, torch.ones_like(logits_obs))
        loss_noise = self.bce(logits_noise, torch.zeros_like(logits_noise))

        loss = loss_obs + loss_noise

        return loss
    

class TrainNCE:
    def __init__(self, train_data: torch.Tensor, test_data: torch.Tensor):
        self.train_data = train_data
        self.test_data = test_data
        self.input_dim = train_data.shape[1]

    def build_model(self, **params) -> nn.Module:
        layers = []
        dim = params['input_dim']
        hidden_sizes = params['hidden_sizes']

        for size in hidden_sizes:
            layers.append(nn.Linear(dim, size))
            layers.append(nn.ReLU())
            dim = size

        layers.append(nn.Linear(dim, 1))

        return nn.Sequential(*layers)
    
    def objective(self,
                trial: optuna.trial.Trial,
                lr: float = 1e-3,
                dr1: float = 0.9, 
                dr2: float = 0.999, 
                epsilon: float = 1e-8) -> float:
        
        n_layers = trial.suggest_int('n_layers', 1, 4)
        hidden_sizes = [trial.suggest_categorical(f'n_units_l{i}', [16, 32, 64, 128, 256]) for i in range(n_layers)]

        # TODO address reward hacking issues with noise multiplier
        noise_multiplier = trial.suggest_int('noise_multiplier', 1, 5)
        # inflation = trial.suggest_float('inflation', 1.0, 5.0)

        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

        input_dim = self.train_data.shape[1]

        model = self.build_model(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            n_layers=n_layers
        )

        noise_gen = NoiseGeneration(self.train_data, inflation=1.5)
        nce_model = NoiseContrastiveEstimation(model, noise_gen, noise_multiplier=noise_multiplier)
        optimizer = optim.Adam(nce_model.parameters(), lr=lr, betas=(dr1, dr2), eps=epsilon)

        dataset = torch.utils.data.TensorDataset(self.train_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        num_epochs = 30
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in dataloader:
                obs_batch = batch[0]
                
                optimizer.zero_grad()
                loss = nce_model(obs_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            accuracy = total_loss / len(dataloader)

            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        nce_model.eval()
        with torch.no_grad():
            test_loss = nce_model(self.test_data).item()
            
        return test_loss
    
    def run_optimization(self, n_trials: int = 50) -> optuna.study.Study:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )

        study.optimize(self.objective, n_trials=n_trials)

        return study
    
    def train_final_model(self, best_params: dict) -> nn.Module:
        if 'hidden_sizes' in best_params:
            hidden_sizes = best_params['hidden_sizes']
        else:
            n_layers = best_params['n_layers']
            hidden_sizes = [best_params[f'n_units_l{i}'] for i in range(n_layers)]

        model = self.build_model(
            input_dim=self.input_dim,
            hidden_sizes=hidden_sizes,
            n_layers=best_params['n_layers']
        )

        noise_gen = NoiseGeneration(self.train_data, inflation=1.5) #best_params['inflation'])
        nce_model = NoiseContrastiveEstimation(model, noise_gen, noise_multiplier=best_params['noise_multiplier'])
        optimizer = optim.Adam(
            nce_model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8
            # weight_decay=1e-3
        )

        batch_size = best_params['batch_size']
        dataset = torch.utils.data.TensorDataset(self.train_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        num_epochs = 100
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in dataloader:
                obs_batch = batch[0]
                
                optimizer.zero_grad()
                loss = nce_model(obs_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

        return nce_model

    # TODO add docstrings and fix dataclass structures
    def test_model_performance_scatter(self, model):
        """Visualizes how well the model discriminates between real, test, and noise data."""
        model.eval()
        
        '''
        Temporarily regenerating noise generator here for clarity
        Need to refactor to pull directly from trained model object
        '''
        noise_gen = NoiseGeneration(self.train_data, inflation=1.5) # Use same inflation as training!

        with torch.no_grad():
            # Score Training Data
            score_real = model(self.train_data).squeeze()
            logq_real = noise_gen.log_prob(self.train_data)
            # The Correct NCE Probability: Model Score - Noise Log Prob
            real_probs = torch.sigmoid(score_real - logq_real).numpy()
            
            # Score Test Data
            score_test = model(self.test_data).squeeze()
            logq_test = noise_gen.log_prob(self.test_data)
            test_probs = torch.sigmoid(score_test - logq_test).numpy()
            
            # Score Random Noise
            noise_tensor = noise_gen.sample(len(self.test_data)) # Sample from the act noise dist
            score_noise = model(noise_tensor).squeeze()
            logq_noise = noise_gen.log_prob(noise_tensor)
            noise_probs = torch.sigmoid(score_noise - logq_noise).numpy()

        # Fancy printing :))
        print(f"{'Dataset':<15} | {'Mean Prob':<10} | {'Max Prob':<10}")
        print("-" * 45)
        print(f"{'Training':<15} | {np.mean(real_probs):.6f} | {np.max(real_probs):.6f}")
        print(f"{'Testing':<15} | {np.mean(test_probs):.6f} | {np.max(test_probs):.6f}")
        print(f"{'Noise':<15} | {np.mean(noise_probs):.6f} | {np.max(noise_probs):.6f}")

        sns.set_theme(style="darkgrid")
        colors = sns.color_palette('deep')
        plt.figure()
        sns.kdeplot(real_probs, color=colors[2], fill=True, alpha=0.5, label='Training Set')
        sns.kdeplot(test_probs, color=colors[1], fill=True, alpha=0.5, label='Testing Set')
        sns.kdeplot(noise_probs, color=colors[3], fill=True, alpha=0.5, label='Random Noise')

        # Old scatter plot code
        # plt.scatter(range(len(real_probs)), real_probs, c='green', alpha=0.4, s=15, label='Training Set')
        # plt.scatter(range(len(test_probs)), test_probs, c='orange', alpha=0.6, s=15, marker='x', label='Testing Set')
        # plt.scatter(range(len(noise_probs)), noise_probs, c='red', alpha=0.3, s=15, marker='v', label='Random Noise')

        # Remember golden rule, no titles
        plt.xlabel("Probability Score")
        # plt.ylabel("Density")
        # plt.yscale('log')
        plt.legend(loc='upper left')
        # plt.grid(True, alpha=0.3)
        plt.savefig("figures/nce_model_performance.pdf", dpi=300)
        plt.show()

if __name__ == "__main__":
    data_handler = dp.DataProcessing()
    complete_tensor_shipping = data_handler.process_data(data_handler.excel_path)
    
    # TODO vary seed for different validation
    torch.manual_seed(42)

    # 70/30 train/test split
    k = int(len(complete_tensor_shipping) * 0.7)
    indices = torch.randperm(len(complete_tensor_shipping))[:k]
    tensor_shipping = complete_tensor_shipping[indices]

    test_set = complete_tensor_shipping[~torch.isin(torch.arange(len(complete_tensor_shipping)), indices)]

    trainer = TrainNCE(tensor_shipping, test_set)
    study = trainer.run_optimization(n_trials=50)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_nce_model = trainer.train_final_model(trial.params)
    # best_nce_model = trainer.train_final_model({
    #     'n_layers': 3,
    #     'hidden_sizes': [256, 256, 16],
    #     'noise_multiplier': 4,
    #     'inflation': 1.5,
    #     'batch_size': 16
    # })
    trainer.test_model_performance_scatter(best_nce_model.model)
