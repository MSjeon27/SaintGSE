#! /usr/bin/env python
# Code for Autoencoder training and prediction

import torch
import torch.nn as nn
import random, os
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--predict', action = 'store_true')
parser.add_argument('--dset', required=True)
parser.add_argument('--set_seed', default= [42], type=int, nargs='+')
parser.add_argument('--dim', default=[256], type=int, nargs='+')
parser.add_argument('--outdir', default=None)
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

current_dir = os.path.dirname(os.path.abspath(__file__))

seeds = opt.set_seed if isinstance(opt.set_seed, list) else [opt.set_seed]
dims = opt.dim if isinstance(opt.dim, list) else [opt.dim]

# Set split seed
SPLIT_SEED = 42

# numpy -> torch tensor
from torch.utils.data import DataLoader, TensorDataset
to_tensor = lambda a: torch.from_numpy(a).float()

# Fix seed for reproducibility
def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Preprocessing function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, test_size=0.2, val_size=0.1, random_state=SPLIT_SEED):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size/(1-test_size), random_state=random_state   # Validation size: 10 %
    )

    # Scaling
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

# For saving model
from pathlib import Path

def encode_in_batches(model, X_np, device, batch_size=1024):
    dl = DataLoader(TensorDataset(torch.from_numpy(X_np).float()),
                    batch_size=batch_size, shuffle=False)
    outs = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl:
            z = model.encoder(xb.to(device))
            outs.append(z.detach().cpu().numpy())
    return np.vstack(outs)  # (N, latent_dim)


# Autoencoder model structure
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=512, noise_std=0.05):
        super().__init__()
        self.noise_std = noise_std

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self._init_weights()

        self.register_buffer("scaler_mean", torch.zeros(input_dim))
        self.register_buffer("scaler_std", torch.ones(input_dim))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.training and self.noise_std and self.noise_std > 0:
            x = x + self.noise_std * torch.randn_like(x)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Autoencoder training process
def train_autoencoder(target_df, seed, latent_dim, input_dim, save_model_path=None):
    print(f"Training AE with {latent_dim} latent dimensions and seed {seed}.")
    fix_seed(seed)

    AEmodel = Autoencoder(input_dim, latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(AEmodel.parameters(), lr=1e-4, weight_decay=1e-4)
    num_epochs = 100
    batch_size = 256

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(target_df)

    # copy StandardScaler stats into model buffers
    AEmodel.scaler_mean.data.copy_(torch.from_numpy(scaler.mean_.astype(np.float32)))
    AEmodel.scaler_std.data.copy_(torch.from_numpy(scaler.scale_.astype(np.float32)))

    train_loader = DataLoader(TensorDataset(to_tensor(X_train)), batch_size=batch_size, shuffle=True)
    val_tensor = to_tensor(X_val).to(device)
    test_tensor = to_tensor(X_test).to(device)

    best_val = float('inf'); patience = 10; wait = 0
    best_state = None; best_epoch = 0
    test_loss = float('inf')

    alpha_l1 = 1e-4

    print("Autoencoder starts!")
    for epoch in range(num_epochs):
        AEmodel.train()
        running = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            z, xhat = AEmodel(xb)
            recon = criterion(xhat, xb)
            sparse = z.abs().mean()
            loss = recon + alpha_l1 * sparse

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += recon.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)

        AEmodel.eval()
        with torch.no_grad():
            _, vhat = AEmodel(val_tensor)
            val_loss = criterion(vhat, val_tensor).item()

        print(f'Epoch [{epoch+1:03d} | train {train_loss:.6f}], '
              f'Loss: {val_loss:.6f} (L1={alpha_l1})')

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            wait = 0
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in AEmodel.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_state is not None:
        AEmodel.load_state_dict(best_state)
    AEmodel.eval()
    with torch.no_grad():
        _, that = AEmodel(test_tensor)
        test_loss = criterion(that, test_tensor).item()
    print(f'Test MSE: {test_loss:.6f}')

    X_test_np = X_test
    Xhat_test_np = that.detach().cpu().numpy()

    se = (Xhat_test_np - X_test_np) ** 2
    mse_per_sample = se.mean(axis=1)
    mse_per_gene = se.mean(axis=0)
    mse_sample_mean = float(mse_per_sample.mean())
    mse_sample_std = float(mse_per_sample.std())
    mse_gene_mean = float(mse_per_gene.mean())
    mse_gene_std = float(mse_per_gene.std())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # save best AE model if path is given
    if save_model_path is not None:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        torch.save(AEmodel.state_dict(), save_model_path)
        print(f"Saved AE model to: {save_model_path}")

    return {
        "latent_dim": latent_dim,
        "seed": seed,
        "best_epoch": best_epoch,
        "val_mse": round(float(best_val), 4),
        "test_mse": round(float(test_loss), 4),
        "mse_sample_mean": round(mse_sample_mean, 4),
        "mse_sample_std": round(mse_sample_std, 4),
        "mse_gene_mean": round(mse_gene_mean, 4),
        "mse_gene_std": round(mse_gene_std, 4)
    }


# Get prediction results
if opt.predict:
    target_df = pd.read_csv(opt.dset, sep='\t')
    input_dim = target_df.shape[1] - 1
    results = []

    # where to save AE model
    ae_model_path = os.path.join(current_dir, 'datasets', 'AE_100cycle_model.pth')

    for dim in dims:
        for seed in seeds:
            save_path = ae_model_path if (dim == 256 and seed == 1) else None

            res = train_autoencoder(
                target_df,
                seed=seed,
                latent_dim=dim,
                input_dim=input_dim,
                save_model_path=save_path
            )
            results.append(res)

    df_out = pd.DataFrame(results)
    df_out.to_csv("AE_ablation_results.csv", index=False)
