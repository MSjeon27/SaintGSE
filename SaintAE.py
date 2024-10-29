#! /usr/bin/env python
# Code for Autoencoder training and prediction

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--predict', action = 'store_true')
parser.add_argument('--dset', required=True)
parser.add_argument('--set_seed', default= 1 , type=int)
parser.add_argument('--model')

opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

input_dim = 36926
latent_dim = 256

if not opt.predict:
    # Autoencoder training structure
    class Autoencoder(nn.Module):
        def __init__(
                self,
                input_dim,
                latent_dim,
            ):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 512),
                nn.ReLU(),
                nn.Linear(512, input_dim),
                nn.Sigmoid()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded
        
        def preprocess_data(self, df, test_size=0.2, random_state=42):
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

            return X_train, X_test, y_train, y_test, scaler
    AEmodel = Autoencoder(input_dim, latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(AEmodel.parameters(), lr=1e-4)
    num_epochs = 100
    batch_size = 256

    target_df = pd.read_csv(opt.dset, sep='\t')
    columns_to_remove = ['Unnamed: 0', 'Unnamed: 0.1', 'Expressed_Genes']
    target_df.drop(columns=columns_to_remove, inplace=True)

    torch.manual_seed(opt.set_seed)
    
    X_train, X_test, y_train, y_test, scaler = AEmodel.preprocess_data(target_df)

    print("Autoencoder starts!")

    # Early stopping patience 설정
    early_stopping_patience = 30
    no_improvement_count = 0
    best_loss = float('inf')  # Initialize best loss as infinity
    best_model_wts = AEmodel.state_dict().copy()  # Initialize with the initial weights

    for epoch in range(num_epochs):
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        AEmodel.train()
        for data in train_loader:
            inputs = data[0].to(device)
            encoded, decoded = AEmodel(inputs)

            loss = criterion(decoded, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # Evaluate on test data
        AEmodel.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            encoded_test, decoded_test = AEmodel(X_test_tensor)

            test_loss = criterion(decoded_test, X_test_tensor)
            
            print(f'Test Loss: {test_loss.item():.4f}')
            
            # Early stopping check
            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                best_model_wts = AEmodel.state_dict().copy()
                print("Best model updated")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= early_stopping_patience:
                    print("Early stopping triggered after no improvement for 30 epochs")
                    break

    # Load the best model weights
    AEmodel.load_state_dict(best_model_wts)

    encoder_weights = AEmodel.encoder[0].weight.data.cpu().numpy()

    # Get gene list of weight 0
    zero_weight_genes = np.where(np.all(encoder_weights == 0, axis=1))[0]

    gene_names = target_df.columns[:-1]  # Assuming the gene names are in the columns
    with open('zero_weight_genes.txt', 'w') as f:
        for gene_index in zero_weight_genes:
            f.write(f"{gene_names[gene_index]}\n")

    print(f"Number of genes with zero weights: {len(zero_weight_genes)}")
    print(f"Gene names with zero weights have been saved to zero_weight_genes.txt")

    # Save the best model
    torch.save(AEmodel.state_dict(), 'OrthoAE_whole_100_model.pth')
    
    # Encode the dataset using the final model
    with torch.no_grad():
        # Encode X_train and X_test respectively
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        
        encoded_train, _ = AEmodel(X_train_tensor)
        encoded_test, _ = AEmodel(X_test_tensor)
        
        # Move from GPU to CPU and convert to numpy
        encoded_train = encoded_train.cpu().numpy()
        encoded_test = encoded_test.cpu().numpy()

    # Create encoded DataFrames
    column_names = [f'latent geneset {i}' for i in range(encoded_train.shape[1])]

    train_df = pd.DataFrame(encoded_train, columns=column_names)
    train_df['Enrichment Results'] = y_train

    test_df = pd.DataFrame(encoded_test, columns=column_names)
    test_df['Enrichment Results'] = y_test

    # Combine the final encoded DataFrames
    encoded_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    # Check the result
    print(encoded_df)

    encoded_df.to_csv('AE_enrichment_final.tsv', sep='\t')


elif opt.predict:
    torch.manual_seed(opt.set_seed)
    np.random.seed(opt.set_seed)

    # Get reference dataset
    whole_data = pd.read_csv(opt.dset, sep='\t')
    whole_data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Expressed_Genes'], inplace=True)
    print(whole_data)

    def preprocess_data(df, test_size=0.2, random_state=42):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test, scaler

    X_train, X_test, y_train, y_test, scaler = preprocess_data(whole_data)

    whole_preprocessed = np.concatenate((X_train, X_test), axis=0)

    # Autoencoder structure
    class Autoencoder(nn.Module):
        def __init__(
                self,
                input_dim,
                latent_dim,
            ):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 512),
                nn.ReLU(),
                nn.Linear(512, input_dim),
                nn.Sigmoid()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            return encoded
        
    AEmodel = Autoencoder(input_dim, latent_dim).to(device)
    AEmodel.load_state_dict(torch.load(opt.model, weights_only=True), strict=False)
    AEmodel.eval()


    # Calculate shap contribution through batch
    batch_size = 150
    n_batches = len(whole_preprocessed) // batch_size

    for i in range(n_batches):
        sampled_data = whole_preprocessed[i*batch_size:(i+1)*batch_size]

        sampled_torch = torch.tensor(sampled_data, dtype=torch.float32, requires_grad=True).to(device)

        explainer = shap.GradientExplainer(AEmodel, sampled_torch)

        # SHAP calculation
        shap_values = explainer.shap_values(sampled_torch)

        import pickle
        with open(f'shap_values_AE_batch_{i}.pkl', 'wb') as f:
            pickle.dump(shap_values, f)
        
        del sampled_data, sampled_torch, explainer, shap_values
        torch.cuda.empty_cache()