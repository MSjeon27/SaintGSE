#! /usr/bin/env python
# Code for SaintGSE training
# This file has been modified from the original SAINT project.
# Original SAINT project: https://github.com/somepago/saint
# Copyright 2020 - present, Facebook, Inc
# Licensed under the Apache License, Version 2.0
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0

import torch
import argparse
import os
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()

parser.add_argument('--AEmodel', default=os.path.join(current_dir, 'datasets', 'AE_100cycle_model.pth'))
parser.add_argument('--predict')
parser.add_argument('--dset', default=os.path.join(current_dir, 'datasets', 'AE_enrichment.tsv'))
parser.add_argument('--pathway', default='all', nargs='+', type=str)
parser.add_argument('--vision_dset', action = 'store_true')
parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])

parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])
parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])

parser.add_argument('--lr', default=0.00001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--set_seed', default= 1 , type=int)
parser.add_argument('--dset_seed', default= 5 , type=int)

parser.add_argument('--pretrain', action = 'store_true')
parser.add_argument('--pretrain_epochs', default=50, type=int)
parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix'])
parser.add_argument('--pt_aug_lam', default=0.1, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)

parser.add_argument('--train_mask_prob', default=0, type=float)
parser.add_argument('--mask_prob', default=0, type=float)

parser.add_argument('--ssl_avail_y', default= 0, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)

parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])

opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

latent_df = pd.read_csv(opt.dset, sep='\t', index_col=0)


# Check for duplicate indices in the DataFrame
if latent_df.index.duplicated().any():
    print("Duplicate indices found. Resetting index.")
    latent_df = latent_df.reset_index(drop=True)
else:
    print("No duplicate indices found.")

# Get the pathway lists in enrichment results
# Set the target pathways
target_pathways = []
unique_pathways = []

# Get path of current file
opt.current_dir = current_dir

from models.pretrainmodel import SaintGSE, Autoencoder

with open(os.path.join(current_dir, 'datasets', 'pathway_list_in_DEG.txt'), 'r') as plf:
    for l in plf.readlines():
        pathway = l.rstrip()
        if pathway:
            unique_pathways.append(pathway)

print(f'Total {len(unique_pathways)} pathways detected!')

if opt.pathway == 'all':
    target_pathways = unique_pathways
else:
    if not all(path in unique_pathways for path in opt.pathway):
        missing_pathways = [path for path in opt.pathway if path not in unique_pathways]
        raise ValueError(f"The following elements are missing in the main list: {missing_pathways}")
    else:
        target_pathways = opt.pathway

if opt.predict:
    torch.manual_seed(opt.set_seed)
    np.random.seed(opt.set_seed)
    # Get dataframe for prediction
    pred_df = pd.read_csv(opt.predict, sep='\t', index_col='filename')
    
    # Preprocessing prediction dataframe
    input_dim = 36926
    latent_dim = 256
    AEmodel = Autoencoder(input_dim, latent_dim).to(device)
    AEmodel.load_state_dict(torch.load(opt.AEmodel, weights_only=True))
    AEmodel.eval()

    with torch.no_grad():
        pred_scaled = AEmodel.preprocess_pred(pred_df)

        pred_encoded = AEmodel.encoder(torch.tensor(pred_scaled, dtype=torch.float32).to(device))

        pred_encoded = pred_encoded.cpu().numpy()
        pred_encoded = pd.DataFrame(pred_encoded, index=pred_df.index)

    pathway = opt.pathway[0]

    path_rename = pathway.replace(' ', '_')
    
    # load model
    opt.SGmodel = os.path.join(current_dir, 'datasets', f'bestmodels/binary/{path_rename}/{opt.run_name}/saint_gse_model.pth')

    checkpoint = torch.load(opt.SGmodel, weights_only=False)
    opt_loaded = checkpoint['opt']

    for key, value in opt_loaded.items():
        setattr(opt, key, value)

    saint_gse_loaded = SaintGSE(device, latent_df, pathway, opt)
    saint_gse_loaded.saint.load_state_dict(checkpoint['model_state_dict'])
    saint_gse_loaded.eval()
    
    latent_df.drop(columns=['Enrichment Results'], inplace=True)
    
    # SHAP explaination
    import shap

    # SHAP GradientExplainer initialization
    sample_size = 6000
    indices = torch.randperm(len(latent_df))[:sample_size]
    sampled_latent_df = torch.tensor(latent_df.iloc[indices].values, dtype=torch.float32).to(device)
    shap_exp = shap.GradientExplainer(saint_gse_loaded, sampled_latent_df)

    for index, row in pred_encoded.iterrows():
        print(f'\ncurrent index: {index}')

        feature_names = latent_df.columns.to_list()
        num_features = len(feature_names)

        target_data = row

        target_tensor = torch.tensor(target_data.values, dtype=torch.float32).unsqueeze(0).to(device)
        
        shap_values = shap_exp.shap_values(target_tensor, nsamples=500)

        expected_value = saint_gse_loaded.forward(sampled_latent_df).mean().item()
        result_predict = saint_gse_loaded.predict(target_tensor)
        print(f'prediction_result: {result_predict}')
        
        shap_values_df = pd.DataFrame(shap_values, columns=feature_names).T

        shap_values_df.rename(columns={0: "Absolute Mean SHAP"}, inplace=True)
        shap_values_df = shap_values_df.sort_values(by='Absolute Mean SHAP', ascending=False)

        shap_values_df.to_csv(f'{index}_{path_rename}_shap_contributions.tsv', sep='\t')

        print(f"SHAP contributions saved to '{index}_{path_rename}_shap_contributions.tsv'\n")

else:
    # SaintGSE 인스턴스 생성
    for pathway in target_pathways:
        saint_gse = SaintGSE(device, latent_df, pathway, opt)

        # SaintGSE 실행
        saint_gse.run()

        # Save model for SaintGSE
        os.makedirs(saint_gse.modelsave_path, exist_ok=True)

        torch.save({
            'model_state_dict': saint_gse.best_saint.state_dict(),
            'opt': vars(saint_gse.opt)
            }, '%s/saint_gse_model.pth' % (saint_gse.modelsave_path))