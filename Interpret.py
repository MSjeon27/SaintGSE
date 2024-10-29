#! /usr/bin/env python
# Calculate the total contribution of SaintGSE

import argparse
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()


parser.add_argument('-d', '--deg', required=True)
parser.add_argument('-p', '--pathway', required=True)

opt = parser.parse_args()

# Get DEG dataset
deg_df = pd.read_csv(opt.deg, sep='\t', index_col='filename')

fc_dict = {}

# Get significant genes in foldchange results
for index, row in deg_df.iterrows():
    non_zero_columns = row[row != 0].index.tolist()
    fc_dict[index] = non_zero_columns

path_rename = opt.pathway.replace(' ', '_')

# Generate final shap result
for file, gene in fc_dict.items():
    final_contribution = {key: 0 for key in list(gene)}
    print(f'current_index: {file}')
    GSEshap_file = f'{file}_{path_rename}_shap_contributions.tsv'

    GSEdf = pd.read_csv(GSEshap_file, sep='\t', index_col='Unnamed: 0')
    latent_dict = {}

    for index, row in GSEdf.iterrows():
        value = float(row["Absolute Mean SHAP"])
        key_name = f"shap_values_latent_dim_{index.split(' ')[-1]}.csv"
        latent_dict[key_name] = value

    for latent_dim, latent_shap in latent_dict.items():
        AEshap_file = os.path.join(current_dir, 'datasets/AEshap', latent_dim)

        AEshap_df = pd.read_csv(AEshap_file, index_col='Unnamed: 0')
        AE_temp_dict = {}
        latent_shap_sum = float(AEshap_df['Absolute Mean SHAP'].sum())
        for index, row in AEshap_df.iterrows():
            key_name = index
            if key_name in list(gene):
                value = float(row["Absolute Mean SHAP"])
                normalized_value = value / latent_shap_sum
                AE_temp_dict[key_name] = normalized_value

        AE_temp_calculated_dict = {key: value * latent_shap for key, value in AE_temp_dict.items()}

        for key, value in AE_temp_calculated_dict.items():
            final_contribution[key] += value

    sorted_keys = sorted(final_contribution, key=final_contribution.get, reverse=True)

    df = pd.DataFrame(list(final_contribution.items()), columns=['Gene', 'Contribution'])

    df['abs_Contribution'] = df['Contribution'].abs()
    df = df.sort_values(by='abs_Contribution', ascending=False)

    df = df.drop(columns=['abs_Contribution'])

    df.to_csv(f'{file}_{path_rename}_significant_gene_shap.csv', index=False)

    