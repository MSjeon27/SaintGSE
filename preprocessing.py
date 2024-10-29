#! /usr/bin/env python
# Preprocess the format such as process of 'Convert MGI gene ID to human ensemble gene ID'

import pandas as pd
import argparse
import gseapy as gp
from collections import defaultdict
import os
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()

parser.add_argument('--query_fc', required=True)
parser.add_argument('--MGIdb', default = os.path.join(current_dir, 'datasets', 'MGI_Gene_Model_Coord.tsv'))
parser.add_argument('--ref', required=True)
parser.add_argument('--org', default = 'mouse', choices = ['mouse','human'])
parser.add_argument('--out', default='Preprocessed_fc.tsv')

opt = parser.parse_args()

query_fc = pd.read_csv(opt.query_fc, sep='\t')

if opt.org == 'mouse':
    #Get ID information from MGI database
    MGI_db = pd.read_csv(opt.MGIdb, sep='\t')
    class GeneID_MGI:
        def __init__(self, species_db, query_fc):
            super().__init__()
            self.gene_dict = self._create_gene_dict(species_db)
            self.query_dict = self._filter_genes(query_fc)

        def _create_gene_dict(self, db):
            gene_dict = {}
            for _, row in db.iterrows():
                marker_symbol = row['3. marker symbol']
                ensemble_gene_id = row['11. Ensembl gene id']
                if pd.notna(ensemble_gene_id): # filtering
                    gene_dict[marker_symbol] = ensemble_gene_id
            return gene_dict
        
        def _filter_genes(self, fcdata):
            filteredfc_dict = {}
            filtered_data = fcdata[fcdata['q_value'] <= 0.05]
            for _, row in filtered_data.iterrows():
                gene_id = row['gene_id']
                if gene_id in self.gene_dict:
                    ensembl_id = self.gene_dict[gene_id]
                    log2_fold_change = row['log2(fold_change)']
                    if log2_fold_change != float('-inf') and log2_fold_change != float('inf'):
                        filteredfc_dict[ensembl_id] = log2_fold_change
            print(f'Total filtered mouse genes with log2fc: {len(filteredfc_dict.keys())}')
            return filteredfc_dict

    MGI_idclass = GeneID_MGI(MGI_db, query_fc)
    filteredfc_dict = MGI_idclass.query_dict

    # Get mouse to human database from Biomart
    from gseapy import Biomart

    bm = Biomart()
    h2m = bm.query(dataset='mmusculus_gene_ensembl',
                attributes=['ensembl_gene_id',
                            'external_gene_name',
                            'hsapiens_homolog_ensembl_gene',
                            'hsapiens_homolog_associated_gene_name']
                )

    filtered_df = h2m[h2m['hsapiens_homolog_ensembl_gene'].notna() & h2m['hsapiens_homolog_associated_gene_name'].notna()]
    matched_genes = filtered_df[filtered_df['ensembl_gene_id'].isin(filteredfc_dict.keys())]

    ortholog_fc = defaultdict(list)
    for index, row in matched_genes.iterrows():
        gene_name = row['hsapiens_homolog_associated_gene_name']
        fc_value = filteredfc_dict[row['ensembl_gene_id']]
        ortholog_fc[gene_name].append(fc_value)

    # Process duplicated human genes
    final_fc_dict = {gene: sum(values) / len(values) for gene, values in ortholog_fc.items()}

else:
    class GeneID_human:
        def __init__(self, query_fc):
            super().__init__()
            self.query_dict = self._filter_genes(query_fc)

        def _filter_genes(self, fcdata):
            filteredfc_dict = {}
            filtered_data = fcdata[fcdata['q_value'] <= 0.05]
            for _, row in filtered_data.iterrows():
                gene_id = row['gene_id']
                log2_fold_change = row['log2(fold_change)']
                if log2_fold_change != float('-inf') and log2_fold_change != float('inf'):
                    filteredfc_dict[gene_id] = log2_fold_change
            print(f'Total filtered mouse genes with log2fc: {len(filteredfc_dict.keys())}')
            return filteredfc_dict
    
    human_idclass = GeneID_human(query_fc)
    final_fc_dict = human_idclass.query_dict

print(final_fc_dict)

# Generate Sample dataframe
gene_list_file = os.path.join(current_dir, 'datasets', 'gene_list.pkl')
with open(gene_list_file, "rb") as f:
    gene_list = pickle.load(f)

sample_df = pd.DataFrame(0.0, index=[0], columns=gene_list)

for gene, fc_value in final_fc_dict.items():
    if gene in sample_df.columns:
        sample_df.at[0, gene] = fc_value

print(sample_df)

sample_df.to_csv(opt.out, sep='\t', index=False)