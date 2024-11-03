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
parser.add_argument('--org', default = 'human', choices = ['mouse','human'])
parser.add_argument('--out', default='Preprocessed_fc.tsv')

opt = parser.parse_args()

query_fc = pd.read_csv(opt.query_fc, sep='\t', index_col=0)

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
            fcdata = fcdata.rename(columns=self.gene_dict)
            result_dict = {
                row_name: {col: fcdata.at[row_name, col] for col in fcdata.columns}
                for row_name in fcdata.index
            }
            return result_dict

    MGI_idclass = GeneID_MGI(MGI_db, query_fc)
    result_dict = MGI_idclass.query_dict

    final_fc_dict = dict()
    for sample, fc_dict in result_dict.items():
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
        matched_genes = filtered_df[filtered_df['ensembl_gene_id'].isin(fc_dict.keys())]

        ortholog_fc = defaultdict(list)
        for index, row in matched_genes.iterrows():
            gene_name = row['hsapiens_homolog_associated_gene_name']
            fc_value = fc_dict[row['ensembl_gene_id']]
            ortholog_fc[gene_name].append(fc_value)

        # Process duplicated human genes
        final_fc_dict[sample] = {gene: sum(values) / len(values) for gene, values in ortholog_fc.items()}

else:
    class GeneID_human:
        def __init__(self, query_fc):
            super().__init__()
            self.query_dict = self._filter_genes(query_fc)

        def _filter_genes(self, fcdata):
            result_dict = {
                row_name: {col: fcdata.at[row_name, col] for col in fcdata.columns}
                for row_name in fcdata.index
            }
            return result_dict
    
    human_idclass = GeneID_human(query_fc)
    final_fc_dict = human_idclass.query_dict

# Generate Sample dataframe
gene_list_file = os.path.join(current_dir, 'datasets', 'gene_list.pkl')
with open(gene_list_file, "rb") as f:
    gene_list = pickle.load(f)

filtered_df = pd.DataFrame(index=final_fc_dict.keys(), columns=gene_list)

for row_name, inner_dict in final_fc_dict.items():
    for col, value in inner_dict.items():
        if col in gene_list:
            if value != 0.0:  # 0.0이 아닌 값만 조건에 따라 대입
                filtered_df.at[row_name, col] = value

filtered_df = filtered_df.fillna(0.0)

filtered_df.to_csv(opt.out, sep='\t')