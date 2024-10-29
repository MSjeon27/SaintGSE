#! /usr/bin/env python
# Integrate foldchange output for SaintGSE prediction

import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--fclist', required=True, nargs='+')
parser.add_argument('--out', default='FC_predquery.tsv')

opt = parser.parse_args()

fc_list = []

# Set index of each experiment
for file in opt.fclist:
    df = pd.read_csv(file, sep='\t')
    df['filename'] = '.'.join(file.split('.')[:-1])
    df.set_index('filename', inplace=True)
    fc_list.append(df)
    
fc_df = pd.concat(fc_list)

print(fc_df)

fc_df.to_csv(opt.out, sep='\t')