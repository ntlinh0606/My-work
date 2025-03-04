#!/usr/bin/env python3
# encoding: utf-8
"""
data_preparation.py

Processes and saves E. coli data for MI-POGUE analysis.
"""
import sys, os, os.path as osp, getopt
import numpy as np
import pickle as cp
import pandas as pa

help_message = '''
Processes gene expression and growth rate data.
Options:
-h, --help : Show help
-e, --expression-file= : Gene expression file
-r, --growth-rate= : Growth rate file
-g, --growth-only : Use only paired growth data
-N, --name= : Output name prefix
'''

class Usage(Exception):
    def __init__(self, msg): self.msg = msg

def SAVE(obj, fn):
    with open(fn, 'wb') as fh: cp.dump(obj, fh, -1)

try:
    FN = 'SF1-EcoMAC/ecoli_compendium_df.pkl'
    GFN = 'data/DatasetS1_CarreraAnnotations.xlsx'
    NAME = 'carrera-corr'
    GROWTH_ONLY = False

    opts, args = getopt.getopt(sys.argv[1:], "he:r:gN:", 
                       ["help", "expression-file=", "growth-rate=", 
                        "growth-only", "name="])
    for opt, val in opts:
        if opt in ("-h", "--help"): raise Usage(help_message)
        if opt in ("-e", "--expression-file"): FN = val
        if opt in ("-r", "--growth-rate"): GFN = val
        if opt in ("-g", "--growth-only"): GROWTH_ONLY = True
        if opt in ("-N", "--name"): NAME = val

    # Load data
    if FN.endswith('.pkl'):
        sc_gndat = pa.read_pickle(FN).T
    elif FN.endswith('.csv'):
        sc_gndat = pa.read_csv(FN, sep='\t').T
    
    sc_gr = pa.read_excel(GFN, sheet_name='EcoMAC and EcoPhe', 
                        index_col='CEL file name')
    
    if NAME.startswith('carrera-corr'):
        sc_gr = sc_gr[sc_gr['Flag growth']==1]
        sc_gr = sc_gr['Growth rate (1/h)'].astype(float)
        sc_gncorr = pa.read_pickle(f'ecoli_compendium_gncorr_{NAME}_df.pkl')
        evecs = pa.read_pickle(f'ecoli_evecs_{NAME}_df.pkl')

    # Align data
    common_idx = sc_gr.index.intersection(sc_gncorr.index)
    sc_gr = sc_gr.loc[common_idx]
    sc_gncorr = sc_gncorr.loc[common_idx]
    sc_gndat = sc_gndat.loc[common_idx]
    
    valid_idx = sc_gr.index[~np.isnan(sc_gr)]
    sc_gr = sc_gr.loc[valid_idx]
    sc_gndat = sc_gndat.loc[valid_idx]
    sc_gncorr = sc_gncorr.loc[valid_idx]
   
    # Save processed data
    processed_data = {
        'sc_gr': sc_gr,
        'sc_gncorr': sc_gncorr,
        'sc_gndat': sc_gndat,
        'evecs': evecs,
        'NAME': NAME,
        'GROWTH_ONLY': GROWTH_ONLY
    }
    SAVE(processed_data, 'processed_data.pkl')

except Usage as err:
    print(f"ERROR: {err.msg}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Processing failed: {str(e)}", file=sys.stderr)
    sys.exit(1)