#!/usr/bin/env python3
# encoding: utf-8
"""
biomass_precursors.py

Analyzes biomass precursors using MI-POGUE framework.
"""
import sys
import os.path as osp
import getopt
import pickle as cp
import numpy as np
import pandas as pa
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.model_selection import KFold

# Load ALL processed data components
with open('processed_data.pkl', 'rb') as f:
    data = cp.load(f)
    sc_gr = data['sc_gr']
    sc_gncorr = data['sc_gncorr']
    sc_gndat = data['sc_gndat'] 
    evecs = data['evecs']
    NAME = data.get('NAME', 'default_name')

def load_precursors(model='iJO1366'):
    """Validate precursors against raw gene data"""
    precursors = pa.read_pickle(f'data/{model}_precursors_l.pkl')
    
    # Validate against raw gene expression columns
    valid = [gene for gene in precursors if gene in sc_gndat.columns]
    missing = set(precursors) - set(valid)
    
    if missing:
        print(f"Warning: {len(missing)} precursors missing from dataset")
        print("First 10 missing:", list(missing)[:10])
    
    if not valid:
        sample_columns = "\n".join(sc_gndat.columns.tolist()[:20])
        raise ValueError(
            "No valid precursors found.\n"
            f"Dataset contains columns like:\n{sample_columns}"
        )
    
    return valid

def biomass_cv(precursors, n_neighbors=7, n_folds=10):
    """Use raw gene expression data (sc_gndat)"""
    model = KNR(n_neighbors=n_neighbors, weights='distance')
    predictions = []
    
    kf = KFold(n_splits=n_folds, shuffle=True)
    
    # Add enumerate to get fold numbers
    for fold, (train_idx, test_idx) in enumerate(kf.split(sc_gndat)):
        train_data = sc_gndat.iloc[train_idx][precursors]
        test_data = sc_gndat.iloc[test_idx][precursors]
        
        model.fit(train_data, sc_gr.iloc[train_idx])
        pred = model.predict(test_data)
        
        predictions.append(pa.DataFrame({
            'Fold': fold,  # Now correctly numbered
            'Actual': sc_gr.iloc[test_idx],
            'Predicted': pred
        }))
    
    return pa.concat(predictions)

def biomass_precursors_analysis(model='iJO1366'):
    """Main analysis workflow"""
    precursors = load_precursors(model)
    print(f"Analyzing {len(precursors)} valid biomass precursors for {model}")
    
    results = biomass_cv(precursors)
    
    # Calculate statistics
    mse = (results['Actual'] - results['Predicted']).pow(2).mean()
    r2 = results[['Actual', 'Predicted']].corr().iloc[0,1]**2
    
    print(f"\nResults for {model} biomass precursors:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Save results
    results.to_csv(f'biomass_{model}_results.csv', index=False)

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "m:", ["model="])
        model = 'iJO1366'
        
        for opt, val in opts:
            if opt in ("-m", "--model"):
                model = val
        
        biomass_precursors_analysis(model)
        
    except Exception as e:
        print(f"Biomass analysis failed: {str(e)}", file=sys.stderr)
        sys.exit(1)