#!/usr/bin/env python3
# encoding: utf-8
"""
feature_selection.py

Performs MI-POGUE feature selection for E. coli growth rate prediction.
"""
import sys, os, getopt
import numpy as np
import pickle as cp
import pandas as pa
from functools import partial, reduce
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.model_selection import KFold  # Changed from StratifiedKFold
from scoop import futures
from collections import defaultdict

# Load processed data from preparation step
with open('processed_dataF.pkl', 'rb') as f:
    data = cp.load(f)
    sc_gr = data['sc_gr']
    sc_gncorr = data['sc_gncorr']
    sc_gndat = data['sc_gndat']
    evecs = data['evecs']
    NAME = data['NAME']
    GROWTH_ONLY = data['GROWTH_ONLY']

# Helper functions
def SAVE(obj, fn):
    with open(fn, 'wb') as fh: cp.dump(obj, fh, -1)

def group_membership(dat, bd_l):
    gm = dat.astype(object).copy()
    for ii, bd in enumerate(bd_l):
        qc = pa.cut(dat.iloc[:, ii], bd)
        gm.loc[qc.index, qc.name] = qc.astype(str).values
    return gm.groupby(gm.columns.tolist())

def bin_divisions(X, nbins=10):
    if isinstance(nbins, np.ndarray):
        qc, bins = pa.cut(X, nbins, retbins=True)
    else:
        qc, bins = pa.qcut(X, nbins, retbins=True, duplicates='drop')
    dat = pa.DataFrame({X.name: X, 'bin_no': qc}, index=X.index)
    return dat, bins

def initialize_model(feats, noiseFolds=4, gr_err=0.1):
    """Initialize model with proper noise projection"""
    # Validate features
    missing = set(feats) - set(evecs.columns)
    if missing:
        raise ValueError(f"Features missing from evecs: {missing}")

    # Create noise in original gene space
    L = len(sc_gr)
    gene_std = sc_gndat.std().values
    noise_shape = (L * noiseFolds, gene_std.shape[0])
    noise = np.random.randn(*noise_shape) * gene_std

    # Project noise to feature space
    projected_noise = noise @ evecs[feats].values

    # Create activated values
    Act_vals = pa.concat([sc_gncorr[feats]] * noiseFolds)
    new_inds = [f"{gn}--{ii}" for ii in range(noiseFolds) for gn in sc_gncorr.index]
    Act_vals.index = new_inds

    # Create growth rate values
    new_gr = np.concatenate([sc_gr + gr_err * np.random.randn(L) * sc_gr 
                           for _ in range(noiseFolds)])
    
    return pa.Series(new_gr, index=new_inds), Act_vals + pa.DataFrame(projected_noise, index=new_inds, columns=feats)

def create_growth_rate_mapping(train_data, train_target, n_neighbors=7):
    """Create KNN regressor with validation"""
    if train_data.shape[0] != train_target.shape[0]:
        raise ValueError(f"Data mismatch: {train_data.shape} vs {train_target.shape}")
    return KNR(n_neighbors=n_neighbors, weights='distance').fit(train_data, train_target)

def do_strat(n_folds, z_gr, rnd_gr, z_phe, rnd_phe):
    """Modified to use KFold instead of StratifiedKFold"""
    results = []
    
    # Create simple KFold splitter
    kf = KFold(n_splits=n_folds, shuffle=True)
    
    for _ in range(100):  # Monte Carlo repetitions
        for train_idx, test_idx in kf.split(z_phe):
            # Prepare training data with noise
            train_data = pa.concat([
                z_phe.iloc[train_idx],
                rnd_phe
            ])
            train_target = pa.concat([
                z_gr.iloc[train_idx],
                rnd_gr
            ])
            
            # Create and train model
            model = create_growth_rate_mapping(train_data, train_target)
            
            # Test predictions
            test_pred = model.predict(z_phe.iloc[test_idx])
            results.append(pa.DataFrame({
                'Actual': z_gr.iloc[test_idx],
                'Predicted': test_pred
            }))
    
    return pa.concat(results).groupby(level=0).mean()

def cv_mc(features, n_neighbors=7, lambda_reg=0.05, noise_folds=4):
    """Simplified cross-validation without binning"""
    rnd_gr, rnd_phe = initialize_model(features, noise_folds)
    
    return do_strat(
        n_folds=5,
        z_gr=sc_gr,
        rnd_gr=rnd_gr,
        z_phe=sc_gncorr[features],
        rnd_phe=rnd_phe
    )

def process_feature(args):
    """Parallel processing wrapper with explicit state passing"""
    feature, current_selected, n_neighbors, lambda_reg = args
    try:
        test_features = current_selected + [feature]
        result = cv_mc(test_features, n_neighbors, lambda_reg)
        return (feature, (result['Actual'] - result['Predicted']).pow(2).mean())
    except Exception as e:
        print(f"Failed on {feature}: {str(e)}")
        return (feature, float('inf'))

def select_features(max_features=20, n_neighbors=7, lambda_reg=0.05):
    selected_features = []
    performance_log = []
    
    while len(selected_features) < max_features:
        # Get candidate features
        candidate_features = list(evecs.columns.difference(selected_features))
        
        # Prepare arguments for parallel processing
        args_list = [(f, list(selected_features), n_neighbors, lambda_reg) 
                   for f in candidate_features]
        
        # Parallel execution with explicit state
        results = list(futures.map(process_feature, args_list))
        
        # Find best feature from results
        valid_results = [(f, s) for f, s in results if s < float('inf')]
        if not valid_results:
            print("No valid features found, stopping early")
            break
            
        best_feature, best_score = min(valid_results, key=lambda x: x[1])
        selected_features.append(best_feature)
        performance_log.append(best_score)
        print(f"Added {best_feature} - MSE: {best_score:.4f}")

    # Save results
    SAVE({
        'selected_features': selected_features,
        'performance_log': performance_log
    }, f'feature_selection_results_{NAME}.pkl')

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "k:l:m:", 
            ["knn=", "lambda=", "max-features="])
        
        params = {
            'n_neighbors': 7,
            'lambda_reg': 0.05,
            'max_features': 20
        }
        
        for opt, val in opts:
            if opt in ("-k", "--knn"): params['n_neighbors'] = int(val)
            if opt in ("-l", "--lambda"): params['lambda_reg'] = float(val)
            if opt in ("-m", "--max-features"): params['max_features'] = int(val)
        
        select_features(**params)
        
    except Exception as e:
        print(f"Feature selection failed: {str(e)}", file=sys.stderr)
        sys.exit(1)