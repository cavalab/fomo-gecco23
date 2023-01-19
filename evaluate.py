"""Evaluate a method on a dataset"""
import copy
from utils import setup_data,evaluate_model,get_hypervolumes
import numpy as np
import pandas as pd
import time
import json
import importlib

def evaluate(model_name, dataset, attributes, seed, rdir):
    """Evaluates the estimator in methods/{model_name}.py on dataset and stores
    results.
    """
    print(f'training {model_name} on {dataset}, seed={seed}, rdir={rdir}')
    # data setup
    X_train, X_test, X_prime_train, X_prime_test, y_train, y_test, sens_cols = \
    setup_data(dataset, attributes, seed)
    dataset_name = dataset.split('/')[-1].split('.')[0]

    # train algorithm
    alg = importlib.import_module(f"methods.{model_name}")

    t0 = time.process_time()
    res = alg.train(alg.est, X_train, X_prime_train, y_train, X_test, sens_cols)

    train_predictions=res[0]
    test_predictions=res[1]
    train_probabilities=res[2]
    test_probabilities=res[3]

    performance = []
    for i, (train_pred, test_pred, train_prob, test_prob) in enumerate(zip(
        train_predictions,
        test_predictions,
        train_probabilities, 
        test_probabilities
        )):
        performance.append({
            'method':model_name,
            'model':model_name+':archive('+str(i)+')',
            'dataset':dataset_name,
            'seed':seed,
            'train':evaluate_model(X_train, X_prime_train, y_train, train_pred, 
                train_prob),
            'test':evaluate_model(X_test, X_prime_test, y_test, test_pred, 
                test_prob)
        })
        
            
    runtime = time.process_time() - t0
    header = {
            'method':model_name,
            'dataset':dataset_name,
            'seed':seed,
            'time':runtime
    }
    # get hypervolume of pareto front
    hv = get_hypervolumes(performance)
    hv = [{**header, **i} for i in hv]
    df_hv = pd.DataFrame.from_records(hv)
    df_hv.to_csv(
            f'{rdir}/hv_{model_name}_{seed}_{dataset_name}.csv',
            index=False
        )
    
    with open(f'{rdir}/perf_{model_name}_{dataset_name}_{seed}.json', 'w') as fp:
        json.dump(performance, fp, sort_keys=True, indent=2)
    return performance, df_hv

import argparse
if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a method on a dataset.", add_help=False)
    parser.add_argument('-data', action='store', type=str,
                        help='Data file to analyze')
    parser.add_argument('-atts', action='store', type=str,
                        help='File specifying protected attributes')
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', default='xgb',type=str,
            help='Name of estimator (with matching file in ml/)')
    parser.add_argument('-rdir', action='store', default='../results', type=str,
                        help='Name of save file')
    parser.add_argument('-seed', action='store', default=42, type=int, help='Seed / trial')
    args = parser.parse_args()

    evaluate( args.ml, args.data, args.atts, args.seed, args.rdir)