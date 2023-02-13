import copy
import numpy as np

def train(est, X_train, X_prime_train, y_train, X_test, sens_cols=None, **kwargs):

    protected_groups = [int(c in sens_cols) for c in X_train.columns]
    est.protected_groups = ','.join(                                       
            [str(int(pg)) for pg in protected_groups]).encode() 

    est.feature_names = ','.join(list(X_train.columns)).encode()
    est.fit(X_train, y_train)
    print('archive size:',est.get_archive_size())
    train_predictions = est.predict_archive(X_train.values)
    test_predictions = est.predict_archive(X_test.values)

    print('getting probabilities')
    train_probabilities, test_probabilities = [],[]
    for i in np.arange(est.get_archive_size()):
        train_probabilities.append(
                np.nan_to_num(
                    est.predict_proba_archive(i,X_train.values).flatten()
                    )
                )
        test_probabilities.append(
                np.nan_to_num(
                    est.predict_proba_archive(i,X_test.values).flatten()
                    )
                )

    return (
        train_predictions,
        test_predictions,
        train_probabilities,
        test_probabilities
    )
