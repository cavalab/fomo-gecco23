import copy
import numpy as np
import warnings
import time

def train(est, X_train, X_prime_train, y_train, X_test, sens_cols, **kwargs):
    gamma_list = np.linspace(0.001,0.999,100)
    # models = []
    train_predictions = []
    test_predictions = []
    train_probabilities = []
    test_probabilities = []
    t0 = time.time()
    maxtime = 3600
    iter_time = 0
    # est.max_iters = 10
    # est.set_options(max_iters=10)
    with warnings.catch_warnings(): 
        warnings.simplefilter('ignore')
        for g in gamma_list:
            ti0 = time.time()
            if (ti0 - t0) + iter_time > maxtime:
                print('max time reached')
                break
    #         est.gamma = g
            model = copy.deepcopy(est)
            model.set_options(gamma=g)
            print('gamma:',model.gamma)
            #train
            error, fairness_violation = model.train(
                X_train, 
                X_prime_train,
                y_train.values
            )
            train_pred = model.predict(X_train, sample=True)
            train_prob = model.predict(X_train, sample=False)

            train_predictions.append(np.array(train_pred))
            train_probabilities.append(np.array(train_prob))

            test_prob = model.predict(X_test, sample=False)
            test_pred = model.predict(X_test, sample=True)
            test_predictions.append(np.array(test_pred))
            test_probabilities.append(np.array(test_prob))

            iter_delta =time.time() - ti0

            if iter_time == 0:
                iter_time = iter_delta
            else:
                iter_time = (iter_time+iter_delta)/2

            print('gamma iteration time: ',iter_time)

    return (
        train_predictions,
        test_predictions,
        train_probabilities,
        test_probabilities
    )
