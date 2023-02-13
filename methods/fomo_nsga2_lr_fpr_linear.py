from fomo import FomoClassifier
from pymoo.algorithms.moo.nsga2 import NSGA2 as algorithm
from fomo.metrics import subgroup_FPR as metric
from fomo.problem import LinearProblem

from .train_fomo import train
from sklearn.linear_model import LogisticRegression as ml

est = FomoClassifier(
    estimator = ml(),
    algorithm = algorithm(),
    problem_type = LinearProblem, 
    fairness_metrics=[metric],
    store_final_models=True,
    verbose=True,
    n_jobs=1,
)
