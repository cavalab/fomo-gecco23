from fomo import FomoClassifier
from pymoo.algorithms.moo.nsga2 import NSGA2 as algorithm
from fomo.metrics import subgroup_FPR as metric
from fomo.problem import MLPProblem

from .train_fomo import train
from xgboost.sklearn import XGBClassifier as ml

est = FomoClassifier(
    estimator = ml(),
    algorithm = algorithm(),
    problem_type = MLPProblem, 
    fairness_metrics=[metric],
    store_final_models=True,
    verbose=True,
    n_jobs=1,
)
