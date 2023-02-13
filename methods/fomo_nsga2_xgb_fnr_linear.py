from fomo import FomoClassifier
from pymoo.algorithms.moo.nsga2 import NSGA2 as algorithm
from fomo.metrics import subgroup_FNR as metric
from fomo.problem import LinearProblem

from .train_fomo import train
from xgboost.sklearn import XGBClassifier as ml

est = FomoClassifier(
    estimator = ml(),
    algorithm = algorithm(),
    problem_type = LinearProblem, 
    fairness_metrics=[metric],
    store_final_models=True,
    verbose=False,
    n_jobs=1,
)
