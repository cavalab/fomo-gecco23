from fomo import FomoClassifier, metrics
from pymoo.algorithms.moo.nsga2 import NSGA2

from .train_fomo import train
from sklearn.linear_model import LogisticRegression

est = FomoClassifier(
    estimator = LogisticRegression(n_jobs=1),
    algorithm = NSGA2(),
    fairness_metrics=[metrics.subgroup_FNR],
    verbose=True,
    batch_size=0,
    n_jobs=10
)
