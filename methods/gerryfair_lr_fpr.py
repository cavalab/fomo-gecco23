import gerryfair
from .train_gerry_fair import train

# set up Gerry Fair model
est = gerryfair.model.Model(
    C=15, 
    printflag=True, 
    fairness_def='FP', 
    max_iters=100
)