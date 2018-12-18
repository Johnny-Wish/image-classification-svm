from sklearn.svm import SVC
from scipy.stats.distributions import uniform

svc = SVC()
param_dist = {
    "C": uniform(0.5, 2),
    "kernel": ["rbf"],
    "shrinking": [True, False],
    "class_weight": [None, "balanced"],
    "random_state": [0],
}