from sklearn.base import BaseEstimator, RegressorMixin
import sys
sys.path.insert(1, '/gpfs/home/agillig/Projects/DynaPred/code/l0learn')
import l0learn

class L0Regression(BaseEstimator, RegressorMixin):

    def __init__(self):  
        self.lib_loc = '/gpfs/home/agillig/R/x86_64-pc-linux-gnu-library/4.3'
        self.n_folds = 10

    def fit(self, X, y = None):
        self.coef, self.support_ = l0learn.cvfit(X, y, n_cv=self.n_folds, lib_loc=self.lib_loc)
        return self

    def predict(self, X, y = None):
        return l0learn.predict(X, self.coef)


# class AbessRegression(BaseEstimator, RegressorMixin):

#     def __init__(self):
#         self.n_folds = 10

