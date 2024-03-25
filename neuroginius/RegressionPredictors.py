#predictors for continuous behavioral data
from sklearn.kernel_ridge import KernelRidge
from abc import ABC


class RegressionPredictor(ABC):
    # abstract methods to be implemented by subclasses
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class KernelRidgePredictor(RegressionPredictor):
    def __init__(self, alpha=1.0, kernel='linear'):
        self.alpha = alpha
        self.kernel = kernel
        self.model = KernelRidge(alpha=self.alpha, kernel=self.kernel)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)