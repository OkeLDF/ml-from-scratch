from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted, validate_data
import numpy as np

class GDLinearRegressor(RegressorMixin, BaseEstimator):

    def __init__(
            self,
            n_iterations=1000,
            learning_rate = 0.1,
        ) -> None:
        super().__init__()
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._is_fitted = False

    def fit(self, X, y):
        X, y = validate_data(self, X, y, reset=False)
        self.n_features_in_ = X.shape[1]
        
        self.scaler_ = MinMaxScaler()
        self.X_ = self.scaler_.fit_transform(X)
        self.X_b_ = np.c_[np.ones((self.X_.shape[0], 1)), self.X_]
        self.y_ = y

        theta = np.random.rand(self.X_b_.shape[1])
        m = X.shape[0]
        last_gradients = np.zeros(theta.shape[0])

        for i in range(self.n_iterations):
            gradients = (2/m) * self.X_b_.T.dot(self.X_b_.dot(theta) - y)

            if np.array_equiv(gradients, last_gradients):
                break

            theta = theta - self.learning_rate * gradients
            last_gradients = gradients

        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = self.scaler_.transform(X)
        
        return X.dot(self.coef_) + self.intercept_
