from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols  # List of columns to apply cyclical encoding to
        self.max_values = {}  # Store the max values for each column

    def fit(self, X, y=None):
        """
        Learn the max value for each column during training.
        """
        for col in self.cols:
            max_val = X[col].max()
            self.max_values[col] = max_val
        return self

    def transform(self, X):
        """
        Apply cyclical encoding to the columns using learned max values.
        """
        X_copy = X.copy()
        for col in self.cols:
            # Default to 1 if max_val not found
            max_val = self.max_values.get(col, 1)
            X_copy[col + '_sin'] = np.sin(2 *
                                          np.pi * X_copy[col] / max_val).round(8)
            X_copy[col + '_cos'] = np.cos(2 *
                                          np.pi * X_copy[col] / max_val).round(8)
        return X_copy
