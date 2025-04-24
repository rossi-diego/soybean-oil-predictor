import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, period):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = np.asarray(X).astype(int).reshape(-1)
        sin = np.sin(2 * np.pi * x / self.period)
        cos = np.cos(2 * np.pi * x / self.period)
        return np.vstack([sin, cos]).T

    def get_feature_names_out(self, input_features=None):
        name = input_features[0]
        return np.array([f"{name}_sin", f"{name}_cos"])

def coefficients_dataframe(coefs, columns):
    return pd.DataFrame(data=coefs, index=columns, columns=["coefficient"]).sort_values(
        by="coefficient"
    )