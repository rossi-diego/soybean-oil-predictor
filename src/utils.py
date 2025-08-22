import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

def coefficients_dataframe(coefs, columns):
    return pd.DataFrame(data=coefs, index=columns, columns=["coefficient"]).sort_values(
        by="coefficient"
    )