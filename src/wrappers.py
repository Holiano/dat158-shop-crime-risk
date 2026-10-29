"""Små modell-wrappare. Ligg i eigen modul slik at joblib kan laste dei i appen."""
import numpy as np


class Log1pRegressor:
    """Pakkar ein modell slik at det lærer log(1+y) og gir tilbake teljingar."""

    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.est.fit(X, np.log1p(y))
        return self

    def predict(self, X):
        return np.expm1(self.est.predict(X)).clip(min=0)


class DeltaRegressor:
    """Lærer endringa frå fjorårstalet i log-rom: log(1+y) - log(1+lag1)."""

    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        base = np.log1p(X["business_crime_lag1"].to_numpy())
        self.est.fit(X, np.log1p(y) - base)
        return self

    def predict(self, X):
        base = np.log1p(X["business_crime_lag1"].to_numpy())
        return np.expm1(self.est.predict(X) + base).clip(min=0)
