from feature_engineering.encoder.encoder_interface import BaseEncoder
from feature_engine.encoding import (
    OneHotEncoder,
    OrdinalEncoder,
    CountFrequencyEncoder,
    MeanEncoder,
    WoEEncoder,
)


class OneHot(BaseEncoder):
    def __init__(self):
        self.encoder = OneHotEncoder(
            variables=None,  # alternatively pass a list of variables
            drop_last=True,  # to return k-1, use drop=false to return k dummies
        )


class Ordinal(BaseEncoder):
    def __init__(self):
        self.encoder = OrdinalEncoder(encoding_method="arbitrary", unseen="encode")


class Frequency(BaseEncoder):
    def __init__(self):
        self.encoder = CountFrequencyEncoder(encoding_method="frequency", unseen="encode")


class Mean(BaseEncoder):
    def __init__(self):
        self.encoder = MeanEncoder(unseen="encode")

    def fit_transform(self, X_train, X_test, y_train, y_test):
        self.encoder.fit(X_train, y_train)
        X_train = self.encoder.transform(X_train)
        X_test = self.encoder.transform(X_test)

        return (X_train, X_test, y_train, y_test)


class OneHotFrequent(BaseEncoder):
    def __init__(self):
        self.encoder = OneHotEncoder(top_categories=10)


class WoE(BaseEncoder):
    def __init__(self):
        self.encoder = WoEEncoder(unseen="raise")
