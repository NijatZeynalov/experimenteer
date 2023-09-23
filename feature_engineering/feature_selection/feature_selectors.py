from feature_engineering.feature_selection.feature_selection_interface import (
    BaseFeatureSelector,
)
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class KBestSelector(BaseFeatureSelector):
    def __init__(self):
        super().__init__()
        self.selector = SelectKBest(k=self.N_MIN_FEATURES)


class RecursiveFeatureSelector(BaseFeatureSelector):
    def __init__(self):
        super().__init__()
        estimator = RandomForestClassifier()  # You can use a different estimator
        self.selector = RFE(estimator=estimator)

    def fit_transform(self, X_train, X_test, y_train, y_test):
        try:
            return super().fit_transform(X_train, X_test, y_train, y_test)

        except ValueError:
            y_classes = np.where(y_train > np.median(y_train), 1, 0)
            self.selector.fit(X_train, y_classes)

            return self._pandas_transformation(X_train, X_test, y_train, y_test)


class DummySelector(BaseFeatureSelector):
    def __init__(self):
        super().__init__()

    def fit_transform(self, X_train, X_test, y_train, y_test):
        return (X_train, X_test, y_train, y_test)
