from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from feature_engineering.scaling.scaling_interface import BaseScaler


class Standard(BaseScaler):
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler().set_output(transform="pandas")


class MinMax(BaseScaler):
    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler().set_output(transform="pandas")


class Robust(BaseScaler):
    def __init__(self):
        super().__init__()
        self.scaler = RobustScaler().set_output(transform="pandas")


class DummyScaler(BaseScaler):
    def __init__(self):
        super().__init__()

    def fit_transform(self, X_train, X_test, y_train, y_test):
        return (X_train, X_test, y_train, y_test)
