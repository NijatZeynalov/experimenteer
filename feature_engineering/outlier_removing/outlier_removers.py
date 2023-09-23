from feature_engine.outliers import OutlierTrimmer
from feature_engineering.outlier_removing.outlier_removing_interface import (
    BaseOutlierRemover,
)


class IQR(BaseOutlierRemover):
    def __init__(self):
        super().__init__()
        self.trimmer = OutlierTrimmer(
            capping_method="iqr",
            tail="both",
            fold=1.5,
        )

        self.threshold = 3


class Quantile(BaseOutlierRemover):
    def __init__(self):
        super().__init__()
        self.trimmer = OutlierTrimmer(
            capping_method="quantiles",
            tail="both",
            fold=0.05,
        )

        self.threshold = 2


class DummyOutlierRemover(BaseOutlierRemover):
    def __init__(self):
        super().__init__()

    def fit_transform(self, X_train, X_test, y_train, y_test):
        return (X_train, X_test, y_train, y_test)
