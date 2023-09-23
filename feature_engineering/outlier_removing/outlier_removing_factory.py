from feature_engineering.outlier_removing.outlier_removers import (
    IQR,
    Quantile,
    DummyOutlierRemover,
)

OUTLIER_REMOVERS = {
    "removing_outliers_iqr": IQR,
    "removing_outliers_quantiles": Quantile,
    "none": DummyOutlierRemover,
}


class OutlierRemoverFactory:
    @staticmethod
    def get_outlier_remover(remover_type):
        if remover_type not in OUTLIER_REMOVERS:
            raise ValueError(
                f"Invalid Remover type: {remover_type}. "
                f"Supported removers: {list(OUTLIER_REMOVERS.keys())}"
            )

        return OUTLIER_REMOVERS[remover_type]()
