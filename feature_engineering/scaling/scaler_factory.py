from feature_engineering.scaling.scalers import (
    Standard,
    MinMax,
    Robust,
    DummyScaler,
)


SCALERS = {
    "standard": Standard,
    "min_max": MinMax,
    "robust": Robust,
    "none": DummyScaler,
}


class ScalerFactory:
    @staticmethod
    def get_scaler(scaler_type):
        if scaler_type not in SCALERS:
            raise ValueError(
                f"Invalid scaler type: {scaler_type}. " f"Supported scalers: {list(SCALERS.keys())}"
            )

        return SCALERS[scaler_type]()
