from feature_engineering.missing_imputation.imputers import (
    MeanImputer,
    ArbitraryValueImputer,
    MissingIndicatorImputer,
    KnnImputer,
    IterImputer,
)


IMPUTERS = {
    "mean_imputer": MeanImputer,
    "arbitrary_value_imputer": ArbitraryValueImputer,
    "missing_indicator_imputer": MissingIndicatorImputer,
    "knn_imputer": KnnImputer,
    "iterative_imputer": IterImputer,
}


class ImputerFactory:
    @staticmethod
    def get_imputer(imputer_type):
        if imputer_type not in IMPUTERS:
            raise ValueError(
                f"Invalid Imputer type: {imputer_type}. Supported Imputers: {IMPUTERS.keys()}"
            )

        return IMPUTERS[imputer_type]()
