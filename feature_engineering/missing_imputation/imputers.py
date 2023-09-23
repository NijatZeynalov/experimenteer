from feature_engineering.missing_imputation.missing_imputer_interface import (
    BaseImputer,
)
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer

from feature_engine.imputation import (
    MeanMedianImputer,
    ArbitraryNumberImputer,
    AddMissingIndicator,
)
from sklearn.linear_model import BayesianRidge


class FrequentCategoryImputer(BaseImputer):
    def __init__(self):
        super().__init__()

    def fit_transform(self, X_train, X_test, y_train, y_test):
        categorical_columns = X_train.select_dtypes(include=["object"]).columns
        for column in categorical_columns:
            # Calculate value counts
            value_counts = X_train[column].value_counts()

            # Check if there are missing values in the column
            if X_train[column].isnull().any():
                # Replace missing values with the first index value (most frequent category)
                most_frequent_category = value_counts.index[0]
                X_train[column].fillna(most_frequent_category, inplace=True)
                X_test[column].fillna(most_frequent_category, inplace=True)

        return X_train, X_test, y_train, y_test


class MeanImputer(BaseImputer):
    def __init__(self):
        super().__init__()
        self.imputer = MeanMedianImputer(imputation_method="mean")
        self.frequent_category_imputer = FrequentCategoryImputer()

    def fit_transform(self, X_train, X_test, y_train, y_test):
        X_train, X_test, y_train, y_test = super().fit_transform(X_train, X_test, y_train, y_test)
        X_test = X_test.dropna()
        y_test = y_test[y_test.index.isin(X_test.index)]

        assert not y_test.isnull().any().any(), "There are missing values in the test set"

        return (X_train, X_test, y_train, y_test)


class ArbitraryValueImputer(BaseImputer):
    def __init__(self):
        super().__init__()
        self.imputer = ArbitraryNumberImputer(arbitrary_number=-999)
        self.frequent_category_imputer = FrequentCategoryImputer()


class MissingIndicatorImputer(BaseImputer):
    def __init__(self):
        super().__init__()
        self.imputer = AddMissingIndicator()
        self.frequent_category_imputer = FrequentCategoryImputer()


class KnnImputer(BaseImputer):
    def __init__(self):
        super().__init__()
        self.imputer = KNNImputer(
            n_neighbors=5,  # the number of neighbors K
            weights="distance",  # the weighting factor
            metric="nan_euclidean",  # the metric to find the neighbors
            add_indicator=False,  # whether to add a missing indicator
        )

        self.frequent_category_imputer = FrequentCategoryImputer()

    def fit_transform(self, X_train, X_test, y_train, y_test):
        return super().complex_fit_transform(X_train, X_test, y_train, y_test)


class IterImputer(BaseImputer):
    def __init__(self):
        self.imputer = IterativeImputer(
            estimator=BayesianRidge(),  # the estimator to predict the NA
            initial_strategy="mean",  # how will NA be imputed in step 1
            max_iter=10,  # number of cycles
            imputation_order="ascending",  # the order in which to impute the variables
            n_nearest_features=None,  # whether to limit the number of predictors
            skip_complete=True,  # whether to ignore variables without NA
            random_state=0,
        )

        self.frequent_category_imputer = FrequentCategoryImputer()

    def fit_transform(self, X_train, X_test, y_train, y_test):
        return super().complex_fit_transform(X_train, X_test, y_train, y_test)
