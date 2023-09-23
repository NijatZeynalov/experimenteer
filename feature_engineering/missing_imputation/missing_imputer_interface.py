from pandas import DataFrame, concat
from sklearn.experimental import enable_iterative_imputer  # noqa


class BaseImputer:
    def __init__(self):
        self.imputer = None
        self.frequent_category_imputer = None

    def fit_transform(self, X_train, X_test, y_train, y_test):
        self._check_imputers()

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self.frequent_category_imputer.fit_transform(X_train, X_test, y_train, y_test)

        self.imputer.fit(X_train)
        X_train = self.imputer.transform(X_train)
        X_test = self.imputer.transform(X_test)

        return (X_train, X_test, y_train, y_test)

    def complex_fit_transform(self, X_train, X_test, y_train, y_test):
        self._check_imputers()

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self.frequent_category_imputer.fit_transform(X_train, X_test, y_train, y_test)

        X_train_numeric = X_train.select_dtypes(include=["number"])
        X_test_numeric = X_test[X_train_numeric.columns]
        X_train_categoric = X_train.select_dtypes(exclude=["number"])
        X_test_categoric = X_test[X_train_categoric.columns]

        numeric_columns = X_train_numeric.columns

        self.imputer.fit(X_train_numeric)
        X_train_numeric = self.imputer.transform(X_train_numeric)
        X_test_numeric = self.imputer.transform(X_test_numeric)

        # Convert the imputed results back to DataFrames
        X_train_numeric = DataFrame(X_train_numeric, columns=numeric_columns)
        X_test_numeric = DataFrame(X_test_numeric, columns=numeric_columns)
        X_train_numeric.index = X_train.index
        X_test_numeric.index = X_test.index

        # Combine imputed numeric columns with the original categorical columns
        train_combined = concat([X_train_numeric, X_train_categoric], axis=1)
        test_combined = concat([X_test_numeric, X_test_categoric], axis=1)

        return (train_combined, test_combined, y_train, y_test)

    def _check_imputers(self):
        if not self.frequent_category_imputer:
            raise NotImplementedError("Frequent category imputer not implemented")

        if not self.imputer:
            raise NotImplementedError("Imputer not implemented")
