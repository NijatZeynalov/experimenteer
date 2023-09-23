from pandas import DataFrame


class BaseFeatureSelector:
    N_MIN_FEATURES = 5

    def __init__(self):
        self.selector = None

    def fit_transform(self, X_train, X_test, y_train, y_test):
        if not self.selector:
            raise NotImplementedError(
                "Selector not implemented. Please implement the selector in the child class."
            )

        if X_train.shape[1] <= self.N_MIN_FEATURES:
            return (X_train, X_test, y_train, y_test)

        self.selector.fit(X_train, y_train)

        return self._pandas_transformation(X_train, X_test, y_train, y_test)

    def _pandas_transformation(self, X_train, X_test, y_train, y_test):
        X_train_selected = self.selector.transform(X_train)
        X_test_selected = self.selector.transform(X_test)

        X_train_selected = DataFrame(X_train_selected, index=X_train.index)
        X_test_selected = DataFrame(X_test_selected, index=X_test.index)

        return (
            X_train_selected,
            X_test_selected,
            y_train,
            y_test,
        )
