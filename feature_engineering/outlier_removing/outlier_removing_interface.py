class BaseOutlierRemover:
    def __init__(self):
        self.trimmer = None
        self.threshold = None

    def fit_transform(self, X_train, X_test, y_train, y_test):
        try:
            if not self.trimmer:
                raise NotImplementedError("trimmer is not implemented")

            if not self.threshold:
                raise ValueError("threshold is not defined")

            low_variation_columns = []

            # Loop through columns
            for column in X_train.columns:
                unique_count = X_train[column].nunique()
                if unique_count <= self.threshold:
                    low_variation_columns.append(column)

            outlier_columns = X_train.drop(low_variation_columns, axis=1)
            self.trimmer.fit(outlier_columns)

            X_train = self.trimmer.transform(outlier_columns)
            X_test = self.trimmer.transform(X_test[outlier_columns.columns])

            X_train_enc = X_train[X_train[low_variation_columns].index.isin(X_train.index)]
            X_test_enc = X_test[X_test[low_variation_columns].index.isin(X_test.index)]

            y_train = y_train[y_train.index.isin(X_train_enc.index)]
            y_test = y_test[y_test.index.isin(X_test_enc.index)]

            return (X_train_enc, X_test_enc, y_train, y_test)

        except Exception:
            return (X_train, X_test, y_train, y_test)
