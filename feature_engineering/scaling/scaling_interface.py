class BaseScaler:
    def __init__(self):
        self.scaler = None

    def fit_transform(self, X_train, X_test, y_train, y_test):
        if not self.scaler:
            raise NotImplementedError("scaler is not implemented")

        self.scaler.fit(X_train)

        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return (X_train_scaled, X_test_scaled, y_train, y_test)
