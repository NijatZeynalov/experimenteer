class BaseEncoder:
    def __init__(self):
        self.encoder = None

    def fit_transform(self, X_train, X_test, y_train, y_test):
        if not self.encoder:
            raise NotImplementedError(
                "Encoder not implemented. Please implement the encoder in the child class."
            )
        self.encoder.fit(X_train)
        X_train_enc = self.encoder.transform(X_train)
        X_test_enc = self.encoder.transform(X_test)

        return (X_train_enc, X_test_enc, y_train, y_test)
