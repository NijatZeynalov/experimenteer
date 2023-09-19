from feature_engine.datetime import DatetimeFeatures



class DateTimeHandler:

    def __init__(self, variables = None):
        self.variables = variables


    def method_transform(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        try:
            dtfs = DatetimeFeatures(
                variables=None,  # it identifies the datetime variable automatically.
                features_to_extract="all",
            )
            X_train_d = dtfs.fit_transform(self.X_train)
            X_test_d = dtfs.transform(self.X_test)
            return X_train_d, X_test_d,self.y_train, self.y_test

        except:
            return self.X_train, self.X_test,self.y_train, self.y_test
