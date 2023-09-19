from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

class Scaler:

    def __init__(self, scaler_type):
        self.scaler_type = scaler_type

    def standard(self):
        scaler = StandardScaler().set_output(transform="pandas")
        scaler.fit(self.X_train)

        # transform train and test sets
        X_train_scaled = scaler.transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        return X_train_scaled, X_test_scaled,self.y_train, self.y_test

    def min_max(self):
        scaler = MinMaxScaler().set_output(transform="pandas")

        # fit the scaler to the train set, it will learn the parameters
        scaler.fit(self.X_train)

        # transform train and test sets
        X_train_scaled = scaler.transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        return X_train_scaled, X_test_scaled,self.y_train, self.y_test

    def robust(self):
        scaler = RobustScaler().set_output(transform="pandas")

        # fit the scaler to the train set, it will learn the parameters
        scaler.fit(self.X_train)

        # transform train and test sets
        X_train_scaled = scaler.transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        return X_train_scaled, X_test_scaled,self.y_train, self.y_test

    def method_transform(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        if self.scaler_type == 'standard':
            return self.standard()
        elif self.scaler_type == 'robust':
            return self.robust()
        elif self.scaler_type == 'min_max':
            return self.min_max()
        else:
            return self.X_train, self.X_test, self.y_train, self.y_test