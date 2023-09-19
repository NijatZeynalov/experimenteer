from sklearn.feature_selection import SelectKBest, RFE
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
class FeatureSelector:

    def __init__(self, selector_type):
        self.selector_type = selector_type


    def select_k_best(self):
        k = 5
        if self.X_train.shape[1]>k:
            selector = SelectKBest(k=k)
            selector.fit(self.X_train, self.y_train)
            X_train_selected = selector.transform(self.X_train)
            X_test_selected = selector.transform(self.X_test)

            X_train_selected = pd.DataFrame(X_train_selected, index = self.X_train.index)
            X_test_selected = pd.DataFrame(X_test_selected, index = self.X_test.index)


            return X_train_selected, X_test_selected,self.y_train, self.y_test
        else:
            return self.X_train, self.X_test,self.y_train, self.y_test

    def recursive_feature_elimination(self):
        n_features_to_select = 5
        if self.X_train.shape[1]>n_features_to_select:

            estimator = RandomForestClassifier()  # You can use a different estimator
            selector = RFE(estimator)

            try:
                selector.fit(self.X_train, self.y_train)

            except ValueError as e:

                self.y_classes = np.where(self.y_train > np.median(self.y_train), 1, 0)
                selector.fit(self.X_train, self.y_classes)

            X_train_selected = selector.transform(self.X_train)
            X_test_selected = selector.transform(self.X_test)

            X_train_selected = pd.DataFrame(X_train_selected, index=self.X_train.index)
            X_test_selected = pd.DataFrame(X_test_selected, index=self.X_test.index)

            return X_train_selected, X_test_selected,self.y_train, self.y_test

        else:
            return self.X_train, self.X_test,self.y_train, self.y_test

    def method_transform(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        if self.selector_type == 'select_k_best':
            return self.select_k_best()

        elif self.selector_type == 'recursive_feature_elimination':
            return self.recursive_feature_elimination()

        else:

            return self.X_train, self.X_test, self.y_train, self.y_test

