from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import ArbitraryNumberImputer
from feature_engine.imputation import CategoricalImputer, AddMissingIndicator
from sklearn.impute import KNNImputer
from sklearn.linear_model import BayesianRidge
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class Imputer:

    def __init__(self, imputer_type):
        self.imputer_type = imputer_type

    def mean_imputer(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.frequent_category_imputer()
        imputer = MeanMedianImputer(imputation_method="mean")
        imputer.fit(self.X_train)
        X_train_t = imputer.transform(self.X_train)
        X_test_t = imputer.transform(self.X_test)
        X_test_t = X_test_t.dropna()
        print(X_test_t.isnull().sum())
        self.y_test = self.y_test[self.y_test.index.isin(X_test_t.index)]
        return X_train_t, X_test_t,self.y_train, self.y_test

    def arbitrary_value_imputer(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.frequent_category_imputer()

        imputer = ArbitraryNumberImputer(arbitrary_number=-999)
        imputer.fit(self.X_train)
        X_train_t = imputer.transform(self.X_train)
        X_test_t = imputer.transform(self.X_test)
        return X_train_t, X_test_t,self.y_train, self.y_test

    def frequent_category_imputer(self):
            categorical_columns = self.X_train.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                # Calculate value counts
                value_counts = self.X_train[column].value_counts()

                # Check if there are missing values in the column
                if self.X_train[column].isnull().any():
                    # Replace missing values with the first index value (most frequent category)
                    most_frequent_category = value_counts.index[0]
                    self.X_train[column].fillna(most_frequent_category, inplace=True)
                    self.X_test[column].fillna(most_frequent_category, inplace=True)
            return self.X_train, self.X_test, self.y_train, self.y_test


    def missing_category_imputer(self):
        try:
            imputer = CategoricalImputer()
            imputer.fit(self.X_train)
            X_train_t = imputer.transform(self.X_train)
            X_test_t = imputer.transform(self.X_test)
            return X_train_t, X_test_t,self.y_train, self.y_test
        except:
            return self.X_train, self.X_test, self.y_train, self.y_test

    def missing_indicator_imputer(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.frequent_category_imputer()

        imputer = AddMissingIndicator(missing_only=True)
        imputer.fit(self.X_train)
        X_train_t = imputer.transform(self.X_train)
        X_test_t = imputer.transform(self.X_test)
        return X_train_t, X_test_t,self.y_train, self.y_test

    def knn_imputer(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.frequent_category_imputer()
        numeric_columns = self.X_train.select_dtypes(include=['number'])
        categorical_columns = self.X_train.select_dtypes(exclude=['number'])

        # Apply KNN imputation to numeric columns
        imputer = KNNImputer(
            n_neighbors=5,  # the number of neighbors K
            weights='distance',  # the weighting factor
            metric='nan_euclidean',  # the metric to find the neighbors
            add_indicator=False,  # whether to add a missing indicator
        )
        imputer.fit(numeric_columns)
        train_numeric_imputed = imputer.transform(numeric_columns)
        test_numeric_imputed = imputer.transform(self.X_test[numeric_columns.columns])

        # Convert the imputed results back to DataFrames
        train_numeric_imputed = pd.DataFrame(train_numeric_imputed, columns=numeric_columns.columns)
        test_numeric_imputed = pd.DataFrame(test_numeric_imputed, columns=numeric_columns.columns)
        train_numeric_imputed.index = self.X_train.index
        test_numeric_imputed.index = self.X_test.index

        # Combine imputed numeric columns with the original categorical columns
        train_combined = pd.concat([train_numeric_imputed, categorical_columns], axis=1)
        test_combined = pd.concat([test_numeric_imputed, self.X_test[categorical_columns.columns]], axis=1)

        return train_combined, test_combined,self.y_train, self.y_test

    def iterative_imputer(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.frequent_category_imputer()
        numeric_columns = self.X_train.select_dtypes(include=['number'])
        categorical_columns = self.X_train.select_dtypes(exclude=['number'])

        imputer = IterativeImputer(
            estimator=BayesianRidge(),  # the estimator to predict the NA
            initial_strategy='mean',  # how will NA be imputed in step 1
            max_iter=10,  # number of cycles
            imputation_order='ascending',  # the order in which to impute the variables
            n_nearest_features=None,  # whether to limit the number of predictors
            skip_complete=True,  # whether to ignore variables without NA
            random_state=0,
        )
        imputer.fit(numeric_columns)
        train_numeric_imputed = imputer.transform(numeric_columns)
        test_numeric_imputed = imputer.transform(self.X_test[numeric_columns.columns])

        # Convert the imputed results back to DataFrames
        train_numeric_imputed = pd.DataFrame(train_numeric_imputed, columns=numeric_columns.columns)
        test_numeric_imputed = pd.DataFrame(test_numeric_imputed, columns=numeric_columns.columns)
        train_numeric_imputed.index = self.X_train.index
        test_numeric_imputed.index = self.X_test.index

        # Combine imputed numeric columns with the original categorical columns
        train_combined = pd.concat([train_numeric_imputed, categorical_columns], axis=1)
        test_combined = pd.concat([test_numeric_imputed, self.X_test[categorical_columns.columns]], axis=1)

        return train_combined, test_combined,self.y_train, self.y_test

    def method_transform(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        if self.imputer_type == 'mean_imputer':
            return self.mean_imputer()
        elif self.imputer_type == 'arbitrary_value_imputer':
            return self.arbitrary_value_imputer()
        elif self.imputer_type == 'missing_indicator_imputer':
            return self.missing_indicator_imputer()
        elif self.imputer_type == 'knn_imputer':
            return self.knn_imputer()
        elif self.imputer_type == 'iterative_imputer':
            return self.iterative_imputer()
        else:
            raise ValueError(f"Invalid imputer_type: {self.imputer_type}")
