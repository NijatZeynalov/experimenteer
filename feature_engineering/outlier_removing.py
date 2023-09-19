from feature_engine.outliers import OutlierTrimmer


class OutlierRemover:

    def __init__(self, trimmer_type):
        self.trimmer_type = trimmer_type



    def removing_outliers_iqr(self):
        try:
            trimmer = OutlierTrimmer(
                capping_method="iqr",
                tail="both",
                fold=1.5,
            )

            threshold = 3
            low_variation_columns = []

            # Loop through columns
            for column in self.X_train.columns:
                unique_count = self.X_train[column].nunique()
                if unique_count <= threshold:
                    low_variation_columns.append(column)
            outlier_columns = self.X_train.drop(low_variation_columns, axis=1)
            trimmer.fit(outlier_columns)

            X_train = trimmer.transform(outlier_columns)
            X_test = trimmer.transform(self.X_test[outlier_columns.columns])

            X_train_enc = self.X_train[self.X_train[low_variation_columns].index.isin(X_train.index)]
            X_test_enc = self.X_test[self.X_test[low_variation_columns].index.isin(X_test.index)]

            y_train = self.y_train[self.y_train.index.isin(X_train_enc.index)]
            y_test = self.y_test[self.y_test.index.isin(X_test_enc.index)]

            return X_train_enc, X_test_enc, y_train, y_test
        except:
            return self.X_train, self.X_test, self.y_train, self.y_test

    def removing_outliers_quantiles(self):

        try:
            trimmer = OutlierTrimmer(
                capping_method="quantiles",
                tail="both",
                fold=0.05,
            )

            threshold = 2
            low_variation_columns = []

            # Loop through columns
            for column in self.X_train.columns:
                unique_count = self.X_train[column].nunique()
                if unique_count <= threshold:
                    low_variation_columns.append(column)
            outlier_columns = self.X_train.drop(low_variation_columns, axis=1)
            trimmer.fit(outlier_columns)
            X_train = trimmer.transform(outlier_columns)
            X_test = trimmer.transform(self.X_test[outlier_columns.columns])

            X_train_enc = self.X_train[self.X_train[low_variation_columns].index.isin(X_train.index)]
            X_test_enc = self.X_test[self.X_test[low_variation_columns].index.isin(X_test.index)]

            y_train = self.y_train[self.y_train.index.isin(X_train_enc.index)]
            y_test = self.y_test[self.y_test.index.isin(X_test_enc.index)]

            return X_train_enc, X_test_enc, y_train, y_test
        except:
            return self.X_train, self.X_test, self.y_train, self.y_test

    def method_transform(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        if self.trimmer_type == 'removing_outliers_iqr':
            return self.removing_outliers_iqr()
        elif self.trimmer_type == 'removing_outliers_quantiles':
            return self.removing_outliers_quantiles()

        else:
            return self.X_train, self.X_test, self.y_train, self.y_test