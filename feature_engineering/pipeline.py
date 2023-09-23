import nbformat as nbf

missing_dict = {
    "mean_imputer": [
        "from feature_engine.imputation import MeanMedianImputer",
        """imputer = MeanMedianImputer(imputation_method="mean")
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

imputer = CategoricalImputer(imputation_method="frequent")
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)""",
    ],
    "arbitrary_value_imputer": [
        "from feature_engine.imputation import ArbitraryNumberImputer",
        """imputer = ArbitraryNumberImputer(arbitrary_number=-999)
 imputer.fit(X_train)
 X_train= imputer.transform(X_train)
 X_test = imputer.transform(X_test)

 imputer = CategoricalImputer(imputation_method="frequent")
             imputer.fit(X_train)
 X_train = imputer.transform(X_train)
 X_test = imputer.transform(X_test)""",
    ],
    "missing_indicator_imputer": [
        "from feature_engine.imputation import CategoricalImputer, AddMissingIndicator",
        """imputer = AddMissingIndicator(missing_only=True)
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

imputer = CategoricalImputer(imputation_method="frequent")
           imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)""",
    ],
    "missing_category_imputer": [
        "from feature_engine.imputation import CategoricalImputer",
        """imputer = CategoricalImputer()
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

imputer = CategoricalImputer(imputation_method="frequent")
          imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)""",
    ],
    "knn_imputer": [
        "from sklearn.impute import KNNImputer",
        """imputer = KNNImputer(
 n_neighbors=5,  # the number of neighbours K
 weights='distance',  # the weighting factor
 metric='nan_euclidean',  # the metric to find the neighbours
 add_indicator=False,  # whether to add a missing indicator
)
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

imputer = CategoricalImputer(imputation_method="frequent")
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)""",
    ],
    "iterative_imputer": [
        """from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer""",
        """imputer = IterativeImputer(
               estimator=BayesianRidge(),  # the estimator to predict the NA
               initial_strategy='mean',  # how will NA be imputed in step 1
               max_iter=10,  # number of cycles
               imputation_order='ascending',  # the order in which to impute the variables
               n_nearest_features=None,  # whether to limit the number of predictors
               skip_complete=True,  # whether to ignore variables without NA
               random_state=0,
               )
imputer.fit(X_train)

X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

imputer = CategoricalImputer(imputation_method="frequent")
           imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)""",
    ],
}

encode_dict = {
    "one_hot_encoder": [
        "from feature_engine.encoding import OneHotEncoder",
        """encoder = OneHotEncoder(
      variables=None,  # alternatively pass a list of variables
      drop_last=True,  # to return k-1, use drop=false to return k dummies
  )
  encoder.fit(X_train)
  X_train= encoder.transform(X_train)
  X_test = encoder.transform(X_test)""",
    ],
    "ordinal_encoder": [
        "from feature_engine.encoding import OrdinalEncoder",
        """encoder = OrdinalEncoder(encoding_method="arbitrary")
encoder.fit(X_train)
X_train= encoder.transform(X_train)
X_test = encoder.transform(X_test)""",
    ],
    "frequency_encoder": [
        "from feature_engine.encoding import CountFrequencyEncoder",
        """encoder = CountFrequencyEncoder(encoding_method="frequency")
  encoder.fit(X_train)
  X_train= encoder.transform(X_train)
  X_test = encoder.transform(X_test)""",
    ],
    "mean_encoder": [
        "from feature_engine.encoding import MeanEncoder",
        """encoder = MeanEncoder()
encoder.fit(X_train)
X_train= encoder.transform(X_train)
X_test = encoder.transform(X_test)""",
    ],
    "woe_encoder": [
        "from feature_engine.encoding import WoEEncoder",
        """encoder  = WoEEncoder()
encoder.fit(X_train)
X_train= encoder.transform(X_train)
X_test = encoder.transform(X_test)""",
    ],
    "ohe_frequent_encoder": [
        """from feature_engine.encoding import OneHotEncoder
""",
        """encoder = OneHotEncoder(top_categories=10)
 encoder.fit(X_train)
 X_train= encoder.transform(X_train)
 X_test = encoder.transform(X_test)""",
    ],
}

outlier_dict = {
    "removing_outliers_iqr": [
        "from feature_engine.outliers import OutlierTrimmer",
        """trimmer = OutlierTrimmer(
capping_method="iqr",
tail="both",
fold=1.5,
)

trimmer.fit(X_train)
X_train = trimmer.transform(X_train)
X_test = trimmer.transform(X_test)
y_train = y_train[y_train.index.isin(X_train.index)]
y_test = y_test[y_test.index.isin(X_test.index)]""",
    ],
    "none": ["", ""],
    "removing_outliers_quantiles": [
        "from feature_engine.outliers import OutlierTrimmer",
        """trimmer = OutlierTrimmer(
capping_method="quantiles",
tail="both",
fold=0.05,
)

trimmer.fit(X_train)
X_train = trimmer.transform(X_train)
X_test = trimmer.transform(X_test)
y_train = y_train[y_train.index.isin(X_train.index)]
y_test = y_test[y_test.index.isin(X_test.index)]""",
    ],
}

feature_selection_dict = {
    "select_k_best": [
        "from sklearn.feature_selection import SelectKBest, RFE",
        """selector = SelectKBest(k=5)
selector.fit(X_train, y_train)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

X_train = pd.DataFrame(X_train_selected, index = X_train.index)
X_test = pd.DataFrame(X_test_selected, index = X_test.index)
""",
    ],
    "recursive_feature_elimination": [
        "from sklearn.feature_selection import SelectKBest, RFE",
        """estimator = RandomForestClassifier()  # You can use a different estimator
selector = RFE(estimator)

try:
    selector.fit(X_train, y_train)

except ValueError as e:

    y_classes = np.where(y_train > np.median(y_train), 1, 0)
    selector.fit(X_train, y_classes)

X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

""",
    ],
    "none": ["", ""],
}

scaling_dict = {
    "standard": [
        "from sklearn.preprocessing import StandardScaler",
        """
scaler = StandardScaler().set_output(transform="pandas")
scaler.fit(X_train)

# transform train and test sets
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)
""",
    ],
    "min_max": [
        "from sklearn.preprocessing import MinMaxScaler",
        """
scaler = MinMaxScaler().set_output(transform="pandas")
scaler.fit(X_train)

# transform train and test sets
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)
""",
    ],
    "none": ["", ""],
    "robust": [
        "from sklearn.preprocessing import RobustScaler",
        """
scaler = RobustScaler().set_output(transform="pandas")
scaler.fit(X_train)

# transform train and test sets
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)
""",
    ],
}


def notebook_generator(dataset, target, missing, encode, outlier, selection, scaling, model):
    nb = nbf.v4.new_notebook()
    text_1 = """# 1. Import libraries."""

    code_1 = f"""import pandas as pd
                from sklearn.model_selection import train_test_split
                {missing_dict[missing][0]}
                {encode_dict[encode][0]}
                {outlier_dict[outlier][0]}
                {feature_selection_dict[selection][0]}
                {scaling_dict[scaling][0]}"""

    text_2 = """# 2. Train test split"""

    code_2 = f"""df = pd.read_csv('{dataset}')
                X = df.drop(columns='{target}')
                y = df['{target}']
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=0,
                )"""

    text_3 = """# 3. Missing value imputation"""

    code_3 = f"""{missing_dict[missing][1]}"""

    text_4 = """# 4. Categorical Encoding"""

    code_4 = f"""{encode_dict[encode][1]}"""

    text_5 = """# 5. Outlier removing"""

    code_5 = f"""{outlier_dict[outlier][1]}"""

    text_6 = """# 6. Feature selection"""

    code_6 = f"""{feature_selection_dict[selection][1]}"""

    text_7 = """# 7. Scaling"""

    code_7 = f"""{scaling_dict[scaling][1]}"""

    text_8 = """# 8. Modeling"""

    code_8 = f"""model = {model}\nmodel.fit(X_train,y_train)\nmodel.predict(X_test)"""

    nb["cells"] = [
        nbf.v4.new_markdown_cell(text_1),
        nbf.v4.new_code_cell(code_1),
        nbf.v4.new_markdown_cell(text_2),
        nbf.v4.new_code_cell(code_2),
        nbf.v4.new_markdown_cell(text_3),
        nbf.v4.new_code_cell(code_3),
        nbf.v4.new_markdown_cell(text_4),
        nbf.v4.new_code_cell(code_4),
        nbf.v4.new_markdown_cell(text_5),
        nbf.v4.new_code_cell(code_5),
        nbf.v4.new_markdown_cell(text_6),
        nbf.v4.new_code_cell(code_6),
        nbf.v4.new_markdown_cell(text_7),
        nbf.v4.new_code_cell(code_7),
        nbf.v4.new_markdown_cell(text_8),
        nbf.v4.new_code_cell(code_8),
    ]
    fname = "jupyter_notebook.ipynb"

    with open(fname, "w") as f:
        nbf.write(nb, f)


class CustomPipeline:
    def __init__(self, steps, X_train, X_test, y_train, y_test):
        self.steps = steps
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit_transform(self):
        X_train_t = self.X_train
        X_test_t = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        for step in self.steps:
            X_train_t, X_test_t, y_train, y_test = step.fit_transform(
                X_train_t, X_test_t, y_train, y_test
            )

        return X_train_t, X_test_t, y_train, y_test


class DatasetCleaner:
    def __init__(self, df):
        self.df = df

    def drop_duplicates(self):
        self.df = self.df.drop_duplicates()

    def drop_unique_features(self):
        """Drops unique features from the dataset."""

        unique_features = [column for column in self.df.columns if self.df[column].nunique() == 1]
        self.df = self.df.drop(unique_features, axis=1)

    def clean(self):
        """Cleans the dataset by dropping duplicates, collinear features, unique features,
        and other features as specified."""

        self.drop_duplicates()
        self.drop_unique_features()

        return self.df
