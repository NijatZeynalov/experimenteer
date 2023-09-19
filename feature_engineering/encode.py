# 'one-hot-encoding','ordinal-encoding','frequency_encoding'
# 'mean_encoding',
# 'weight_of_evidence',
# 'ohe_frequent_categories'
from feature_engine.encoding import MeanEncoder
from feature_engine.encoding import OneHotEncoder
from feature_engine.encoding import OrdinalEncoder
from feature_engine.encoding import CountFrequencyEncoder
from feature_engine.encoding import WoEEncoder
import warnings

class Encoder:

    def __init__(self, encoder_type):
        self.encoder_type = encoder_type

    def one_hot_encoder(self):
        encoder = OneHotEncoder(
            variables=None,  # alternatively pass a list of variables
            drop_last=True,  # to return k-1, use drop=false to return k dummies
        )

        encoder.fit(self.X_train)
        X_train_enc = encoder.transform(self.X_train)
        X_test_enc = encoder.transform(self.X_test)
        return X_train_enc, X_test_enc,self.y_train, self.y_test

    def ordinal_encoder(self):
        encoder = OrdinalEncoder(encoding_method="arbitrary", unseen = 'encode')
        encoder.fit(self.X_train)
        X_train_enc = encoder.transform(self.X_train)
        X_test_enc = encoder.transform(self.X_test)
        return X_train_enc, X_test_enc,self.y_train, self.y_test

    def frequency_encoder(self):
        encoder = CountFrequencyEncoder(encoding_method="frequency", unseen = 'encode')
        encoder.fit(self.X_train)
        X_train_enc = encoder.transform(self.X_train)
        X_test_enc = encoder.transform(self.X_test)

        return X_train_enc, X_test_enc,self.y_train, self.y_test

    def mean_encoder(self):
        encoder = MeanEncoder(unseen = 'encode')
        encoder.fit(self.X_train, self.y_train)
        X_train_enc = encoder.transform(self.X_train)
        X_test_enc = encoder.transform(self.X_test)

        return X_train_enc, X_test_enc,self.y_train, self.y_test


    def ohe_frequent_encoder(self):
        encoder = OneHotEncoder(top_categories=10)
        encoder.fit(self.X_train)
        X_train_enc = encoder.transform(self.X_train)
        X_test_enc = encoder.transform(self.X_test)
        return X_train_enc, X_test_enc,self.y_train, self.y_test

    def method_transform(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        try:
            if self.encoder_type == 'one_hot_encoder':
                return self.one_hot_encoder()
            elif self.encoder_type == 'ordinal_encoder':
                return self.ordinal_encoder()
            elif self.encoder_type == 'frequency_encoder':
                return self.frequency_encoder()
            elif self.encoder_type == 'mean_encoder':
                return self.mean_encoder()
            elif self.encoder_type == 'woe_encoder':
                return self.woe_encoder()
            elif self.encoder_type == 'ohe_frequent_encoder':
                return self.ohe_frequent_encoder()
            else:
                raise ValueError(f"Invalid encoder_type: {self.encoder_type}")
        except:
            return self.X_train, self.X_test,self.y_train, self.y_test
