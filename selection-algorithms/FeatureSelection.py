import numpy as np

from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso, LinearRegression


# SEED = 15452
# random.seed(SEED)
class FeatureSelection:
    def __init__(self, df, targets, num_feats, select_target):
        self.num_feats = num_feats
        self.targets = targets
        self.df = df
        self.X = self.df.drop(columns=self.targets)
        self.select_target = select_target
        self.target = self.df[self.select_target]

        # print(df.isnull().sum())
        # train_x, train_y, test_x, test_y = train_test_split(
        #    df, df[select_target])

    def random_forest(self):

        embeded_rf_selector = SelectFromModel(
            RandomForestRegressor(n_estimators=100), max_features=self.num_feats)
        embeded_rf_selector.fit(self.X, self.target)

        embeded_rf_support = embeded_rf_selector.get_support()
        embeded_rf_feature = self.X.loc[:, embeded_rf_support].columns.tolist()
        # print(str(len(embeded_rf_feature)),
        # 'selected features to', self.select_target)
        print(embeded_rf_support, '\n\n')

        return embeded_rf_support

    def person_correlation(self):
        feat_list = []
        feature_name = self.X.columns.tolist()
        # calculate the correlation with y for each feature
        for i in self.X.columns.tolist():
            feat = np.corrcoef(self.X[i], self.target)[0, 1]
            feat_list.append(feat)
        # replace NaN with 0
        feat_list = [0 if np.isnan(i) else i for i in feat_list]
        # feature name
        feat_feature = self.X.iloc[:, np.argsort(
            np.abs(feat_list))[-self.num_feats:]].columns.tolist()
        # feature selection? 0 for not select, 1 for select
        feat_support = [
            True if i in feat_feature else False for i in feature_name]

        print(str(len(feat_feature)),
              'selected features to', self.select_target)
        print(feat_feature, '\n\n')
        print(feat_support, '\n\n')

        return feat_support

    def chi_squared(self):
        X_norm = MinMaxScaler().fit_transform(self.X)
        chi_selector = SelectKBest(chi2, k=self.num_feats)
        chi_selector.fit(X_norm, self.target)
        chi_support = chi_selector.get_support()
        chi_feature = self.X.loc[:, chi_support].columns.tolist()
        # print(str(len(chi_feature)),
        # 'selected features to', self.select_target)
        # print(chi_feature, '\n\n')

        return chi_support

    def recursive_feature_elimination(self):
        rfe_selector = RFE(estimator=LinearRegression(),
                           n_features_to_select=self.num_feats, step=10, verbose=5)
        X_norm = MinMaxScaler().fit_transform(self.X)
        rfe_selector.fit(X_norm, self.target)
        rfe_support = rfe_selector.get_support()
        rfe_feature = self.X.loc[:, rfe_support].columns.tolist()

        # print(str(len(rfe_feature)),
        # 'selected features to', self.select_target)
        # print(rfe_feature, '\n\n')

        return rfe_support

    def lasso(self):
        X_norm = MinMaxScaler().fit_transform(self.X)

        embeded_lr_selector = SelectFromModel(
            Lasso(), max_features=self.num_feats)
        embeded_lr_selector.fit(X_norm, self.target)

        embeded_lr_support = embeded_lr_selector.get_support()
        embeded_lr_feature = self.X.loc[:, embeded_lr_support].columns.tolist()
        print(str(len(embeded_lr_feature)),
              'selected features to', self.select_target)
        print(embeded_lr_feature, '\n\n')
        print(embeded_lr_support, '\n\n')

        return embeded_lr_support
