import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
import pandas as pd

df = pd.read_csv("./teste/Group1_BancoTCE_24h_FUP_Impulsividade.csv")

criterions = ['BIS attentional', 'BIS motor', 'BIS nonplanning']
num_feats = 5

for criterion in criterions:
    lgbc = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
                              reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

    embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
    embeded_lgb_selector.fit(df, df[criterion])

    embeded_lgb_support = embeded_lgb_selector.get_support()
    embeded_lgb_feature = df.loc[:, embeded_lgb_support].columns.tolist()
    print(str(len(embeded_lgb_feature)), 'selected features to', criterion)
    print(embeded_lgb_feature, '\n\n')
