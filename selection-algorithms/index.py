import pandas as pd
import numpy as np
from FeatureSelection import FeatureSelection

num_feats = 5
targets = ['BIS attentional', 'BIS motor',	'BIS nonplanning',
           'BIS total',	'BIS 2F inhibitory control',	'BIS 2F nonplanning']
df = pd.read_csv("./teste/Group1_BancoTCE_24h_FUP_Impulsividade.csv")

for target in targets:
    select_target = target

    featureSelection = FeatureSelection(df, targets, num_feats, select_target)
    person_correlation = featureSelection.person_correlation()
    chi_squared = featureSelection.chi_squared()
    recursive_feature_elimination = featureSelection.recursive_feature_elimination()
    lasso = featureSelection.lasso()
    random_forest = featureSelection.random_forest()

    pd.set_option('display.max_rows', None)
    # put all selection together
    feature_selection_df = pd.DataFrame({
        'Feature': select_target,
        'Pearson': person_correlation,
        'Chi-2': chi_squared,
        'RFE': recursive_feature_elimination,
        'Logistics': lasso,
        'Random Forest': random_forest,
        # 'LightGBM': embeded_lgb_support - não rodou
    })
    # count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(
        ['Total', 'Feature'], ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)

    print(feature_selection_df.head(num_feats))

# Verificar o total
