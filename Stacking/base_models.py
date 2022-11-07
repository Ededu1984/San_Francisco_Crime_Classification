import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectPercentile, chi2

import warnings
warnings.filterwarnings('ignore')

def run_training(fold: int, model: str) -> pd.DataFrame:
    df = pd.read_csv("../datasets/train_folds.csv")
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    xtrain = df_train.drop('Category', axis=1)
    xvalid = df_valid.drop('Category', axis=1)
    ytrain = df_train['Category']
    yvalid = df_valid['Category']
    
    scaler = MaxAbsScaler()
    selection = SelectPercentile(chi2, percentile=50)
    xvalid = scaler.fit_transform(xvalid)

    if model == 'Random_Forest':
      clf = RandomForestClassifier(max_depth=8, n_estimators=100)
      column_name = 'rf_pred'
      
    elif model == 'LightGBM':
      clf = LGBMClassifier()
      column_name = 'lgbm_pred'

    elif model == 'Logistic_Regression':
      clf = LogisticRegression()
      column_name = 'lr_pred'
    else:
      clf = ExtraTreesClassifier()
      column_name = 'ext_pred'


    clf_pipe = make_pipeline(scaler, selection, clf)
    clf_pipe.fit(xtrain, ytrain)  
    pred = clf_pipe.predict_proba(xvalid)
    loss = log_loss(yvalid, pred)
    print(f"fold={fold}, loss={loss}")

    prob_list = []
    for i in pred:
        max_prob = i.max()
        probs = list(i)
        prob = probs.index(max_prob)
        prob_list.append(prob)

    df_valid[column_name] = prob_list

    return df_valid[['id', 'Category','kfold', column_name]]

if __name__ == "__main__":
    base_models = ['Random_Forest', 'LightGBM', 'Logistic_Regression', 'Extra_Trees_Classifier']

    for n, i in enumerate(base_models):
        print("-"*80)
        print(f"Base model {n+1} - " + i)
        model = i
        dfs = []
        for j in range(5):
            temp_df = run_training(j, model)
            dfs.append(temp_df)

        fin_valid_df = pd.concat(dfs)
        print(fin_valid_df.shape)
        fin_valid_df.to_csv("../model_preds/" + i + ".csv", index=False)
    print("-"*80)