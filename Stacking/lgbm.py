import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from utils import Transformation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import glob

import warnings
warnings.filterwarnings('ignore')

def run_training(pred_df, fold):

    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[['ext_pred', 'rf_pred', 'xgb_pred', 'lr_pred']].values
    xvalid = train_df[['ext_pred', 'rf_pred', 'xgb_pred', 'lr_pred']].values
    ytrain = train_df['Category'].values
    yvalid = valid_df['Category'].values

    scaler = StandardScaler()

    lgbm = LGBMClassifier()
    lgbm_pipe = make_pipeline(scaler, lgbm)
    lgbm_pipe.fit(xtrain, ytrain)
    pred = lgbm_pipe.predict_proba(xvalid)
    loss = log_loss(yvalid, pred)
    print(f"fold={fold}, loss={loss}")
    pred = list(pred)
    valid_df["lgbm_pred"] = pred

    return valid_df

if __name__ == "__main__":

    files = glob.glob("../model_preds/*.csv")
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on="id", how="left")
    targets = df['Category'].values
    pred_cols = ['ext_pred', 'rf_pred', 'xgb_pred', 'lr_pred']

    dfs = []
    for j in range(5):
        temp_df = run_training(df, j)
        dfs.append(temp_df)

    fin_valid_df = pd.concat(dfs)
    print(log_loss(fin_valid_df['Category'].values, fin_valid_df['lgbm_pred'].values))