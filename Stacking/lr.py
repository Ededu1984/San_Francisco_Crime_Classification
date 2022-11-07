import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from utils import Transformation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectPercentile, chi2

import warnings
warnings.filterwarnings('ignore')

def run_training(fold):
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

    lr = LogisticRegression()
    lr_pipe = make_pipeline(scaler, selection, lr)
    lr_pipe.fit(xtrain, ytrain)

    pred = lr_pipe.predict_proba(xvalid)
    loss = log_loss(yvalid, pred)
    print(f"fold={fold}, loss={loss}")

    prob_list = []
    for i in pred:
        max_prob = i.max()
        probs = list(i)
        prob = probs.index(max_prob)
        prob_list.append(prob)
   
    df_valid["lr_pred"] = prob_list

    return df_valid[['id', 'Category','kfold', 'lr_pred']]

if __name__ == "__main__":
    print("-"*80)
    print("Logistic Regression")
    dfs = []
    for j in range(5):
        temp_df = run_training(j)
        dfs.append(temp_df)

    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)
    fin_valid_df.to_csv("../model_preds/lr.csv", index=False)
    print("-"*80)