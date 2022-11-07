import pandas as pd
from sklearn.model_selection import StratifiedKFold


if __name__ == "__main__":
    df = pd.read_csv("../datasets/train_encoded.csv")
    df.loc[:, "kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    df=df.reset_index().rename(columns={'index': 'id'})
    y = df.Category.values
    skf = StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(skf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f

    df.to_csv("../datasets/train_folds.csv", index=False)