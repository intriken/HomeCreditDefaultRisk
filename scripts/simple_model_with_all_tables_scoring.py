# Basic Kernel or reference: https://www.kaggle.com/kailex/tidy-xgb-0-778/code

import numpy as np
import pandas as pd
import gc
import lightgbm as gbm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

print("Loading data...\n")
lb = LabelEncoder()


def LabelEncoding_Cat(df):
    df = df.copy()
    Cat_Var = df.select_dtypes("object").columns.tolist()
    for col in Cat_Var:
        df[col] = lb.fit_transform(df[col].astype("str"))
    return df


def Fill_NA(df):
    df = df.copy()
    Num_Features = df.select_dtypes(["float64", "int64"]).columns.tolist()
    df[Num_Features] = df[Num_Features].fillna(-999)
    return df


bureau = pd.read_csv("../input/bureau.csv").pipe(LabelEncoding_Cat)

cred_card_bal = pd.read_csv("../input/credit_card_balance.csv").pipe(LabelEncoding_Cat)

pos_cash_bal = pd.read_csv("../input/POS_CASH_balance.csv").pipe(LabelEncoding_Cat)

prev = pd.read_csv("../input/previous_application.csv").pipe(LabelEncoding_Cat)

print("Preprocessing...\n")
Label_1 = [
    s + "_" + l
    for s in bureau.columns.tolist()
    if s != "SK_ID_CURR"
    for l in ["mean", "count", "median", "max"]
]
avg_bureau = (
    bureau.groupby("SK_ID_CURR").agg(["mean", "count", "median", "max"]).reset_index()
)
avg_bureau.columns = ["SK_ID_CURR"] + Label_1

Label_2 = [
    s + "_" + l
    for s in cred_card_bal.columns.tolist()
    if s != "SK_ID_CURR"
    for l in ["mean", "count", "median", "max"]
]
avg_cred_card_bal = (
    cred_card_bal.groupby("SK_ID_CURR")
    .agg(["mean", "count", "median", "max"])
    .reset_index()
)
avg_cred_card_bal.columns = ["SK_ID_CURR"] + Label_2

Label_3 = [
    s + "_" + l
    for s in pos_cash_bal.columns.tolist()
    if s not in ["SK_ID_PREV", "SK_ID_CURR"]
    for l in ["mean", "count", "median", "max"]
]
avg_pos_cash_bal = (
    pos_cash_bal.groupby(["SK_ID_PREV", "SK_ID_CURR"])
    .agg(["mean", "count", "median", "max"])
    .groupby(level="SK_ID_CURR")
    .agg("mean")
    .reset_index()
)
avg_pos_cash_bal.columns = ["SK_ID_CURR"] + Label_3

Label_4 = [
    s + "_" + l
    for s in prev.columns.tolist()
    if s != "SK_ID_CURR"
    for l in ["mean", "count", "median", "max"]
]
avg_prev = (
    prev.groupby("SK_ID_CURR").agg(["mean", "count", "median", "max"]).reset_index()
)
avg_prev.columns = ["SK_ID_CURR"] + Label_4

del (Label_1, Label_2, Label_3, Label_4)
te = pd.read_csv("../input/application_test.csv")

tri = te.shape[0]

te_labeled = (
    te.pipe(LabelEncoding_Cat)
    .pipe(Fill_NA)
    .merge(avg_bureau, on="SK_ID_CURR", how="left")
    .merge(avg_cred_card_bal, on="SK_ID_CURR", how="left")
    .merge(avg_pos_cash_bal, on="SK_ID_CURR", how="left")
    .merge(avg_prev, on="SK_ID_CURR", how="left")
)

del (
    bureau,
    cred_card_bal,
    pos_cash_bal,
    prev,
    avg_prev,
    avg_bureau,
    avg_cred_card_bal,
    avg_pos_cash_bal,
)
gc.collect()

print("Preparing data...\n")
te_labeled.drop(labels=["SK_ID_CURR"], axis=1, inplace=True)

print("Scoring ...\n")


with open("../pickle/risk_model.pickle", "rb") as handle:
    m_gmm_pickle = pickle.load(handle)

oof_preds_pickle_test = np.zeros(te_labeled.shape[0])

oof_preds_pickle_test = m_gmm_pickle.predict(te_labeled)


print("Output data")
te["TARGET_oof"] = oof_preds_pickle_test.copy()
te.to_csv("../output/application_test_with_output.csv", index=False)
