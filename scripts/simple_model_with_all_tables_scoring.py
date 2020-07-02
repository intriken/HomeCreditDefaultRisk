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


def LabelEncodingCategory(df):
    df = df.copy()
    category_variables = df.select_dtypes("object").columns.tolist()
    for col in category_variables:
        df[col] = lb.fit_transform(df[col].astype("str"))
    return df


def FillNA(df):
    df = df.copy()
    Num_Features = df.select_dtypes(["float64", "int64"]).columns.tolist()
    df[Num_Features] = df[Num_Features].fillna(-999)
    return df


# pull in data from csvs
bureau = pd.read_csv("../input/bureau.csv").pipe(LabelEncodingCategory)

creditCardBal = pd.read_csv("../input/credit_card_balance.csv").pipe(
    LabelEncodingCategory
)

posCashBal = pd.read_csv("../input/posCashBalance.csv").pipe(LabelEncodingCategory)

previousApp = pd.read_csv("../input/previous_application.csv").pipe(
    LabelEncodingCategory
)

print("Preprocessing...\n")

# creating new features
labels1 = [
    s + "_" + l
    for s in bureau.columns.tolist()
    if s != "SK_ID_CURR"
    for l in ["mean", "count", "median", "max"]
]
avgBureau = (
    bureau.groupby("SK_ID_CURR").agg(["mean", "count", "median", "max"]).reset_index()
)
avgBureau.columns = ["SK_ID_CURR"] + labels1

labels2 = [
    s + "_" + l
    for s in creditCardBal.columns.tolist()
    if s != "SK_ID_CURR"
    for l in ["mean", "count", "median", "max"]
]
avgCreditCardBal = (
    creditCardBal.groupby("SK_ID_CURR")
    .agg(["mean", "count", "median", "max"])
    .reset_index()
)
avgCreditCardBal.columns = ["SK_ID_CURR"] + labels2

labels3 = [
    s + "_" + l
    for s in posCashBal.columns.tolist()
    if s not in ["SK_ID_PREV", "SK_ID_CURR"]
    for l in ["mean", "count", "median", "max"]
]
avgPosCashBal = (
    posCashBal.groupby(["SK_ID_PREV", "SK_ID_CURR"])
    .agg(["mean", "count", "median", "max"])
    .groupby(level="SK_ID_CURR")
    .agg("mean")
    .reset_index()
)
avgPosCashBal.columns = ["SK_ID_CURR"] + labels3

labels4 = [
    s + "_" + l
    for s in previousApp.columns.tolist()
    if s != "SK_ID_CURR"
    for l in ["mean", "count", "median", "max"]
]
avgPreviousApp = (
    previousApp.groupby("SK_ID_CURR")
    .agg(["mean", "count", "median", "max"])
    .reset_index()
)
avgPreviousApp.columns = ["SK_ID_CURR"] + labels4

del (labels1, labels2, labels3, labels4)

# read in data for running scorring on test data
te = pd.read_csv("../input/application_test.csv")

tri = te.shape[0]

# cleaning data for input into model
teLabeled = (
    te.pipe(LabelEncodingCategory)
    .pipe(FillNA)
    .merge(avgBureau, on="SK_ID_CURR", how="left")
    .merge(avg_creditCardBal, on="SK_ID_CURR", how="left")
    .merge(avg_posCashBal, on="SK_ID_CURR", how="left")
    .merge(avg_prev, on="SK_ID_CURR", how="left")
)

# cleaning up memory
del (
    bureau,
    creditCardBal,
    posCashBal,
    previousApp,
    avgPreviousApp,
    avgBureau,
    avgCreditCardBal,
    avgPosCashBal,
)
gc.collect()

print("Preparing data...\n")
teLabeled.drop(labels=["SK_ID_CURR"], axis=1, inplace=True)

print("Scoring ...\n")

# loading model from pickeled file
with open("../pickle/risk_model.pickle", "rb") as handle:
    mGmmPickle = pickle.load(handle)

oofPredsPickleTest = np.zeros(te_labeled.shape[0])

# scoring data
oofPredsPickleTest = mGmmPickle.predict(te_labeled)


print("Output data")
te["TARGET_oof"] = oofPredsPickleTest.copy()
te.to_csv("../output/application_test_with_output.csv", index=False)
