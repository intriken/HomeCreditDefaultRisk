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

# creating new features based off the aggregate values of columns
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

# clean up variables
del (labels1, labels2, labels3, labels4)

# load test and train data
tr = pd.read_csv("../input/application_train.csv")
te = pd.read_csv("../input/application_test.csv")

tri = tr.shape[0]
y = tr.TARGET.copy()

# cleaning up data for input into model
tr_te = (
    tr.drop(labels=["TARGET"], axis=1)
    .append(te)
    .pipe(LabelEncodingCategory)
    .pipe(FillNA)
    .merge(avgBureau, on="SK_ID_CURR", how="left")
    .merge(avgCreditCardBal, on="SK_ID_CURR", how="left")
    .merge(avgPosCashBal, on="SK_ID_CURR", how="left")
    .merge(avgPreviousApp, on="SK_ID_CURR", how="left")
)

# clean up memory
del (
    tr,
    te,
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
tr_te.drop(labels=["SK_ID_CURR"], axis=1, inplace=True)
tr = tr_te.iloc[:tri, :].copy()
te = tr_te.iloc[tri:, :].copy()

del tr_te

# lightgbm parameter setup
Dparam = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "auc",
    "nthread": 4,
    "shrinkage_rate": 0.025,
    "max_depth": 8,
    "min_data_in_leaf": 100,
    "min_child_weight": 2,
    "bagging_fraction": 0.75,
    "feature_fraction": 0.75,
    "min_split_gain": 0.01,
    "lambda_l1": 1,
    "lambda_l2": 1,
    "num_leaves": 36,
}

print("Training model...\n")

# defining iterations
folds = KFold(n_splits=5, shuffle=True, random_state=123456)

oofPreds = np.zeros(tr.shape[0])
subPreds = np.zeros(te.shape[0])
featureImportance_df = pd.DataFrame()
feats = [f for f in tr.columns if f not in ["SK_ID_CURR"]]

# training and saving model for each iteration
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(tr)):
    dtrain = gbm.Dataset(tr.iloc[trn_idx], y.iloc[trn_idx])
    dval = gbm.Dataset(tr.iloc[val_idx], y.iloc[val_idx])
    m_gbm = gbm.train(
        params=Dparam,
        train_set=dtrain,
        num_boost_round=3000,
        verbose_eval=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
    )
    oofPreds[val_idx] = m_gbm.predict(tr.iloc[val_idx])
    subPreds += m_gbm.predict(te) / folds.n_splits
    foldImportance_df = pd.DataFrame()
    foldImportance_df["feature"] = feats
    foldImportance_df["importance"] = m_gbm.feature_importance()
    foldImportance_df["fold"] = n_fold + 1
    featureImportance_df = pd.concat([featureImportance_df, foldImportance_df], axis=0)
    print(
        "Fold %2d AUC : %.6f"
        % (n_fold + 1, roc_auc_score(y.iloc[val_idx], oofPreds[val_idx]))
    )
    del dtrain, dval
    gc.collect()

print("Full AUC score %.6f" % roc_auc_score(y, oofPreds))


# dumping out trained model to pickle file for version control
pickle.dump(m_gbm, open("../pickle/risk_model.pickle", "wb"))


m_gbm.save_model("../model/gbm_classifier.txt")


def display_importances(featureImportance_df_):
    # Plot feature importances
    cols = (
        featureImportance_df_[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:50]
        .index
    )
    best_features = featureImportance_df_.loc[featureImportance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False),
    )
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.savefig("lgbm_importances.png")


display_importances(featureImportance_df)

print("Output Model")
tr_oof = pd.read_csv("../input/application_train.csv", usecols=["SK_ID_CURR", "TARGET"])
tr_oof["TARGET_oof"] = oofPreds.copy()
tr_oof.to_csv("Target_Simple_2_Model_GBM_oof.csv", index=False)


Submission = pd.read_csv("../input/sample_submission.csv")
Submission["TARGET"] = subPreds.copy()
Submission.to_csv("Lightgbm_Simple_2_Model.csv", index=False)
