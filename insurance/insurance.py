import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import optuna

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from datetime import datetime

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

random_state = 42
opt_iter = 100

target = "Premium Amount"
train_df = pd.read_csv("train.csv")

test_df = pd.read_csv("test.csv")

train_df.drop(columns=["id"], inplace=True)

test_df, idx = test_df.drop("id", axis=1), test_df["id"]

print(train_df.info())
print("=" * 50)
print(train_df.describe())
print("=" * 50)
print(train_df.isna().sum())
print("=" * 50)

cat_cols = train_df.select_dtypes(include="object").columns.tolist()
num_cols = train_df.select_dtypes(exclude="object").columns.tolist()

num_cols.remove(target)
cat_cols.remove("Policy Start Date")

for col in cat_cols:
    print(train_df[col].value_counts())
    print("=" * 50)

for col in num_cols:
    print(train_df[col].value_counts())
    print("=" * 50)

for col in train_df.columns:
    print(f"Number of unique values in {col}")
    print(train_df[col].nunique())
    print("=" * 50)

# fig, axes = plt.subplots(
#     nrows=len(num_cols),
#     ncols=3,
#     figsize=(30, 10 * len(num_cols))
# )

# palette = sns.color_palette("tab10", len(num_cols))
# colors = dict(zip(num_cols, palette))

# df_binned = train_df.copy()

# # Histogram of numerical data
# for i, col in enumerate(num_cols):
#     if train_df[col].nunique() > 50:
#         discrete = False
#     else:
#         discrete = True

#     sns.histplot(
#         data=train_df,
#         x=col,
#         color=colors[col],
#         kde=True,
#         ax=axes[i, 0],
#         discrete=discrete
#     )
#     axes[i, 0].set_title(f"Distirbution of {col}")
#     sns.despine(ax=axes[i, 0])

#     if train_df[col].nunique() <= 10:
#         sns.violinplot(
#             data=train_df,
#             x=col,
#             y=target,
#             color=colors[col],
#             ax=axes[i, 1],
#         )
#         axes[i, 1].set_title(f"Violin plot of {col}")
#         sns.despine(ax=axes[i, 1])
#     else:
#         df_binned[f"{col}_binned"] = pd.cut(train_df[col], bins=10)
#         sns.violinplot(
#             data=df_binned,
#             x=f"{col}_binned",
#             y=target,
#             color=colors[col],
#             ax=axes[i, 1],
#         )
#         axes[i, 1].set_title(f"Violin plot of {col}")
#         sns.despine(ax=axes[i, 1])

#     sns.boxplot(
#         data=train_df,
#         x=col,
#         color=colors[col],
#         ax=axes[i, 2]
#     )
#     axes[i, 2].set_title(f"Box plot of {col}")
#     sns.despine(ax=axes[i, 2])

# plt.tight_layout()
# plt.savefig("num_values.jpg", dpi=300)
# plt.show()

# fig, axes = plt.subplots(
#     nrows=len(cat_cols),
#     ncols=2,
#     figsize=(30, 10 * len(cat_cols))
# )

# for i, col in enumerate(cat_cols):
#     sns.countplot(
#         data=train_df,
#         x=col,
#         hue=col,
#         palette="tab10",
#         ax=axes[i, 0]
#     )
#     axes[i, 0].set_title(f"Count of {col}")
#     sns.despine(ax=axes[i, 0])

#     sns.boxplot(
#         data=train_df,
#         x=col,
#         y=target,
#         palette="tab10",
#         ax=axes[i, 1]
#     )
#     axes[i, 1].set_title(f"Box plot of {col}")
#     sns.despine(ax=axes[i, 1])

# plt.tight_layout()
# plt.savefig("cat_values.jpg", dpi=300)
# plt.show()

# plt.figure(figsize=(30, 15))

# corr = train_df[num_cols].corr()
# sns.heatmap(
#     data=corr,
#     cmap="coolwarm",
#     fmt=".2f",
#     linewidths=0.7,
#     annot=True
# )

# plt.savefig("corr_map.jpg", dpi=300)
# plt.show()


def feature_engineering(df) -> pd.DataFrame:

    df["Age Binnded"] = pd.cut(df["Age"], bins=10)

    # 'Annual Income' is positively skewed thus
    # I'm applying square root transformation
    df["Annual Income"] = np.sqrt(df["Annual Income"])

    oe = OrdinalEncoder(categories=[
        ["High School", "Bachelor's", "Master's", "PhD"]
    ])

    # Using ordinal encoding on the 'Education Level' data
    df["Education Level"] = oe.fit_transform(df[["Education Level"]])

    # df["Income/Credit"] = df["Annual Income"] / df["Credit Score"]
    # df["VAge/Duration"] = df["Vehicle Age"] / df["Insurance Duration"]
    # df["Education-Income"] = df["Education Level"] * df["Annual Income"]

    df["Policy Start Date"] = pd.to_datetime(
        df["Policy Start Date"],
        format="ISO8601"
    )

    upper_bound = df["Annual Income"].mean() + 2 * df["Annual Income"].std()
    df["Annual Income"] = df["Annual Income"].clip(upper=upper_bound)

    upper_bound = df["Previous Claims"].mean() + 2 * df["Previous Claims"].std()
    df["Previous Claims"] = df["Previous Claims"].clip(upper=upper_bound)

    df["year"] = df["Policy Start Date"].dt.year
    df["month"] = df["Policy Start Date"].dt.month
    df["day"] = df["Policy Start Date"].dt.day
    df["day_of_the_week"] = df["Policy Start Date"].dt.dayofweek

    df.drop("Policy Start Date", axis=1, inplace=True)

    return df


train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

train, eval = train_test_split(train_df,
                               train_size=0.8,
                               random_state=random_state)

num_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]
)

cat_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ]
)

X_train, y_train = train.drop(target, axis=1), train[target]
X_eval, y_eval = eval.drop(target, axis=1), eval[target]

X_train = preprocessor.fit_transform(X_train)
X_eval = preprocessor.transform(X_eval)

test = preprocessor.transform(test_df)

y_train = np.sqrt(y_train)


def objective_XGB(trial) -> float:
    params = {
        'objective': 'reg:squarederror',
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.0, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.0, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 9),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'verbosity': 0,
        'device': 'cuda',
        'n_jobs': -1
    }

    model = XGBRegressor(**params, random_state=random_state)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)
    y_pred = y_pred**2

    rmsle = root_mean_squared_log_error(y_true=y_eval, y_pred=y_pred)

    print("=" * 14)
    print("RMSLE: %.5f" % (rmsle))
    print("=" * 14)

    return rmsle


study_XGB = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner()
)
study_XGB.optimize(objective_XGB, n_trials=opt_iter)

model_XGB = XGBRegressor(**study_XGB.best_params, random_state=random_state)

model_XGB.fit(X_train, y_train)


def objective_LGB(trial) -> float:
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'verbosity': -1,
        'device': 'gpu'
    }

    model = LGBMRegressor(**params, random_state=random_state)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)
    y_pred = y_pred**2

    rmsle = root_mean_squared_log_error(y_true=y_eval, y_pred=y_pred)

    print("=" * 14)
    print("RMSLE: %.5f" % (rmsle))
    print("=" * 14)

    return rmsle


study_LGB = optuna.create_study(direction='minimize')
study_LGB.optimize(objective_LGB, n_trials=opt_iter)

model_LGB = LGBMRegressor(**study_LGB.best_params, random_state=random_state)

model_LGB.fit(X_train, y_train)

pred_XGB = model_XGB.predict(test)**2
pred_LGB = model_LGB.predict(test)**2

submission_XGB = pd.DataFrame({
    "id": idx,
    target: pred_XGB
})

submission_LGB = pd.DataFrame({
    "id": idx,
    target: pred_LGB
})

submission_XGB.to_csv(
    f"submission_{datetime.now().strftime("%Y%m%d_%H%M%S")}_XGB.csv",
    index=False
)

submission_LGB.to_csv(
    f"submission_{datetime.now().strftime("%Y%m%d_%H%M%S")}_LGB.csv",
    index=False
)

