
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc

from scipy.stats import chi2_contingency

from xgboost import XGBClassifier

pd.set_option('display.max_columns', None)

scaler = MinMaxScaler()
le = LabelEncoder()
random_state = 42
opt_iter = 200

train_df = pd.read_csv("train.csv")
train_df.drop("id", axis=1, inplace=True)

test_df = pd.read_csv("test.csv")
test_id = test_df["id"]
test_df.drop("id", axis=1, inplace=True)

print(train_df.info())
print("=" * 50)
print(train_df.describe())
print("=" * 50)
print(train_df.isnull().sum())
print("=" * 50)

for col in train_df.columns:
    print(train_df[col].value_counts())
    print("=" * 50)

num_cols = train_df.select_dtypes(exclude="object").columns.tolist()
cat_cols = train_df.select_dtypes("object").columns.tolist()

plt.figure(figsize=(20, 16))

for i, col in enumerate(num_cols):
    plt.subplot((len(num_cols) + 1)// 2, 2, i + 1)
    plt.title(f"Distribution of {col}")
    sns.histplot(data=train_df, x=col, color="red", kde=True)
    sns.despine()

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 16))

sns.pairplot(train_df[num_cols], hue="loan_status")
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 16))

for i, col in enumerate(cat_cols):
    contingency_table = pd.crosstab(train_df[col], train_df["loan_status"])
    plt.subplot((len(cat_cols) + 1)// 2, 2, i + 1)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    plt.title(f"Contingency Table Heatmap for {col}")
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", cbar=True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 16))

for i, col in enumerate(cat_cols):
    plt.subplot((len(cat_cols) + 1)// 2, 2, i + 1)
    plt.title(f"Distribution of {col}")
    sns.histplot(data=train_df, x=col, color="blue")
    sns.despine()

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 16))

for i, col in enumerate(cat_cols):
    plot_data = train_df.groupby([col, "loan_status"]).size().unstack()
    plot_data = plot_data.div(plot_data.sum(axis=1), axis=0)
    plt.subplot((len(cat_cols) + 1)// 2, 2, i + 1)
    plot_data.plot(kind="bar", stacked=True, color=["red", "blue"], ax=plt.gca())

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 16))

train_df = pd.get_dummies(train_df, columns=cat_cols, prefix_sep="_", drop_first=True, dtype="int")
test_df = pd.get_dummies(test_df, columns=cat_cols, prefix_sep="_", drop_first=True, dtype="int")

corr_matrix = train_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.tight_layout()
plt.show()

num_cols.remove("loan_status")

def make_features(df):
    df["loan_to_income"] = df["loan_amnt"] / df["person_income"]
    df["income_to_emp"] = df["person_income"] / (df["person_emp_length"] + 1)
    df["interest_amnt"] = df["loan_amnt"] * (df["loan_int_rate"] / 100)
    df["credit_to_age"] = df["person_age"] / (df["cb_person_cred_hist_length"] + 1)
    df["total_borrowing"] = df["loan_amnt"] + df["cb_person_cred_hist_lengtht"]
    df['loan_int_emp_interaction'] = df['loan_int_rate'] * df['person_emp_length']
    df['debt_to_credit_ratio'] = df['loan_amnt'] / (df['cb_person_cred_hist_length'] + 1)
    df['int_to_cred_hist'] = df['loan_int_rate'] / (df['cb_person_cred_hist_length'] + 1)
    df['int_per_year_emp'] = df['loan_int_rate'] / (df['person_emp_length'] + 1)
    df['loan_amt_per_emp_year'] = df['loan_amnt'] / (df['person_emp_length'] + 1)      
    df['income_to_loan_ratio'] = df['person_income'] / (df['loan_amnt'] + 1)

    grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    train_df["loan_grade_num"] = train_df["loan_grade"].map(grade_mapping)
    train_df["weighted_loan_percent_income"] = train_df["loan_percent_income"] * train_df["loan_grade_num"]

    return df

train_df = make_features(train_df)
test_df = make_features(test_df)

train, eval = train_test_split(train_df, train_size=0.8)

X_train = train.drop("loan_status", axis=1)
y_train = train["loan_status"]

X_eval = eval.drop("loan_status", axis=1)
y_eval = eval["loan_status"]

X_train = scaler.fit_transform(X_train)
X_eval = scaler.transform(X_eval)
test_df = scaler.transform(test_df)

y_train = le.fit_transform(y_train)
y_eval = le.fit_transform(y_eval)


def objective_XGB(trial):
    param = {
        'verbosity': 0,
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 9)
    }

    model = XGBClassifier(**param,
                          use_label_encoder=True,
                          eval_metric='mlogloss',
                          device='cuda',
                          predictors='gpu_predictor',
                          random_state=random_state)

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_eval = model.predict(X_eval)

    auc_tuning_train = roc_auc_score(y_train, y_pred_train)
    auc_tuning_eval = roc_auc_score(y_eval, y_pred_eval)

    print("=" * 65)
    print("%15s %15.5f | %15s %15.5f" % ("Train accuracy:",
                                         auc_tuning_train,
                                         "Eval accuracy:",
                                         auc_tuning_eval))
    print("=" * 65)

    return auc_tuning_eval


study = optuna.create_study(direction="maximize")
study.optimize(objective_XGB, n_trials=opt_iter)

model = XGBClassifier(**study.best_params, random_state=random_state)

model.fit(X_train, y_train)

y_pred = model.predict(X_eval)
y_test_pred = model.predict(test_df)

roc_auc = roc_auc_score(y_eval, y_pred)

print("=" * 31)
print("%15s %15.5f" % ("Accuracy score:", accuracy_score(y_pred=y_pred, y_true=y_eval)))
print("=" * 31)
print("%10s %10.5f" % ("AUC:", roc_auc))
print("=" * 31)

fpr, tpr, thresholds = roc_curve(y_eval, y_pred)
model_auc = auc(fpr, tpr)

plt.figure(figsize=(20, 16))

plt.plot(fpr, tpr, color="darkorange", label=f"ROC curve (area = {model_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

submission = pd.DataFrame({"id": test_id,
                           "loan_status": y_test_pred})

submission.to_csv(f"submission_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv", index=False)

