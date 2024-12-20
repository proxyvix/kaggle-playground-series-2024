import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

split = .8
random_state = 42

train_df = pd.read_csv('train.csv')
train_df = train_df.drop('id', axis=1)

# Data visualisation

print(train_df.info())
print(train_df.describe())
print(train_df.isnull().sum())
print(train_df.duplicated().sum())

# Data analysis
"""
response_counts = train_df['Response'].value_counts()
response_values = train_df['Response'].unique().tolist()

plt.figure(figsize=(5, 5))
plt.pie(response_counts, labels=response_values, autopct='%1.1f%%')
plt.title('Response distribution')
plt.show()

plt.figure(figsize=(6, 12))
plotnumber = 1
"""
dist_data = [
    'Age',
    'Annual_Premium',
    'Policy_Sales_Channel',
    'Vintage',
]
"""
for col in dist_data:
    if plotnumber <= len(train_df.columns):
        ax = plt.subplot(len(dist_data)//3 + 1, 3, plotnumber)
        sb.histplot(train_df[col], bins=20, kde=True, ax=ax)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plotnumber += 1

plt.tight_layout()
plt.show()
"""
count_data = [
    'Gender',
    'Previously_Insured',
    'Vehicle_Damage',
]
"""
plotnumber = 1

for col in count_data:
    if plotnumber <= len(train_df.columns):
        ax = plt.subplot(len(count_data)//3 + 1, 3, plotnumber)
        ax.hist(train_df[col])
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plotnumber += 1

plt.tight_layout()
plt.show()

"""
train_df = pd.get_dummies(train_df,
                          columns=['Vehicle_Age', 'Vehicle_Damage', 'Gender'],
                          dtype='int')

"""
corr_matrix = train_df.corr()

sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
"""
# Feature engineering

train_df.columns = train_df.columns.str.replace('<', '_')
train_df.columns = train_df.columns.str.replace('>', '_')

scaler = StandardScaler()

train_df[dist_data] = scaler.fit_transform(train_df[dist_data])

# Training a base model

train, eval = train_test_split(train_df, train_size=split)

X_train, y_train = train.drop('Response', axis=1), train['Response']
X_eval, y_eval = eval.drop('Response', axis=1), eval['Response']

model_XGB = XGBClassifier(random_state=random_state, verbosity=0, device='cuda', predictors='gpu_predictor')
model_CBC = CatBoostClassifier(random_state=random_state, verbose=0)

model_XGB.fit(X_train, y_train)
model_CBC.fit(X_train, y_train)

y_train_pred_XGB = model_XGB.predict(X_train)
y_eval_pred_XGB = model_XGB.predict(X_eval)

y_train_pred_CBC = model_CBC.predict(X_train)
y_eval_pred_CBC = model_CBC.predict(X_eval)

train_acc_XGB = accuracy_score(y_true=y_train, y_pred=y_train_pred_XGB)
eval_acc_XGB = accuracy_score(y_true=y_eval, y_pred=y_eval_pred_XGB)

train_acc_CBC = accuracy_score(y_true=y_train, y_pred=y_train_pred_CBC)
eval_acc_CBC = accuracy_score(y_true=y_eval, y_pred=y_eval_pred_CBC)

print('==================================================')
print(f'XGB - Train accuracy: {train_acc_XGB:.5f}, Eval accuracy: {eval_acc_XGB:.5f}')
print('==================================================')
print(f'CBC - Train accuracy: {train_acc_CBC:.5f}, Eval accuracy: {eval_acc_CBC:.5f}')
print('==================================================')

# Fine tuning


def objective_XGB(trial):

    param = {
        'verbosity': 0,
        'num_class': 2,
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.0, 1.0),
        'colsample_bytree': trial.suggest_float('subsample', 0.0, 1.0),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 9)
    }

    model = XGBClassifier(**param,
                          use_label_encoder=False,
                          eval_metric='mlogloss',
                          device='cuda',
                          predictors='gpu_predictor',
                          random_state=random_state)

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_eval = model.predict(X_eval)

    acc_tuning_train = accuracy_score(y_true=y_train, y_pred=y_pred_train)
    acc_tuning_eval = accuracy_score(y_true=y_eval, y_pred=y_pred_eval)

    print('=============================================')
    print(f'Accuracy Train: {acc_tuning_train:.5f}, Eval: {acc_tuning_eval:.5f}')
    print('=============================================')

    return acc_tuning_eval


study_XGB = optuna.create_study(direction='maximize')
study_XGB.optimize(objective_XGB, n_trials=50)

print('=============================================')
print(f'XGB - best accuracy: {study_XGB.best_value:.5f}')
