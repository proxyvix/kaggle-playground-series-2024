import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import datetime

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier


random_state = 42
opt_iter = 100
current_date = datetime.date.today()
le = LabelEncoder()

# -------------------------------------------------
# Importing the data
# -------------------------------------------------

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df = train_df.drop('id', axis=1)

print(train_df.info())
print(train_df.describe())
print(train_df.isna().sum())
print('===============================')


# Making a function which replaces data with "NA"
# if their count is below a certain treshold

def replace_missing(df, col, treshold=0.05):
    counts = df[col].value_counts()
    low_count_data = counts[counts / df.shape[0] <= treshold].index
    df[col] = df[col].apply(lambda x: 'NA' if x in low_count_data else x)
    return df


# -------------------------------------------------
# Understanding the data
# -------------------------------------------------

print('Before data cleaning:')
for col in train_df.columns:
    print(f'{col} has {train_df[col].nunique()} unique values\n')
    print(f'Values in {col}: \n{train_df[col].value_counts()}')
    print('-------------------------------')

print('===============================')

# -------------------------------------------------
# Data Cleaning
# -------------------------------------------------

caterogical_data = list(train_df.select_dtypes(include=object))
numerical_data = list(train_df.select_dtypes(exclude=object))

caterogical_data.remove('class')

for cat_col in caterogical_data:
    train_df = replace_missing(train_df, cat_col)
    test_df = replace_missing(test_df, cat_col)

train_df = train_df.drop_duplicates()
test_df = test_df.drop_duplicates()

print(train_df[numerical_data].skew())

# Since the skewness of the numerical data is higher than 1,
# the missing data can be replaced with the median of the values

train_df[numerical_data] = train_df[numerical_data].fillna(
    train_df[numerical_data].median()
)

test_df[numerical_data] = test_df[numerical_data].fillna(
    train_df[numerical_data].median()
)

train_df[caterogical_data] = train_df[caterogical_data].fillna('NA')
test_df[caterogical_data] = test_df[caterogical_data].fillna('NA')

print('After data cleaning:')
for col in train_df.columns:
    print(f'{col} has {train_df[col].nunique()} unique values\n')
    print(f'Values in {col}: \n{train_df[col].value_counts()}')
    print('-------------------------------')

print('===============================')

# -------------------------------------------------
# Exploratory Data Analysis
# -------------------------------------------------

plt.figure(figsize=(8, 10))

for i, num_col in enumerate(numerical_data):
    plt.subplot(len(numerical_data), 1, i+1)
    sns.histplot(data=train_df, x=num_col, palette='rocket')
    plt.title(f'{num_col} distribution')
    sns.despine()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))

for i, cat_col in enumerate(caterogical_data):
    plt.subplot((len(caterogical_data) + 1) // 3, 3, i+1)
    plt.title(f'{cat_col} distribution')
    sns.countplot(data=train_df, x=cat_col, palette='rocket')
    sns.despine()

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 10))

for i, num_col in enumerate(numerical_data):
    plt.subplot(len(numerical_data), 1, i+1)
    plt.title(f'Distribution of {num_col} by Class')
    sns.violinplot(data=train_df, x='class', y=num_col, palette='rocket')
    sns.despine()

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
plt.pie(train_df['class'].value_counts(),
        labels=train_df['class'].unique(),
        autopct='%1.1f%%')
plt.title('Pie chart of Classes')
plt.show()

# -------------------------------------------------
# Training on 3 different models
# -------------------------------------------------

train_df[caterogical_data] = train_df[caterogical_data].astype('category')
test_df[caterogical_data] = test_df[caterogical_data].astype('category')

categorical_pipeline = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(dtype=np.int32,
                               handle_unknown='use_encoded_value',
                               unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_pipeline, caterogical_data)
    ]
)

train, eval = train_test_split(train_df,
                               train_size=0.8,
                               random_state=random_state)

y_train = train['class']
y_eval = eval['class']

X_train = train.drop('class', axis=1)
X_eval = eval.drop('class', axis=1)

y_train = le.fit_transform(y_train)
y_eval = le.fit_transform(y_eval)

test_id = test_df['id']

X_train = preprocessor.fit_transform(X_train)
X_eval = preprocessor.transform(X_eval)
test = preprocessor.transform(test_df.drop('id', axis=1))


def objective_XGB(trial):
    params = {
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.0, 1.0),
        'colsample_bytree': trial.suggest_float('subsample', 0.0, 1.0),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 9),
        'device': 'cuda',
        'eval_metric': 'mlogloss',
        'verbosity': 0,
        'enable_categorical': True
    }

    model = XGBClassifier(**params, random_state=random_state)

    model.fit(X_train, y_train)

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    cv_score = cross_val_score(model,
                               X_train,
                               y_train,
                               cv=kf).mean()

    print('=============================================')
    print(f'Cross-Validation Score: {cv_score:.5f}')
    print('=============================================')

    return cv_score


study_XGB = optuna.create_study(direction='maximize')
study_XGB.optimize(objective_XGB, n_trials=opt_iter)

best_params_XGB = study_XGB.best_params

model_XGB = XGBClassifier(**best_params_XGB, random_state=42)

model_XGB.fit(X_train, y_train)

y_train_XGB = model_XGB.predict(X_train)
y_eval_XGB = model_XGB.predict(X_eval)

train_acc_XGB = accuracy_score(y_true=y_train, y_pred=y_train_XGB)
eval_acc_XGB = accuracy_score(y_true=y_eval, y_pred=y_eval_XGB)

print('===============================')
print(f'Train accuracy XGB: {train_acc_XGB:.5f}')
print('-------------------------------')
print(f'Eval accuracy XGB: {eval_acc_XGB:.5f}')
print('===============================')

test_pred = model_XGB.predict(test)
test_pred = le.inverse_transform(test_pred)

submission = pd.DataFrame({'id': test_id,
                           'class': test_pred})

submission.to_csv(f'results/submission_{current_date}_v3.csv', index=False)

