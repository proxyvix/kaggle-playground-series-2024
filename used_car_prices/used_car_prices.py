import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import optuna
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

from xgboost import XGBRegressor

random_state = 42
opt_iter = 200
current_date = datetime.datetime.today()
scaler = StandardScaler()

train_df = pd.read_csv('train.csv')
train_df = train_df.drop('id', axis=1)

test_df = pd.read_csv('test.csv')
test_id = test_df['id']
test_df = test_df.drop('id', axis=1)

print(f'Columns in the train data set:\n{train_df.columns}')
print('-----------------------------')
print(f'Columns in the test data set:\n{test_df.columns}')
print('=============================')
print(f'Info on the train data set:\n{train_df.info()}')
print('-----------------------------')
print(f'Info on the test data set:\n{test_df.info()}')
print('=============================')
print(f'Description of the train data set:\n{train_df.describe()}')
print('-----------------------------')
print(f'Description of the test data set:\n{test_df.describe()}')
print('=============================')
print(f'NA values in the train data set:\n{train_df.isna().sum()}')
print('-----------------------------')
print(f'NA values in the test data set:\n{test_df.isna().sum()}')
print('=============================')
print(f'Missing values in the train data set:\n{train_df.isnull().sum()}')
print('-----------------------------')
print(f'Missing values in the test data set:\n{test_df.isnull().sum()}')
print('=============================')


for col in train_df.columns:
    print(train_df[col].value_counts())
    print('=============================')


def extract_transmission_type(trans_type):
    if re.search(r'A/T|AUTOMATIC|CVT', trans_type):
        return 'Automatic'
    elif re.search(r'M/T|MANUAL', trans_type):
        return 'Manual'
    else:
        return 'Other'


def extract_speed(trans_type):
    speed_match = re.search(r'(\d+)[\s-]*SPEED', trans_type)
    if speed_match:
        return int(speed_match.group(1))
    elif re.search('SINGLE', trans_type):
        return 1
    else:
        return 0


def transform(df):
    df['transmission'] = df['transmission'].str.upper()

    df['Transmission_type'] = df['transmission'].apply(extract_transmission_type)

    df['Cylinders'] = df['engine'].str.extract(r'(\d+) Cylinder')[0]
    df['Cylinders'] = pd.to_numeric(df['Cylinders'], errors='coerce')

    df['Horse_power'] = df['engine'].str.extract(r'(\d+\.\d+)HP')
    df['Horse_power'] = pd.to_numeric(df['Horse_power'], errors='coerce')

    df['Displacement'] = df['engine'].str.extract(r'(\d+\.\d+)L')
    df['Displacement'] = pd.to_numeric(df['Displacement'], errors='coerce')

    df['Speed'] = df['transmission'].apply(extract_speed)

    return df


train_df = transform(train_df)
test_df = transform(test_df)

categorical_data = list(train_df.select_dtypes('object'))
numerical_data = list(train_df.select_dtypes(exclude='object'))

plt.figure(figsize=(8, 8))

for i, num_col in enumerate(numerical_data):
    plt.subplot((len(numerical_data) + 1) // 2, 2, i+1)
    plt.title(f'Distribution of {num_col}')
    sns.histplot(data=train_df, x=num_col, color='red', kde=True)
    sns.despine()

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
corr_matrix = train_df[numerical_data].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.tight_layout()
plt.show()

na_values = train_df.isna().sum()
na_values = na_values[na_values > 0]

plt.figure(figsize=(8, 8))
sns.barplot(x=na_values.index, y=na_values.values, color='red')
plt.xlabel('NA values')
plt.tight_layout()
plt.show()

missing_values = train_df.isnull().sum()
missing_values = missing_values[missing_values > 0]

plt.figure(figsize=(8, 8))
sns.barplot(x=missing_values.index, y=missing_values.values, color='red')
plt.xlabel('Missing values')
plt.tight_layout()
plt.show()

train, eval = train_test_split(train_df,
                               train_size=0.8,
                               random_state=random_state)

X_train, y_train = train.drop('price', axis=1), train['price']
X_eval, y_eval = eval.drop('price', axis=1), eval['price']

numerical_data.remove('price')

X_train = scaler.fit_transform(X_train[numerical_data])
X_eval = scaler.fit_transform(X_eval[numerical_data])
test_df = scaler.fit_transform(test_df[numerical_data])

def objective_XGB(trial):
    params = {
        'objective': 'reg:squarederror',
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
        'max_depth': trial.suggest_int('max_depth', 1, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.0, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.0, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 9),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'eval_metric': 'rmse', 
        'verbosity': 0,
        'device': 'cuda',
        'n_jobs': -1
    }

    model = XGBRegressor(**params, random_state=random_state)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)

    rmse = root_mean_squared_error(y_true=y_eval, y_pred=y_pred)

    print('=============================================')
    print(f'Cross-Validation Score: {rmse:.5f}')
    print('=============================================')

    return rmse

study_XGB = optuna.create_study(direction='minimize')
study_XGB.optimize(objective_XGB, n_trials=opt_iter)

model_XGB = XGBRegressor(**study_XGB.best_params, random_state=random_state)

model_XGB.fit(X_train, y_train)

prediction = model_XGB.predict(test_df)

submission = pd.DataFrame({'id': test_id,
                           'price': prediction})

submission.to_csv(f'submission_{current_date}.csv', index=False)

