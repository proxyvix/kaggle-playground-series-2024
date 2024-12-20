import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
import seaborn as sb
import optuna

train = pd.read_csv('playground-series-s4e5/train.csv')

train_eval_split = .8

train, eval = train_test_split(train,
                               train_size=train_eval_split,
                               random_state=42)


def transform_df(df):

    columns = list(df.columns)
    columns = [columns[i] for i in range(
        len(columns)) if columns[i] != 'id' and columns[i] != 'FloodProbability']

    df['sum'] = df[columns].sum(axis=1)
    df['mean'] = df[columns].mean(axis=1)
    df['median'] = df[columns].median(axis=1)
    df['std'] = df[columns].std(axis=1)
    df['skew'] = df[columns].skew(axis=1)
    df['mode'] = df[columns].mode(axis=1)[0]
    df['cv'] = df['std']/df['mean']
    df['max'] = df[columns].max(axis=1)
    df['min'] = df[columns].min(axis=1)
    df['25percent'] = df[columns].quantile(.25, axis=1)
    df['75percent'] = df[columns].quantile(.75, axis=1)

    df['harmonic'] = len(columns) / \
        df[columns].apply(lambda x: (1/x).mean(), axis=1)

    df['aggr1'] = df['MonsoonIntensity'] + df['ClimateChange'] + \
        df['WetlandLoss'] + df['Deforestation']

    df['aggr2'] = df['TopographyDrainage'] + \
        df['RiverManagement'] + df['DrainageSystems']

    df['aggr'] = np.log((df['aggr1'] * df['aggr2']) + 1)

    df = df.drop(['aggr1', 'aggr2'], axis=1)

    return df


train = transform_df(df=train)
eval = transform_df(df=eval)
print(train)

corr_matrix = train.drop(['id'], axis=1).corr()

plt.figure(figsize=(20, 20))
sb.heatmap(corr_matrix, linewidths=.5, cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Correlation Matrix of Numerical Features', fontsize=15)
plt.xticks(fontsize=8, fontweight='bold')
plt.yticks(fontsize=8, fontweight='bold')
plt.show()

scaler = StandardScaler()

features = list(train.columns)
features = [features[i] for i in range(
    len(features)) if features[i] != 'id' and features[i] != 'FloodProbability']

train[features] = scaler.fit_transform(train[features])
eval[features] = scaler.fit_transform(eval[features])

X_train, y_train = train[features], train['FloodProbability']
X_eval, y_eval = eval[features], eval['FloodProbability']

model = lgbm.LGBMRegressor(boosting_type='gbdt',
                           objective='regression',
                           num_leaves=30,
                           learning_rate=.05,
                           n_estimators=100)

model.fit(X_train,
          y_train,
          eval_metric='l2')

y_pred_train = model.predict(X_train, num_iteration=model.best_iteration_)
y_pred_eval = model.predict(X_eval, num_iteration=model.best_iteration_)

mse_train = mean_squared_error(y_pred=y_pred_train, y_true=y_train)
mse_eval = mean_squared_error(y_pred=y_pred_eval, y_true=y_eval)

r2_train = r2_score(y_pred=y_pred_train, y_true=y_train)
r2_eval = r2_score(y_pred=y_pred_eval, y_true=y_eval)

print(f'MSE Train: {mse_train:.5f}, MSE Eval: {mse_eval:.5f}')
print('---------------------------------------------')
print(f'R2 Train: {r2_train:.5f}, R2 Eval: {r2_eval:.5f}')
print('=============================================')


def objective(trial):

    model = lgbm.LGBMRegressor(objective='regression',
                               metric='rmse',
                               verbosity=-1,
                               boosting_type='gbdt',
                               lambda_l1=trial.suggest_float(
                                   'lambda_l1', 1e-8, 10.0, log=True),
                               lambda_l2=trial.suggest_float(
                                   'lambda_l2', 1e-8, 10.0, log=True),
                               num_leaves=trial.suggest_int(
                                   'num_leaves', 2, 256),
                               feature_fraction=trial.suggest_float(
                                   'feature_fraction', 0.4, 1.0),
                               bagging_fraction=trial.suggest_float(
                                   'bagging_fraction', 0.4, 1.0),
                               bagging_freq=trial.suggest_int(
                                   'bagging_freq', 1, 7),
                               min_child_samples=trial.suggest_int(
                                   'min_child_samples', 5, 100),
                               learning_rate=trial.suggest_float(
                                   'learning_rate', 1e-4, 1e-1, log=True),
                               random_state=42)

    model.fit(X_train,
              y_train,
              eval_set=[(X_eval, y_eval)])

    y_pred_train = model.predict(X_train, num_iteration=model.best_iteration_)
    y_pred_eval = model.predict(X_eval, num_iteration=model.best_iteration_)

    mse_train = mean_squared_error(y_pred=y_pred_train, y_true=y_train)
    mse_eval = mean_squared_error(y_pred=y_pred_eval, y_true=y_eval)

    r2_train = r2_score(y_pred=y_pred_train, y_true=y_train)
    r2_eval = r2_score(y_pred=y_pred_eval, y_true=y_eval)

    print(f'MSE Train: {mse_train:.5f}, MSE Eval: {mse_eval:.5f}')
    print('---------------------------------------------')
    print(f'R2 Train: {r2_train:.3f}, R2 Eval: {r2_eval:.3f}')
    print('=============================================')

    return mse_eval


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f'Best parameters: {study.best_params}')
print(f'Best MSE: {study.best_value}')

best_params = study.best_params

final_model = model = lgbm.LGBMRegressor(objective='regression',
                                         metric='rmse',
                                         verbosity=-1,
                                         boosting_type='gbdt',
                                         random_state=42,
                                         **best_params)

final_model.fit(X_train,
                y_train,
                eval_set=[(X_eval, y_eval)])

y_preds_train = final_model.predict(
    X_train, num_iteration=final_model.best_iteration_)
y_preds_eval = final_model.predict(
    X_eval, num_iteration=final_model.best_iteration_)

final_mse_train = mean_squared_error(y_train, y_preds_train)
final_mse_eval = mean_squared_error(y_eval, y_preds_eval)

final_r2_train = r2_score(y_train, y_preds_train)
final_r2_eval = r2_score(y_eval, y_preds_eval)

print(f'Final MSE Train: {final_mse_train:.5f}, Final MSE Eval: {final_mse_eval:.5f}')
print('-------------------------------------------------')
print(f'Final R2 Train: {final_r2_train:.5f}, Final R2 Eval: {final_r2_eval:.5f}')
print('=================================================')

test = pd.read_csv('playground-series-s4e5/test.csv')
test = transform_df(test)

ids = test['id']

test_features = test.drop(['id'], axis=1)

test_features = scaler.fit_transform(test_features)

test_predict = final_model.predict(
    test_features, num_iteration=final_model.best_iteration_)

submission_df = pd.DataFrame({'id': ids, 'FloodProbability': test_predict})

submission_df.to_csv('submission_4.csv', index=False)
