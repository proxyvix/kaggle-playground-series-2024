import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import optuna
from lightgbm import LGBMClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

train_df = pd.read_csv('playground-series-s4e6/train.csv')
test_df = pd.read_csv('playground-series-s4e6/test.csv')

scaler = StandardScaler()
le = LabelEncoder()

train_split = .8
random_state = 42
opt_iter = 200

total_missing_values = train_df.isnull().sum().sum()

print(total_missing_values)


def transfrom_df(df):

    df['sem12_approved'] = df[[
        'Curricular units 1st sem (approved)',
        'Curricular units 2nd sem (approved)'
    ]].sum(axis=1)

    df['sem12_approved_mean'] = df[[
        'Curricular units 1st sem (approved)',
        'Curricular units 2nd sem (approved)'
    ]].mean(axis=1)

    df['sem1'] = df[[
        'Curricular units 1st sem (credited)',
        'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)',
        'Curricular units 1st sem (without evaluations)'
    ]].sum(axis=1)

    df['sem2'] = df[[
        'Curricular units 2nd sem (credited)',
        'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)',
        'Curricular units 2nd sem (without evaluations)'
    ]].sum(axis=1)

    df_agg = df.groupby('Course').agg({
        'Previous qualification (grade)': ['mean', 'max'],
        'Admission grade': ['mean', 'min'],
        'Curricular units 1st sem (approved)': ['sum', 'mean'],
        'Curricular units 2nd sem (approved)': ['sum', 'mean']
    }).reset_index()

    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    df_agg.rename(columns={'Course_': 'Course'}, inplace=True)

    df = pd.merge(df, df_agg, on='Course', how='left')

    df = pd.get_dummies(df,
                        columns=['Marital status'],
                        dtype=int)

    scalable_cols = list(df.select_dtypes(include=['float64']).columns)

    additional_cols = ('Curricular units 1st sem (credited)',
                       'Curricular units 1st sem (enrolled)',
                       'Curricular units 1st sem (evaluations)',
                       'Curricular units 1st sem (without evaluations)',
                       'Curricular units 2nd sem (credited)',
                       'Curricular units 2nd sem (enrolled)',
                       'Curricular units 2nd sem (evaluations)',
                       'Curricular units 2nd sem (without evaluations)',
                       'sem1',
                       'sem2',
                       )

    scalable_cols.extend(additional_cols)

    df[scalable_cols] = scaler.fit_transform(df[scalable_cols])

    return df


# %% Train %%

train_df.drop('id', axis=1)
train_df = transfrom_df(train_df)

train, eval = train_test_split(train_df,
                               train_size=train_split,
                               random_state=random_state)

X_train, y_train = train.drop('Target', axis=1), train['Target']
X_eval, y_eval = eval.drop('Target', axis=1), eval['Target']

y_train = le.fit_transform(y_train)
y_eval = le.fit_transform(y_eval)

model_CBC = CatBoostClassifier(random_state=random_state, verbose=0)
model_XGB = xgb.XGBClassifier(random_state=random_state, verbosity=0)
model_LGBM = LGBMClassifier(random_state=random_state, verbosity=-1)

model_CBC.fit(X_train, y_train)
model_XGB.fit(X_train, y_train)
model_LGBM.fit(X_train, y_train)

y_train_pred_CBC = model_CBC.predict(X_train)
y_eval_pred_CBC = model_CBC.predict(X_eval)

y_train_pred_XGB = model_XGB.predict(X_train)
y_eval_pred_XGB = model_XGB.predict(X_eval)

y_train_pred_LGBM = model_LGBM.predict(X_train, num_iteration=model_LGBM.best_iteration_)
y_eval_pred_LGBM = model_LGBM.predict(X_eval, num_iteration=model_LGBM.best_iteration_)

train_acc_CBC = accuracy_score(y_true=y_train, y_pred=y_train_pred_CBC)
eval_acc_CBC = accuracy_score(y_true=y_eval, y_pred=y_eval_pred_CBC)

train_acc_XGB = accuracy_score(y_true=y_train, y_pred=y_train_pred_XGB)
eval_acc_XGB = accuracy_score(y_true=y_eval, y_pred=y_eval_pred_XGB)

train_acc_LGBM = accuracy_score(y_true=y_train, y_pred=y_train_pred_LGBM)
eval_acc_LGBM = accuracy_score(y_true=y_eval, y_pred=y_eval_pred_LGBM)

print('=============================')
print(f'Train, CBC: {train_acc_CBC:.5f}, XGB: {train_acc_XGB:.5f}, LGBM: {train_acc_LGBM:.5f}')
print('-----------------------------')
print(f'Eval CBC: {eval_acc_CBC:.5f}, XGB: {eval_acc_XGB:.5f}, LGBM: {eval_acc_LGBM:.5f}')
print('=============================')

voting_clf = VotingClassifier(estimators=[
    ('cbc', model_CBC),
    ('xgb', model_XGB),
    ('lgbm', model_LGBM)
], voting='soft')

voting_clf.fit(X_train, y_train)
y_eval_pred_voting = voting_clf.predict(X_eval)
eval_acc_voting = accuracy_score(y_true=y_eval, y_pred=y_eval_pred_voting)

print('=============================')
print(f'Eval Voting: {eval_acc_voting:.5f}')
print('=============================')

cv_score = cross_val_score(voting_clf, X_eval, y_eval, cv=5, scoring='accuracy')

print('=============================')
print(f'Eval CV: {cv_score.mean():.5f}')
print('=============================')

# %% Hyperparameter tuning %%


def objective_CBC(trial):

    param = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
        'iterations': trial.suggest_int('iterations', 100, 1500),
        'boosting_type': 'Plain',
        'bootstrap_type': 'Bernoulli',
        'thread_count': 4,
        'task_type': 'GPU',
        'loss_function': 'MultiClass',
        'random_strength': trial.suggest_float('random_strength', 1e-3, 1e-1, log=True),
    }

    model = CatBoostClassifier(**param,
                               verbose=0,
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


study_CBC = optuna.create_study(direction='maximize')
study_CBC.optimize(objective_CBC, n_trials=opt_iter)


def objective_XGB(trial):

    param = {
        'verbosity': 0,
        'objective': 'multi:softmax',
        'num_class': 3,
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

    model = xgb.XGBClassifier(**param,
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
study_XGB.optimize(objective_XGB, n_trials=opt_iter)


def objective_LGBM(trial):

    params = {
        'num_leaves': trial.suggest_int('num_leaves', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 300000),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'subsample': trial.suggest_float('subsample', 0.25, 1.0),
        'max_depth': trial.suggest_int('max_depth', 1, 50),
        'num_class': 3
    }

    model = LGBMClassifier(**params,
                           objective='multiclass',
                           random_state=random_state,
                           verbosity=-1)

    model.fit(X_train,
              y_train)

    y_pred_train = model.predict(X_train, num_iteration=model.best_iteration_)
    y_pred_eval = model.predict(X_eval, num_iteration=model.best_iteration_)

    acc_tuning_train = accuracy_score(y_true=y_train, y_pred=y_pred_train)
    acc_tuning_eval = accuracy_score(y_true=y_eval, y_pred=y_pred_eval)

    print('=============================================')
    print(f'Accuracy Train: {acc_tuning_train:.5f}, Eval: {acc_tuning_eval:.5f}')
    print('=============================================')

    return acc_tuning_eval


study_LGBM = optuna.create_study(direction='maximize')
study_LGBM.optimize(objective_LGBM, n_trials=opt_iter)

print('=============================================')
print(f'Best accuracy CBC: {study_CBC.best_value:.5f}')
print('---------------------------------------------')
print(f'Best accuracy XGB: {study_XGB.best_value:.5f}')
print('---------------------------------------------')
print(f'Best accuracy LGBM: {study_LGBM.best_value:.5f}')
print('=============================================')

params_CBC = study_CBC.best_params
params_XGB = study_XGB.best_params
params_LGBM = study_LGBM.best_params

tuned_model_CBC = CatBoostClassifier(**params_CBC,
                                     verbose=0,
                                     random_state=random_state)

tuned_model_CBC.fit(X_train,
                    y_train)

tuned_model_XGB = xgb.XGBClassifier(**params_XGB,
                                    use_label_encoder=False,
                                    eval_metric='mlogloss',
                                    device='cuda',
                                    predictors='gpu_predictor',
                                    random_state=random_state)

tuned_model_XGB.fit(X_train,
                    y_train)

tuned_model_LGBM = LGBMClassifier(verbosity=-1,
                                  random_state=random_state,
                                  objective='multiclass',
                                  **params_LGBM)

tuned_model_LGBM.fit(X_train,
                     y_train)

voting_clf = VotingClassifier(estimators=[
    ('cbc', tuned_model_CBC),
    ('xgb', tuned_model_XGB),
    ('lgbm', tuned_model_LGBM)
], voting='soft')

voting_clf.fit(X_train, y_train)

y_eval_pred_voting = voting_clf.predict(X_eval)
eval_acc_voting = accuracy_score(y_true=y_eval, y_pred=y_eval_pred_voting)

print('=============================')
print(f'Eval Voting: {eval_acc_voting:.5f}')
print('=============================')

test_df = transfrom_df(test_df)

id = test_df['id']
test = test_df.drop('id', axis=1)

prediction = voting_clf.predict(test_df)

submission = pd.DataFrame({'id': id,
                           'Target': le.inverse_transform(prediction)})

submission.to_csv('submission_05.csv', index=False)
