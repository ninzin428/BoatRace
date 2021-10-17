"""WEBからデータを取得する
"""
import datetime
import json
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

import config as cnf

matplotlib.rcParams['font.family'] = 'Hiragino sans' # 日本語対応（Mac） Windowsは別の日本語対応フォントにする必要があるかも

target_col = '組合わせ_2連勝単式'

class SearchParameters:
    """
    """
    def __init__(
        self, 
        dtrain, 
        dtest, 
        y_test, 
        xgb_params, 
        prm_dict, 
        num_boost_round, 
        early_stopping_rounds,
    ):
        """
        """
        self.dtrain = dtrain
        self.dtest = dtest
        self.y_test = y_test
        self.prm_dict = prm_dict
        self.xgb_params = xgb_params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.models = []
        self.best_model = None
        self.evals_results = []
        self.best_evals_result = None
        

    def search_prm(self, n_trials=None, timeout=None):
        """
        """
        study = optuna.create_study(direction='maximize')
        study.optimize(self.optimizer, n_trials=n_trials, timeout=timeout)
        self.best_model = self.models[study.best_trial.number]
        self.best_evals_result = self.evals_results[study.best_trial.number]
        self.best_trial = study.best_trial
        print(f'パラメータ探索開始時間: {study.trials[0].datetime_start}')
        print(f'パラメータ探索終了時間: {study.trials[-1].datetime_complete}')


        return study.best_params

    def optimizer(self, trial):
        """
        """
        # 探索するパラメータをセット
        for key, val in self.prm_dict.items():
            if val['type'] == 'int':
                self.xgb_params[key] = trial.suggest_int(key, val['min'], val['max'])
            else:
                self.xgb_params[key] = trial.suggest_uniform(key, val['min'], val['max'])

        evals = [(self.dtrain, 'train'), (self.dtest, 'eval')]
        evals_result = {}

        # 学習
        #pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-merror")
        bst = xgb.train(self.xgb_params,
                        self.dtrain,
                        num_boost_round=self.num_boost_round,
                        early_stopping_rounds=self.early_stopping_rounds,
                        evals=evals,
                        evals_result=evals_result,
                        #callbacks=[pruning_callback]
                        )
        self.models.append(bst)
        self.evals_results.append(evals_result)

        # 予測
        y_pred = bst.predict(self.dtest).tolist()

        # 評価
        accuracy = calc_accuracy(self.y_test, y_pred)
        score = accuracy

        return score

def main():
    """
    """

    df, label_dict = create_dataset()
    (train_x, test_x ,train_y, test_y) = create_traindata(df)
    #randam_forest(train_x, test_x ,train_y, test_y)
    xgboost(train_x, test_x ,train_y, test_y, label_dict)

def create_dataset():
    """
    """
    #usecols = ['レース場コード', '日付', 'レース', '出場レーサー階級1', '出場レーサー階級2', '出場レーサー階級3', '出場レーサー階級4', '出場レーサー階級5', '出場レーサー階級6']
    #df_raseindex = pd.read_csv(cnf.RACEINDEX_PATH_CONCUT, usecols=usecols)
    df_raselist = pd.read_csv(cnf.RACELIST_PATH_FORMAT)

    df_order = pd.read_csv(cnf.RESULTLIST_PATH_ORDER_CONCUT)
    df_order = df_order.query('着順1_レーサー == 着順1_レーサー')
    df_order = df_order[['レース場コード', '日付', 'レース', '着順1_艇番']]
    df_order['レース'] = df_order['レース'].str.replace('R', '')
    # 着順1_レーサーがNoneの場合は'＿'など文字列が入っているので、型がobjectになることがあるので、
    # '1.0'などの文字列を直接intに変換できないので、floatを経由する
    df_order = df_order.astype(float).astype(int)

    df_return = pd.read_csv(cnf.RESULTLIST_PATH_RETURN_CONCUT)
    df_return = df_return[['レース場コード', '日付', 'レース', '2連勝単式組合わせ']]
    df_return = df_return.rename(columns={'2連勝単式組合わせ': '組合わせ_2連勝単式'})
    ng_list = ['特払', '不成立', 'レース中止']
    df_return = df_return.query(r'組合わせ_2連勝単式.str.match("^[1-6]-[1-6]$")', engine='python').copy()
    print(df_return['組合わせ_2連勝単式'].value_counts())
    df_return['組合わせ_2連勝単式'], label_dict = label_encode(df_return['組合わせ_2連勝単式'])
    df_return['レース場コード'] = df_return['レース場コード'].astype(int)
    df_return['日付'] = df_return['日付'].astype(int)
    df_return['レース'] = df_return['レース'].str.replace('R', '')
    df_return['レース'] = df_return['レース'].astype(int)

    key = ['レース場コード', '日付', 'レース']
    df = pd.merge(df_raselist, df_return, on=key, how='inner')
    df = df.fillna(0)

    df = df.query('日付 >= 20200101')

    return df, label_dict

def label_encode(series: pd.Series):
    """
    """
    #LabelEncoderのインスタンスを生成
    le = LabelEncoder()
    #ラベルを覚えさせる
    le = le.fit(series)
    #ラベルを整数に変換
    series = le.transform(series)

    label_dict = {}
    for value in set(series):
        label_dict[value] = le.inverse_transform([value])[0]

    return series, label_dict

def create_traindata(df: pd.DataFrame):
    """
    """
    train_x = df.drop(columns=['組合わせ_2連勝単式'])
    train_x = pd.get_dummies(train_x, drop_first=True)
    # 着順1_レーサーがNoneの場合は'＿'など文字列が入っているので、型がobjectになることがあるので、
    # '1.0'などの文字列を直接intに変換できないので、floatを経由する
    train_y = df['組合わせ_2連勝単式']
    (train_x, test_x ,train_y, test_y) = train_test_split(train_x, train_y, test_size=0.3, random_state=42, stratify=train_y)

    print(f'学習データラベル: {set(train_y)}')
    print(f'評価データラベル: {set(test_y)}')
    diff_set = set(train_y) - set(test_y)
    if len(diff_set) > 0:
        raise Exception('テストデータと評価データのラベルが一致してません')

    return (train_x, test_x ,train_y, test_y)

def xgboost(X_train, X_test ,y_train, y_test, label_dict):
    """
    """
    weight_train = compute_sample_weight(class_weight='balanced', y=y_train)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns, weight=weight_train)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_train.columns)

    print('パラメータ探索開始')
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    xgb_params = {
        # 多値分類問題
        'objective': 'multi:softprob',
        # クラス数
        'num_class': len(set(y_train)),
        # 学習用の指標 (Multiclass logloss)
        'eval_metric': 'mlogloss',
        # ツリー構築アルゴリズム
        'tree_method': 'hist', #より高速なヒストグラム最適化近似欲張りアルゴリズム。xgboostで最速のアルゴリズム。 default: auto
        # 学習率
        'eta': 0.1, # 0~1
        'subsample': 0.7,
        'colsample_bytree': 0.5,

    }
    prm_dict = {
        # 木構造
        'max_depth': {'type': 'int', 'min': 10, 'max': 15}, # 0~inf 決定木の深さの最大値
        'min_child_weight': {'type': 'float', 'min': 0.0, 'max': 5.0}, # 0~inf 決定木の葉の重みの下限
        'gamma': {'type': 'float', 'min': 0.0, 'max': 5.0}, # 0~inf ツリーのリーフノードにさらにパーティションを作成するために必要な最小の損失削減
        # 正則化
        'lambda': {'type': 'float', 'min': 0.0, 'max': 5.0}, # L2正則化項　大きくすると過学習防止
        'alpha': {'type': 'float', 'min': 0.0, 'max': 5.0}, # 重みに関するL1正則化項 高次元の場合に用いるらしい。
        # サンプリング
        #'subsample': {'type': 'float', 'min': 0.5, 'max': 1.0}, # 0~1 各STEPの決定木の構築に用いるデータの割合
        #'colsample_bytree': {'type': 'float', 'min': 0.5, 'max': 1.0}, # 0~1 各STEPの決定木に用いる特徴量の割合
        #'colsample_bylevel': {'type': 'float', 'min': 0.5, 'max': 1.0}, # 0~1 各STEPの決定木のレベルに用いる特徴量の割合
        #'colsample_bynode': {'type': 'float', 'min': 0.5, 'max': 1.0}, # 0~1 各STEPの決定木のノードに用いる特徴量の割合
    }
    num_boost_round = 10000 #最大ラウンド数
    early_stopping_rounds = 3 #精度向上がなくなったと判断するラウンド数
    sp = SearchParameters(dtrain, dtest, y_test, xgb_params, prm_dict, num_boost_round, early_stopping_rounds)
    best_prm = sp.search_prm(n_trials=30, timeout=None)
    bst = sp.best_model
    evals_result = sp.best_evals_result
    print(f'パラメータ: {xgb_params}')

    print('精度確認開始')
    train_y_pred = bst.predict(dtrain)
    train_accuracy = calc_accuracy(y_train, train_y_pred)
    y_pred = bst.predict(dtest)
    accuracy = calc_accuracy(y_test, y_pred)

    now = datetime.datetime.now()
    now = now.strftime('%Y%m%d_%H%M%S')
    model_name = f'{now}_XGB_TACC={train_accuracy}_ACC={accuracy}'
    model_dir = os.path.join(cnf.OUTPUT_DIR, 'Model', model_name)
    os.makedirs(model_dir)

    model_file = os.path.join(model_dir, 'model.pkl')
    pickle.dump(bst, open(model_file, 'wb'))

    xgb_params['col'] = list(X_train.columns)
    print(label_dict)
    print(type(label_dict))
    label_dict = dict(zip([int(k) for k in label_dict.keys()], label_dict.values()))
    xgb_params['label'] = label_dict
    prm_file = os.path.join(model_dir, 'params.json')
    with open(prm_file, 'w') as f:
        json.dump(xgb_params, f, indent=4, ensure_ascii=False)

    y_test = [label_dict[x] for x in y_test]
    y_pred = [np.argmax(prob) for prob in y_pred]
    y_pred = [label_dict[x] for x in y_pred]
    c_report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    df_c_report = pd.DataFrame(c_report).T
    report_file = os.path.join(model_dir, 'classification_report.csv')
    df_c_report.to_csv(report_file)
    print(df_c_report)

    X_test['着順1_艇番'] = y_test
    X_test['着順1_艇番_予測'] = y_pred

    result = os.path.join(model_dir, 'Result.csv')
    X_test.to_csv(result, index=False)

    train_metric = evals_result['train']['mlogloss']
    plt.plot(train_metric, label='train logloss')
    eval_metric = evals_result['eval']['mlogloss']
    plt.plot(eval_metric, label='eval logloss')
    plt.grid()
    plt.legend()
    plt.xlabel('rounds')
    plt.ylabel('logloss')
    image_path = os.path.join(model_dir, 'mlogloss.png')
    plt.savefig(
        image_path,
        dpi=150,
        format='png'
    )
    plt.close()

    # 性能向上に寄与する度合いで重要度をプロットする
    _, ax = plt.subplots(figsize=(48, 24))
    xgb.plot_importance(
        bst,
        ax=ax,
        importance_type='gain',
        show_values=True,
        max_num_features=50,
    )
    image_path = os.path.join(model_dir, 'importance.png')
    plt.savefig(
        image_path,
        dpi=150,
        format='png'
    )

def calc_accuracy(y, y_pred):
    """
    """
    y_pred = [np.argmax(prob) for prob in y_pred]
    y_pred = [float(x) for x in y_pred]
    y = [float(x) for x in y]
    accuracy = accuracy_score(y, y_pred)
    accuracy = round(accuracy, 4)

    return accuracy

if __name__ == '__main__':
    main()
