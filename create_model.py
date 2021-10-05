"""WEBからデータを取得する
"""
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm
import datetime
import config as cnf
from sklearn.utils.class_weight import compute_sample_weight
import json

matplotlib.rcParams['font.family'] = 'Hiragino sans' # 日本語対応（Mac） Windowsは別の日本語対応フォントにする必要があるかも

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
        

    def search_prm(self, n_trials=None, timeout=None):
        """
        """
        study = optuna.create_study(direction='maximize')
        study.optimize(self.optimizer, n_trials=n_trials, timeout=timeout)

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

        # 予測
        y_pred = bst.predict(self.dtest).tolist()
        y_pred = [x for x in y_pred]
        self.y_test = [float(x) for x in self.y_test]
        
        # 評価
        #print(f'y_pred: {set(y_pred)}, self.y_test: {set(self.y_test)}')
        accuracy = accuracy_score(self.y_test, y_pred)
        #mlogloss = log_loss(self.y_test, y_pred, labels=self.y_test)
        score = accuracy
        #print(f'#{trial.number}, Result: {score}, {trial.params}')

        return score

def main():
    """
    """

    df = create_dataset()
    (train_x, test_x ,train_y, test_y) = create_traindata(df)
    #randam_forest(train_x, test_x ,train_y, test_y)
    xgboost(train_x, test_x ,train_y, test_y)

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

    key = ['レース場コード', '日付', 'レース']
    df = pd.merge(df_raselist, df_order[['レース場コード', '日付', 'レース', '着順1_艇番']], on=key, how='inner')
    df = df.fillna(0)

    return df

def create_traindata(df: pd.DataFrame):
    """
    """
    train_x = df.drop(columns=['着順1_艇番'])
    train_x = pd.get_dummies(train_x, drop_first=True)
    # 着順1_レーサーがNoneの場合は'＿'など文字列が入っているので、型がobjectになることがあるので、
    # '1.0'などの文字列を直接intに変換できないので、floatを経由する
    train_y = df['着順1_艇番']
    (train_x, test_x ,train_y, test_y) = train_test_split(train_x, train_y, test_size = 0.3, random_state = 42)

    print(set(train_y))
    print(set(test_y))
    diff_set = set(train_y) - set(test_y)
    if len(diff_set) > 0:
        raise Exception('テストデータと評価データのラベルが一致してません')

    return (train_x, test_x ,train_y, test_y)

def xgboost(X_train, X_test ,y_train, y_test):
    """
    """
    y_train -= 1
    y_test -= 1
    weight_train = compute_sample_weight(class_weight='balanced', y=y_train)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns, weight=weight_train)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_train.columns)

    print('パラメータ探索開始')
    xgb_params = {
        # 多値分類問題
        'objective': 'multi:softmax',
        # クラス数
        'num_class': 6,
        # 学習用の指標 (Multiclass logloss)
        'eval_metric': 'mlogloss',
        # 学習率
        'eta': 0.05,
    }
    prm_dict = {
        'max_depth': {'type': 'int', 'min': 5, 'max': 10}, # 決定木の深さの最大値
        'lambda': {'type': 'float', 'min': 0.8, 'max': 2.0}, # L2正則化項　大きくすると過学習防止
        'gamma': {'type': 'float', 'min': 0.0, 'max': 1.0}, # 決定木の葉の数に対するペナルティー
        'min_child_weight': {'type': 'float', 'min': 0.0, 'max': 1.0}, # 決定木の葉の重みの下限
    }
    num_boost_round = 10000 #最大ラウンド数
    early_stopping_rounds = 10 #精度向上がなくなったと判断するラウンド数
    sp = SearchParameters(dtrain, dtest, y_test, xgb_params, prm_dict, num_boost_round, early_stopping_rounds)
    best_prm = sp.search_prm(n_trials=100, timeout=3600)

    print('ベストパラメータで学習開始')
    for key in prm_dict.keys():    
        xgb_params[key] = best_prm[key]
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    evals_result = {}
    bst = xgb.train(xgb_params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    early_stopping_rounds=early_stopping_rounds,
                    evals=evals,
                    evals_result=evals_result,
    )

    print('精度確認開始')
    train_y_pred = bst.predict(dtrain)
    train_accuracy = accuracy_score(y_train, train_y_pred)
    train_accuracy = round(train_accuracy, 4)
    y_pred = bst.predict(dtest)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(accuracy, 4)
    print('Accuracy:', accuracy)

    now = datetime.datetime.now()
    now = now.strftime('%Y%m%d_%H%M%S')
    model_name = f'{now}_XGB_TACC={train_accuracy}_ACC={accuracy}'
    model_dir = os.path.join(cnf.OUTPUT_DIR, 'Model', model_name)
    os.makedirs(model_dir)

    model_file = os.path.join(model_dir, 'model.pkl')
    pickle.dump(bst, open(model_file, 'wb'))
    print(f'モデル: {model_file}')

    xgb_params['col'] = list(X_train.columns)
    prm_file = os.path.join(model_dir, 'params.json')
    print(xgb_params)
    print(type(xgb_params))
    with open(prm_file, 'w') as f:
        json.dump(xgb_params, f, indent=4, ensure_ascii=False)


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





def randam_forest(train_x, test_x ,train_y, test_y):
    """
    """
    # ランダムフォレストのパラメータの候補をいくつか決める
    parameters = {
        'n_estimators' :[30, 40, 50],#作成する決定木の数
        'random_state' :[42],
        'max_depth' :[5, 10, 15, 20],#決定木の深さ
        'min_samples_leaf': [5, 10, 20, 50],#分岐し終わったノードの最小サンプル数
        'min_samples_split': [5, 10, 20, 50]#決定木が分岐する際に必要なサンプル数
    }

    # グリッドサーチを使って識別モデルの構築
    print('グリッドサーチ開始')
    random_forest = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=2, iid=False)
    """
    print(train_x)
    print(train_y)
    train_x.info()
    """
    random_forest.fit(train_x, train_y)

    #精度を確認
    best_clf = random_forest.best_estimator_ #ここにベストパラメータの組み合わせが入っています
    random_forest = best_clf
    #モデルを作成する段階でのモデルの識別精度
    train_accuracy = random_forest.score(train_x, train_y)
    train_accuracy = round(train_accuracy, 2)
    print('TrainAccuracy: {:.2%}'.format(train_accuracy))
    #作成したモデルに学習に使用していない評価用のデータセットを入力し精度を確認
    y_pred = random_forest.predict(test_x)
    accuracy = best_clf.score(test_x, test_y)
    accuracy = round(accuracy, 2)
    print('Accuracy: {:.2%}'.format(accuracy))
    model_file = os.path.join(cnf.OUTPUT_DIR, f'RFmodel_TrainAccuracy:{train_accuracy}_Accuracy:{accuracy}.pkl')
    pickle.dump(random_forest, open(model_file, 'wb'))

    print(classification_report(test_y, y_pred))

    #confusion matrix
    mat = confusion_matrix(test_y, y_pred)
    sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='d', cmap='RdPu')
    plt.xlabel('predicted class')
    plt.ylabel('true value')
    confusion_path = os.path.join(cnf.OUTPUT_DIR, 'confusion.png')
    plt.savefig(
        confusion_path,
        dpi=150,
        format='png'
    )

    # 変数の重要度を可視化
    importance = pd.DataFrame({ '変数' :train_x.columns, '重要度' :random_forest.feature_importances_})
    importance_path = os.path.join(cnf.OUTPUT_DIR, 'importance.csv')
    importance.sort_values('重要度', ascending=False).to_csv(importance_path, index=False)

    # 決定木の可視化
    write_tree(random_forest, train_x.columns, list(set(train_y.apply(lambda x: str(x)))))

def write_tree(random_forest, feature_names, class_names):
    """
    """
    for i in tqdm(range(len(random_forest.estimators_)), desc='決定木出力'):
        fig_dt = plt.figure(figsize=(128,8))
        ax_dt = fig_dt.add_subplot(111)
        tree.plot_tree(
            random_forest.estimators_[i],
            feature_names=feature_names,
            class_names=class_names,
            filled=True, 
            rounded=True, 
            proportion=True,
            fontsize=5,
            #impurity=False,
            ax=ax_dt
        )
        image_path = os.path.join(cnf.OUTPUT_DIR, 'tree' + str(i) + '.png')
        plt.savefig(
            image_path,
            dpi=150,
            format='png'
        )
    
    plt.close()

if __name__ == '__main__':
    main()
