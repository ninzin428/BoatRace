"""WEBからデータを取得する
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
import glob

import config as cnf


def main():
    """
    """



    df_raseindex = pd.read_csv(cnf.RACEINDEX_PATH_CONCUT)
    df_order = pd.read_csv(cnf.RESULTLIST_PATH_ORDER_CONCUT)
    df_order = df_order.query('着順1_レーサー == 着順1_レーサー')

    key = ['レース場コード', '日付', 'レース']
    df = pd.merge(df_raseindex, df_order[['レース場コード', '日付', 'レース', '着順1_艇番']], on=key, how='left')
    df = df.fillna(0)
    train_x = df.drop(columns=['着順1_艇番'])
    train_x = pd.get_dummies(train_x)
    # 着順1_レーサーがNoneの場合は'＿'など文字列が入っているので、型がobjectになることがあるので、
    # '1.0'などの文字列を直接intに変換できないので、floatを経由する
    train_y = df['着順1_艇番'].astype(float).astype(int)
    (train_x, test_x ,train_y, test_y) = train_test_split(train_x, train_y, test_size = 0.3, random_state = 42)

    # 識別モデルの構築
    random_forest = RandomForestClassifier(max_depth=30, n_estimators=30, random_state=42)
    print(train_x)
    print(train_y)
    train_x.info()
    random_forest.fit(train_x, train_y)

    # 予測値算出
    y_pred = random_forest.predict(test_x)

    #モデルを作成する段階でのモデルの識別精度
    trainaccuracy_random_forest = random_forest.score(train_x, train_y)
    print('TrainAccuracy: {}'.format(trainaccuracy_random_forest))

    #作成したモデルに学習に使用していない評価用のデータセットを入力し精度を確認
    accuracy_random_forest = accuracy_score(test_y, y_pred)
    print('Accuracy: {}'.format(accuracy_random_forest))


if __name__ == '__main__':
    main()
