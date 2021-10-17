"""WEBからデータを取得する
"""
import os
import pickle

import matplotlib
import pandas as pd
import xgboost as xgb
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import config as cnf

matplotlib.rcParams['font.family'] = 'Hiragino sans' # 日本語対応（Mac） Windowsは別の日本語対応フォントにする必要があるかも

DATE = 20211010
MODEL_NAME = '20211010_023441_XGB_TACC=0.1166_ACC=0.0702'

model_dir = os.path.join(cnf.OUTPUT_DIR, 'Model', MODEL_NAME)


def main():
    """
    """

    df = create_dataset()
    (train_x ,train_y) = create_traindata(df)
    forecast(train_x, train_y)

def create_dataset():
    """
    """
    df_raseindex = pd.read_csv(cnf.RACELIST_PATH_FORMAT)
    print(df_raseindex['日付'].max())
    df = df_raseindex.query(f'日付 == {DATE}')

    df.info()
    print(df)
    
    return df

def create_traindata(df: pd.DataFrame):
    """
    """
    df[cnf.TARGET_COL] = '-1.0'
    train_x = df.drop(columns=[cnf.TARGET_COL])
    train_x = pd.get_dummies(train_x)
    train_x = give_columns(train_x)
    train_x.info()
    print(train_x)


    # 着順1_レーサーがNoneの場合は'＿'など文字列が入っているので、型がobjectになることがあるので、
    # '1.0'などの文字列を直接intに変換できないので、floatを経由する
    train_y = df[cnf.TARGET_COL].astype(float).astype(int)

    print(set(train_y))

    return (train_x, train_y)

def give_columns(df: pd.DataFrame):
    """
    """
    prm_file = os.path.join(model_dir, 'params.json')
    with open(prm_file) as f:
        prm = json.load(f)

    print(f'入力データカラム: {df.columns}')
    for col in prm['col']:
        if col not in df.columns:
            df[col] = 0

    return df[prm['col']]

def forecast(train_x, train_y):
    """
    """
    # 予測準備
    dtest = xgb.DMatrix(train_x, label=train_y)
    model_file = os.path.join(model_dir, 'model.pkl')
    bst = pickle.load(open(model_file, 'rb'))

    # 予測    
    y_pred = bst.predict(dtest)
    prm_file = os.path.join(model_dir, 'params.json')
    with open(prm_file) as f:
        prm = json.load(f)
    label = prm['label']
    df_pred = pd.DataFrame(data=y_pred, columns=label.values())
    y_pred = [np.argmax(prob) for prob in y_pred]
    y_pred = [label[str(x)] for x in y_pred]

    # 成形
    pred_col = f'{cnf.TARGET_COL}_予測'
    train_x[cnf.TARGET_COL] = train_y
    train_x[pred_col] = y_pred
    train_x['レース場'] = train_x['レース場コード'].apply(lambda x: cnf.JCDS[str(int(x)).zfill(2)])
    train_x = train_x.sort_values(['レース場コード', '日付', 'レース'])
    train_x = train_x[['レース場コード', 'レース場', '日付', 'レース', pred_col]]
    train_x = train_x.reset_index(drop=True)
    train_x = pd.concat([train_x, df_pred], axis=1)
    for v in label.values():
        train_x[v] = train_x[v].round(6)

    # 出力
    result = os.path.join(model_dir, f'Predict_Date={DATE}.csv')
    train_x.to_csv(result, index=False)

    fig, ax = plt.subplots(figsize=(50, 50))
    train_x = train_x.drop(columns=['レース場コード', '日付'])
    train_x = train_x.set_index(['レース場', 'レース'])
    train_x.plot(kind='bar', stacked=True, ax=ax)
    plt.title(DATE)
    image_path = os.path.join(model_dir, f'Predict_Date={DATE}.png')
    plt.savefig(
        image_path,
        dpi=150,
        format='png'
    )
    plt.close()


if __name__ == '__main__':
    main()
