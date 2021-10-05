"""WEBからデータを取得する
"""
import os
import pickle

import matplotlib
import pandas as pd
import xgboost as xgb
import json

import config as cnf

matplotlib.rcParams['font.family'] = 'Hiragino sans' # 日本語対応（Mac） Windowsは別の日本語対応フォントにする必要があるかも

DATE = 20210926
MODEL_NAME = 'XGBmodel_TrainAccuracy:0.6379_Accuracy:0.5776'

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
    df = df_raseindex.query(f'日付 == {DATE}')

    df.info()
    print(df)
    
    return df

def create_traindata(df: pd.DataFrame):
    """
    """
    df['着順1_艇番'] = '-1.0'
    train_x = df.drop(columns=['着順1_艇番'])
    train_x = pd.get_dummies(train_x)
    train_x = give_columns(train_x)
    train_x.info()
    print(train_x)


    # 着順1_レーサーがNoneの場合は'＿'など文字列が入っているので、型がobjectになることがあるので、
    # '1.0'などの文字列を直接intに変換できないので、floatを経由する
    train_y = df['着順1_艇番'].astype(float).astype(int)

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
    dtest = xgb.DMatrix(train_x, label=train_y)
    model_file = os.path.join(model_dir, 'model.pkl')
    bst = pickle.load(open(model_file, 'rb'))

    y_pred = bst.predict(dtest)

    train_x['着順1_艇番'] = train_y
    train_x['着順1_艇番_予測'] = y_pred
    train_x['着順1_艇番_予測'] += 1
    train_x['レース場'] = train_x['レース場コード'].apply(lambda x: cnf.JCDS[str(int(x)).zfill(2)])
    train_x = train_x.sort_values(['レース場コード', '日付', 'レース'])
    train_x = train_x[['レース場コード', 'レース場', '日付', 'レース', '着順1_艇番_予測']]
    result = os.path.join(model_dir, f'Predict_Date={DATE}.csv')
    train_x.to_csv(result, index=False)


if __name__ == '__main__':
    main()
