"""WEBからデータを取得する
"""
import pandas as pd
from tqdm import tqdm
import numpy as np
import config as cnf


def main():
    """
    """

    usecols = [
        'レース場コード',
        '日付',
        'レース',
        '枠',
        '登録番号',
        '級別',
        #'氏名',
        #'支部',
        #'出身地',
        '年齢',
        '体重',
        'F数',
        'L数',
        '平均ST',
        '全国_勝率',
        '全国_2連率',
        '全国_3連率',
        '当地_勝率',
        '当地_2連率',
        '当地_3連率',
        'モーター_No',
        'モーター_2連率',
        'モーター_3連率',
        'ボート_No',
        'ボート_2連率',
        'ボート_3連率',
    ]
    keys = [
        'レース場コード', 
        '日付', 
        'レース'
    ]
    df = pd.read_csv(cnf.RACELIST_PATH_CONCUT, usecols=usecols, low_memory=False)
    df = df.drop_duplicates()
    df = set_dtype(df)
    drop_cols = ['モーター_No', 'ボート_2連率', 'ボート_3連率', '登録番号']
    df = df.drop(columns=drop_cols)
    df.info()
    lanes = set(df['枠'])
    df_merge = None
    for lane in tqdm(lanes, desc='枠で横持ち'):
        df_lane = df.query('枠 == @lane')
        df_lane = df_lane.drop(columns='枠')
        if df_merge is None:
            df_merge = df_lane.copy()
            first_lane = lane
        else:
            df_merge = pd.merge(df_merge, df_lane, how='left', on=keys, suffixes=['', f'_{lane}枠'])

    rename_dict = {}
    for col in usecols:
        if col not in keys:
            rename_dict[col] = col + f'_{first_lane}枠'

    df_merge = df_merge.rename(columns=rename_dict)
    print('ソート開始')
    df_merge = df_merge.sort_values(['レース場コード', '日付', 'レース'])
    df_merge.to_csv(cnf.RACELIST_PATH_FORMAT, index=False)
    df_merge.info()
    print(df_merge.describe())

def set_dtype(df):
    """
    """
    df = astype(df, 'レース場コード', np.int8)
    df = astype(df, '日付', np.int32)
    df = astype(df, 'レース', np.int8)
    df = astype(df, '枠', np.int8)
    df = astype(df, '登録番号', np.int16)
    df = astype(df, '年齢', np.int8)
    df = astype(df, 'F数', np.int8)
    df = astype(df, 'L数', np.int8)
    df = astype(df, 'モーター_No', np.int8)
    df = astype(df, 'ボート_No', np.int8)
    df = astype(df, '体重', np.float16)
    df = astype(df, '平均ST', np.float16)
    df = astype(df, '全国_勝率', np.float64)
    df = astype(df, '全国_2連率', np.float64)
    df = astype(df, '全国_3連率', np.float64)
    df = astype(df, '当地_勝率', np.float64)
    df = astype(df, '当地_2連率', np.float64)
    df = astype(df, '当地_3連率', np.float64)
    df = astype(df, 'モーター_2連率', np.float64)
    df = astype(df, 'モーター_3連率', np.float64)
    df = astype(df, 'ボート_2連率', np.float64)
    df = astype(df, 'ボート_3連率', np.float64)

    return df

def astype(df, col, type):
    """
    """
    df[col] = df[col].astype(str)
    df[col] = df[col].str.replace('-', '0')
    df[col] = df[col].str.replace('F', '')
    df[col] = df[col].str.replace('L', '')
    df[col] = df[col].astype(type)

    return df 

if __name__ == '__main__':
    main()
