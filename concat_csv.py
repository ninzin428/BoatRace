"""WEBからデータを取得する
"""
import pandas as pd
import glob

import config as cnf


def main():
    """
    """

    concut(cnf.RACEINDEX_PATH, cnf.RACEINDEX_PATH_CONCUT)
    concut(cnf.RESULTLIST_PATH_ORDER, cnf.RESULTLIST_PATH_ORDER_CONCUT)
    concut(cnf.RESULTLIST_PATH_RETURN, cnf.RESULTLIST_PATH_RETURN_CONCUT)

def concut(input_path: str, output_path: str):
    """
    """
    df = read_csv(input_path)
    df.to_csv(output_path, index=False)

def read_csv(path):
    """
    """
    files = glob.glob(path.replace('.csv', '*.csv'))
    list = []
    for file in files:
        print(file)
        df = pd.read_csv(file)
        list.append(df)
    df = pd.concat(list)

    return df

if __name__ == '__main__':
    main()
