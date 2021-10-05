"""WEBからデータを取得する
"""
import pandas as pd
import glob
from tqdm import tqdm
from multiprocessing import Pool

import config as cnf


def main():
    """
    """

    #concut(cnf.RACEINDEX_PATH, cnf.RACEINDEX_PATH_CONCUT)
    concut(cnf.RESULTLIST_PATH_ORDER, cnf.RESULTLIST_PATH_ORDER_CONCUT)
    concut(cnf.RESULTLIST_PATH_RETURN, cnf.RESULTLIST_PATH_RETURN_CONCUT)
    #concut(cnf.RACELIST_PATH, cnf.RACELIST_PATH_CONCUT)

def concut(input_path: str, output_path: str, suffix='*'):
    """
    """
    df = read_csv(input_path, suffix)
    df.to_csv(output_path, index=False)

def read_csv(path, suffix):
    """
    """
    path = path.replace('.csv', f'{suffix}.csv')
    files = glob.glob(path)
    dfs = []
    with Pool() as pool:
        imap = pool.imap(read_one_csv, files)
        dfs = list(tqdm(imap, desc=f'concut csv. path: {path}', total=len(files)))

    """
    for file in tqdm(files, desc=f'結合中: {path}'):
        df = pd.read_csv(file)
        dfs.append(df)
    """
    df = pd.concat(dfs)

    return df

def read_one_csv(path):
    """
    """
    df = pd.read_csv(path)
    
    return df

if __name__ == '__main__':
    main()
