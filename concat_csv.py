"""WEBからデータを取得する
"""
import pandas as pd
import glob
from tqdm import tqdm
from multiprocessing import Pool
import re

import config as cnf


def main():
    """
    """

    #concut(cnf.RACEINDEX_PATH, cnf.RACEINDEX_PATH_CONCUT)
    #concut(cnf.RESULTLIST_PATH_ORDER, cnf.RESULTLIST_PATH_ORDER_CONCUT)
    concut(cnf.RESULTLIST_PATH_RETURN, cnf.RESULTLIST_PATH_RETURN_CONCUT, border_date=20201009)
    concut(cnf.RACELIST_PATH, cnf.RACELIST_PATH_CONCUT, border_date=20201009)

def concut(input_path: str, output_path: str, border_date=0):
    """
    """
    df = read_csv(input_path, border_date)
    df.to_csv(output_path, index=False)

def read_csv(path, border_date):
    """
    """
    path = path.replace('.csv', '*.csv')
    all_files = glob.glob(path)

    pattern = r'_([0-9]{8})'
    prog = re.compile(pattern)

    files = []
    for f in all_files:
        d = int(prog.search(f).group(1))
        if d >= border_date:
            files.append(f)
    
    with Pool() as pool:
        imap = pool.imap(read_one_csv, files)
        dfs = list(tqdm(imap, desc=f'concut csv. path: {path}', total=len(files)))
    df = pd.concat(dfs)

    return df

def read_one_csv(path):
    """
    """
    df = pd.read_csv(path)
    
    return df

if __name__ == '__main__':
    main()
