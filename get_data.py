"""WEBからデータを取得する
"""
import datetime
import time

import pandas as pd
from tqdm import tqdm

import config as cnf


def main():
    raceindex = RaceIndex()
    resultlist = Resultlist()
    today = datetime.datetime.now()
    for jcd in cnf.JCDS.keys():
        date = cnf.START_DATE
        term = (today - date).days + 1
        for i in tqdm(range(term), desc=f'jcd: {jcd} get html table.'):
            raceindex.read_one_day(jcd=jcd, date=date)
            resultlist.read_one_day(jcd=jcd, date=date)


            date = date + datetime.timedelta(days=1)

class Data():

    def __init__(self,
    ):

        self.dfs = []

    def read_html(self, url: str) -> None:
        """
        """
        try:
            self.dfs = pd.read_html(url)

        except Exception as e:
            self.dfs = []

    def write_csv(self, df: pd.DataFrame, output_path: str, output_cols: list) -> None:
        """
        """
        df = df[output_cols]
        df.to_csv(output_path, index=False)

    def get_safix_output_path(self, output_path: str, safix: str):
        return output_path.replace('.csv', f'{safix}.csv')

class RaceIndex(Data):
    """
    """
    def read_one_day(self, jcd: int, date: datetime.date) -> None:
        """
        """
        yyyymmdd = date.strftime('%Y%m%d')
        url = cnf.RACEINDEX_URL.format(jcd=jcd, yyyymmdd=yyyymmdd)
        self.read_html(url)
        if len(self.dfs) > 0:
            df = self.dfs[0]
            df.columns = cnf.RACEINDEX_TABLE_COLS

            for i in range(6):
                df = self.split_racer(df, f'出場レーサー{i + 1}')

            df['レース場コード'] = jcd
            df['日付'] = yyyymmdd

            output_path = self.get_safix_output_path(cnf.RACEINDEX_PATH, f'_{jcd}_{yyyymmdd}')
            self.write_csv(df, output_path, cnf.RACEINDEX_CSV_COLS)

    def split_racer(self, df: pd.DataFrame, col: str) -> None:
        """
        """
        df[col.replace('出場レーサー', '出場レーサー名')] = df[col].apply(
            lambda x: x.split('  ')[0])
        df[col.replace('出場レーサー', '出場レーサー階級')] = df[col].apply(
            lambda x: x.split('  ')[1])
        df = df.drop(columns=col)

        return df

class Resultlist(Data):
    """
    """
    def read_one_day(self, jcd: int, date: datetime.date) -> None:
        """
        """
        yyyymmdd = date.strftime('%Y%m%d')
        url = cnf.RESULTLIST_URL.format(jcd=jcd, yyyymmdd=yyyymmdd)
        self.read_html(url)
        if len(self.dfs) > 0:
            """
            df_return = self.dfs[0]
            df_return.columns = cnf.RESULTLIST_TABLE_COLS_RETURN
            df_return['レース場コード'] = jcd
            df_return['日付'] = yyyymmdd
            output_path = self.get_safix_output_path(cnf.RESULTLIST_PATH_RETURN, f'_{jcd}_{yyyymmdd}')
            self.write_csv(df_return, output_path, cnf.RESULTLIST_CSV_COLS_RETURN)
            """

            df_order = self.dfs[1]
            df_order.columns = cnf.RESULTLIST_TABLE_COLS_ORDER
            df_order = df_order.query('着順1 != "１着"').copy()
            df_order['レース場コード'] = jcd
            df_order['日付'] = yyyymmdd

            for i in range(6):
                df_order = self.trim_order(df_order, f'着順{i + 1}')
            output_path = self.get_safix_output_path(cnf.RESULTLIST_PATH_ORDER, f'_{jcd}_{yyyymmdd}')
            self.write_csv(df_order, output_path, cnf.RESULTLIST_CSV_COLS_ORDER)


    def trim_order(self, df: pd.DataFrame, col: str) -> None:
        """
        """
        df[col + '_艇番'] = df[col].str[0]
        df[col + '_レーサー'] = df[col].str[1:-1]
        df = df.drop(columns=col)

        return df


if __name__ == '__main__':
    main()
