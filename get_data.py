"""WEBからデータを取得する
"""
import datetime
import os
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

import config as cnf


def main():
    for jcd in tqdm(cnf.JCDS.keys(), desc=f'Downloading Data.'):
        dl = Download(jcd)
        dl.download()

class Download:
    """
    """
    def __init__(self, jcd):
        """
        """
        self.jcd = jcd
        self.resultlist = Resultlist()
        self.racelist = Racelist()

        self.dates = []

    def download(self):
        """
        """
        self.create_dates()
        with Pool() as pool:
            imap = pool.imap(self.read_one_day, self.dates)
            ret = list(tqdm(imap, desc=f'read one day. jcd: {self.jcd}', total=len(self.dates)))
            

    def create_dates(self):
        """
        """
        date = cnf.START_DATE
        today = datetime.datetime.now()
        term = (today - date).days + 2
        for i in range(term):
            self.dates.append(date)
            date = date + datetime.timedelta(days=1)

    def read_one_day(self, date):
        """
        """
        dfs_len = self.racelist.read_one_day(jcd=self.jcd, date=date)
        if dfs_len > 0:
            self.resultlist.read_one_day(jcd=self.jcd, date=date)

def check():
    """
    """
    url = 'https://www.boatrace.jp/owpc/pc/race/racelist?rno=1&jcd=12&hd=20210922'
    dfs = pd.read_html(url)
    print(f'データ数: {len(dfs)}')

    for i in range(len(dfs)):
        df = dfs[i]
        path = os.path.join(cnf.OUTPUT_DIR, f'df{i}.csv')
        df.to_csv(path)
        print(f'データ{i}')
        print(df.head(1))
        print(df.columns)


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
    def read_one_day(self, jcd: int, date: datetime.date) -> int:
        """
        """
        yyyymmdd = date.strftime('%Y%m%d')
        url = cnf.RACEINDEX_URL.format(jcd=jcd, yyyymmdd=yyyymmdd)
        self.read_html(url)
        dfs_len = len(self.dfs)
        if dfs_len > 0:
            df = self.dfs[0]
            df.columns = cnf.RACEINDEX_TABLE_COLS

            for i in range(6):
                df = self.split_racer(df, f'出場レーサー{i + 1}')

            df['レース場コード'] = jcd
            df['日付'] = yyyymmdd

            output_path = self.get_safix_output_path(cnf.RACEINDEX_PATH, f'_{jcd}_{yyyymmdd}')
            self.write_csv(df, output_path, cnf.RACEINDEX_CSV_COLS)

        return dfs_len

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
        output_path = self.get_safix_output_path(cnf.RESULTLIST_PATH_RETURN, f'_{jcd}_{yyyymmdd}')
        if os.path.isfile(output_path):
            return 1

        url = cnf.RESULTLIST_URL.format(jcd=jcd, yyyymmdd=yyyymmdd)
        self.read_html(url)
        if len(self.dfs) > 0:
            df_return = self.dfs[0]
            df_return.columns = cnf.RESULTLIST_TABLE_COLS_RETURN
            df_return['レース場コード'] = jcd
            df_return['日付'] = yyyymmdd
            self.write_csv(df_return, output_path, cnf.RESULTLIST_CSV_COLS_RETURN)

            df_order = self.dfs[1]
            df_order.columns = cnf.RESULTLIST_TABLE_COLS_ORDER
            df_order = df_order.query('着順1 != "１着"').copy()
            df_order['レース場コード'] = jcd
            df_order['日付'] = yyyymmdd

            for i in range(6):
                df_order = self.trim_order(df_order, f'着順{i + 1}')
            output_path_order = self.get_safix_output_path(cnf.RESULTLIST_PATH_ORDER, f'_{jcd}_{yyyymmdd}')
            self.write_csv(df_order, output_path_order, cnf.RESULTLIST_CSV_COLS_ORDER)


    def trim_order(self, df: pd.DataFrame, col: str) -> None:
        """
        """
        df[col + '_艇番'] = df[col].str[0]
        df[col + '_レーサー'] = df[col].str[1:-1]
        df = df.drop(columns=col)

        return df

class Racelist(Data):
    """
    """
    def read_one_day(self, jcd: int, date: datetime.date) -> None:
        """
        """
        yyyymmdd = date.strftime('%Y%m%d')
        count_all_df = 0
        for i in range(12):
            count_df = self.read_one_race(jcd, yyyymmdd, i + 1)
            count_all_df += count_df
            if count_df == 0:
                break

        return count_all_df


    def read_one_race(self, jcd: int, yyyymmdd: int, rno: int):
        """
        """
        output_path = self.get_safix_output_path(cnf.RACELIST_PATH, f'_{jcd}_{yyyymmdd}_{rno}')
        if os.path.isfile(output_path):
            return 1
        
        url = cnf.RACELIST_URL.format(jcd=jcd, yyyymmdd=yyyymmdd, rno=rno)
        self.read_html(url)
        if len(self.dfs) > 0:
            df = self.dfs[1]
            df.columns = cnf.RACELIST_TABLE_COLS
            df['レース場コード'] = jcd
            df['日付'] = yyyymmdd
            df['レース'] = rno
            df = df.reset_index()
            df = df.rename(columns={'index':'ROW'})
            df = self.split_racer1(df)
            df = self.split_racer2(df)
            df = self.split_racer3(df)
            df = self.split_racer4(df)
            df = self.split_racer5(df)
            df = self.split_racer6(df)
            self.write_csv(df, output_path, cnf.RACELIST_CSV_COLS)

        return len(self.dfs)

    def split_racer1(self, df: pd.DataFrame) -> None:
        """
        """
        col = '登録番号/級別氏名支部/出身地年齢/体重'
        df[col] = df[col].str.split('  ')
        df['登録番号/級別'] = df[col].str[0]
        df['氏名'] = df[col].str[1]
        df['支部/出身地'] = df[col].str[2]
        df['年齢/体重'] = df[col].str[3]

        df['登録番号/級別'] = df['登録番号/級別'].str.split(' / ')
        df['登録番号'] = df['登録番号/級別'].str[0]
        df['級別'] = df['登録番号/級別'].str[1]

        df['支部/出身地'] = df['支部/出身地'].str.split('/')
        df['支部'] = df['支部/出身地'].str[0]
        df['出身地'] = df['支部/出身地'].str[1]

        df['年齢/体重'] = df['年齢/体重'].str.split('/')
        df['年齢'] = df['年齢/体重'].str[0]
        df['体重'] = df['年齢/体重'].str[1]

        df['年齢'] = df['年齢'].str.replace('歳', '')
        df['体重'] = df['体重'].str.replace('kg', '')

        df = df.drop(columns=col)

        return df

    def split_racer2(self, df: pd.DataFrame) -> None:
        """
        """
        col = 'F数L数平均ST'
        df[col] = df[col].str.split('  ')
        df['F数'] = df[col].str[0]
        df['L数'] = df[col].str[1]
        df['平均ST'] = df[col].str[2]

        return df

    def split_racer3(self, df: pd.DataFrame) -> None:
        """
        """
        col = '全国_勝率2連率3連率'
        df[col] = df[col].str.split('  ')
        df['全国_勝率'] = df[col].str[0]
        df['全国_2連率'] = df[col].str[1]
        df['全国_3連率'] = df[col].str[2]

        return df

    def split_racer4(self, df: pd.DataFrame) -> None:
        """
        """
        col = '当地_勝率2連率3連率'
        df[col] = df[col].str.split('  ')
        df['当地_勝率'] = df[col].str[0]
        df['当地_2連率'] = df[col].str[1]
        df['当地_3連率'] = df[col].str[2]

        return df

    def split_racer5(self, df: pd.DataFrame) -> None:
        """
        """
        col = 'モーター_No2連率3連率'
        df[col] = df[col].str.split('  ')
        df['モーター_No'] = df[col].str[0]
        df['モーター_2連率'] = df[col].str[1]
        df['モーター_3連率'] = df[col].str[2]

        return df

    def split_racer6(self, df: pd.DataFrame) -> None:
        """
        """
        col = 'ボート_No2連率3連率'
        df[col] = df[col].str.split('  ')
        df['ボート_No'] = df[col].str[0]
        df['ボート_2連率'] = df[col].str[1]
        df['ボート_3連率'] = df[col].str[2]

        return df

if __name__ == '__main__':
    main()
    #check()
