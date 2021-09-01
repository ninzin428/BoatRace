import datetime
import os

JCDS = {
    #1: '桐生',
    #2: '戸田',
    #3: '江戸川',
    #4: '平和島',
    #5: '多摩川',
    #6: '浜名湖',
    #7: '蒲郡',
    #8: '常滑',
    #9: '津',
    #10: '三国',
    #11: 'びわこ',
    #12: '住之江',
    13: '尼崎',
    #14: '鳴門',
    #15: '丸亀',
    #16: '児島',
    #17: '宮島',
    #18: '徳山',
    #19: '下関',
    #20: '若松',
    #21: '芦屋',
    #22: '福岡',
    #23: '唐津',
    #24: '大村',
}

START_DATE = datetime.datetime(2014, 4, 8)
START_DATE = datetime.datetime(2021, 9, 1)

WORK_DIR = '/Users/yn/Desktop/python/BoatRace'
OUTPUT_DIR = os.path.join(WORK_DIR, 'output')

# -----------------------------------------------------------------------------
RACEINDEX_URL = 'https://www.boatrace.jp/owpc/pc/race/raceindex?jcd={jcd}&hd={yyyymmdd}'
RACEINDEX_TABLE_COLS = [
    'レース',
    '締切予定時刻/投票',
    '締切予定時刻/投票.1',
    '出場レーサー1',
    '出場レーサー2',
    '出場レーサー3',
    '出場レーサー4',
    '出場レーサー5',
    '出場レーサー6',
    'レース別情報',
]
RACEINDEX_PATH = os.path.join(OUTPUT_DIR, 'RaceIndex', 'RaceIndex.csv')
RACEINDEX_PATH_CONCUT = os.path.join(OUTPUT_DIR, 'RaceIndex.csv')
RACEINDEX_CSV_COLS = [
    'レース場コード',
    '日付',
    'レース',
    '締切予定時刻/投票',
    '出場レーサー名1',
    '出場レーサー階級1',
    '出場レーサー名2',
    '出場レーサー階級2',
    '出場レーサー名3',
    '出場レーサー階級3',
    '出場レーサー名4',
    '出場レーサー階級4',
    '出場レーサー名5',
    '出場レーサー階級5',
    '出場レーサー名6',
    '出場レーサー階級6',
]

# -----------------------------------------------------------------------------
RESULTLIST_URL = 'https://www.boatrace.jp/owpc/pc/race/resultlist?jcd={jcd}&hd={yyyymmdd}'
RESULTLIST_TABLE_COLS_RETURN = [
    'レース',
    '3連勝単式組合わせ',
    '3連勝単式払戻金',
    '2連勝単式組合わせ',
    '2連勝単式払戻金',
    '備考',
]
RESULTLIST_PATH_RETURN = os.path.join(OUTPUT_DIR, 'RaceResultReturn', 'RaceResultReturn.csv')
RESULTLIST_PATH_RETURN_CONCUT = os.path.join(OUTPUT_DIR, 'RaceResultReturn.csv')
RESULTLIST_CSV_COLS_RETURN = [
    'レース場コード',
    '日付',
    'レース',
    '3連勝単式組合わせ',
    '3連勝単式払戻金',
    '2連勝単式組合わせ',
    '2連勝単式払戻金',
    '備考',
]
RESULTLIST_TABLE_COLS_ORDER = [
    'レース',
    '種別',
    '着順1',
    '着順2',
    '着順3',
    '着順4',
    '着順5',
    '着順6',
    '決まり手',
    '備考',
]
RESULTLIST_PATH_ORDER = os.path.join(OUTPUT_DIR, 'RaceResultOrder', 'RaceResultOrder.csv')
RESULTLIST_PATH_ORDER_CONCUT = os.path.join(OUTPUT_DIR, 'RaceResultOrder.csv')
RESULTLIST_CSV_COLS_ORDER = [
    'レース場コード',
    '日付',
    'レース',
    '種別',
    '着順1_艇番',
    '着順1_レーサー',
    '着順2_艇番',
    '着順2_レーサー',
    '着順3_艇番',
    '着順3_レーサー',
    '着順4_艇番',
    '着順4_レーサー',
    '着順5_艇番',
    '着順5_レーサー',
    '着順6_艇番',
    '着順6_レーサー',
    '決まり手',
    '備考',
]

