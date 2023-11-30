
from datetime import datetime
import calendar
import numpy as np
import pandas as pd


def preprocess_data_czech(df):
    #df = pd.read_csv('tr_by_acct_w_age.csv')

    czech_date_parser = lambda x: datetime.strptime(str(x), "%y%m%d")
    df["datetime"] = df["date"].apply(czech_date_parser)
    #df["datetime"] = pd.to_datetime(df["datetime"])

    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["dow"] =  df["datetime"].dt.dayofweek
    
    df["td"] = df[["account_id", "datetime"]].groupby("account_id").diff()
    df["td"] = df["td"].apply(lambda x: x.days)
    df["td"].fillna(0.0, inplace=True)
    

    # dtme - days till month end
    df["dtme"] = df.datetime.apply(lambda dt: calendar.monthrange(dt.year, dt.month)[1] - dt.day)

    df['raw_amount'] = df.apply(lambda row: row['amount'] if row['type'] == 'CREDIT' else -row['amount'], axis=1)


    cat_code_fields = ['type', 'operation', 'k_symbol']
    TCODE_SEP = "__"
    # create tcode by concating fields in "cat_code_fields"
    tcode = df[cat_code_fields[0]].astype(str)
    for ccf in cat_code_fields[1:]:
        tcode += TCODE_SEP + df[ccf].astype(str)

    df["tcode"] = tcode

    df["log_amount"] = np.log10(df["amount"]+1)
    LOG_AMOUNT_SCALE = df["log_amount"].std()
    df["log_amount_sc"] = df["log_amount"] / LOG_AMOUNT_SCALE
        
    TD_SCALE = df["td"].std()
    df["td_sc"] = df["td"] / TD_SCALE

    TCODE_TO_NUM = dict([(tc, i) for i, tc in enumerate(df['tcode'].unique())])
    NUM_TO_TCODE = dict([(i, tc) for i, tc in enumerate(df['tcode'].unique())])

    df['tcode' + "_num"] = df['tcode'].apply(lambda x: TCODE_TO_NUM[x])
    START_DATE = df["datetime"].min()
    return df, LOG_AMOUNT_SCALE, TD_SCALE, START_DATE, TCODE_TO_NUM, NUM_TO_TCODE 