# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 21:15:12 2024

@author: alanx
"""
import pandas as pd

def save_csv_in_folder():

    paths = ["C:/Users/alanx/elen project/stock_data1.txt",
     "C:/Users/alanx/elen project/stock_data2.txt",
     "C:/Users/alanx/elen project/stock_data3.txt"]
    
    df = pd.DataFrame()
    for pp in paths:
        data = pd.read_csv(pp, sep="\t")
        df = pd.concat([df,data],axis=1)
        
    desc = pd.read_csv(r"C:/Users/alanx/elen project/descriptive_data.txt", sep="\t")
    
    mapper = dict(zip(desc.security_id,desc.sedol))
    df.loc[:,'sedol'] = df.security_id.map(mapper)
    df.loc[:,'factor'] = 1 
    df = df.loc[df.weight.notna()]
    
    selected_fields = ['sedol','date','close','open','low','high','volume']
    df = df.loc[:,selected_fields]
    df = df.rename({'sedol':'symbol'},axis=1)
    symbols = set(df['symbol'])
    for sy in symbols:
        df_temp = df.loc[df['symbol'].isin([sy])]
        df_temp = df_temp.sort_values('date')
        df_temp.to_csv(fr"C:\Users\alanx\.qlib\csv_data\custom_us_data\{sy}.csv", index=False)

if __name__ == '__main__':
    save_csv_in_folder()
    #activate elenenv
    #cd elen project\qlib
    #python scripts/dump_bin.py dump_all --csv_path  ~/.qlib/csv_data/custom_us_data --qlib_dir ~/.qlib/qlib_data/custom_us_data --include_fields open,close,high,low,volume,factor