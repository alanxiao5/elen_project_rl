from math import isnan

import pandas as pd
from alphagen.trade.base import StockPosition, StockStatus
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.strategy import TopKSwapNStrategy
from alphagen_qlib.utils import load_alpha_pool_by_path, load_recent_data
from backtest import QlibBacktest
from alphagen.data.expression import *


device = "cpu"
normalise = False
close = Feature(FeatureType.CLOSE)
target = Ref(close, -6) / Ref(close, -1) - 1
instruments = 'all'
data_train = StockData(instrument=instruments,
                       start_time='2010-01-02',
                       end_time='2019-12-31',
                       device = device)

data_valid = StockData(instrument=instruments,
                       start_time='2020-01-02',
                       end_time='2020-12-30',
                       device = device)

data_test = StockData(instrument=instruments,
                      start_time='2021-01-04',
                      end_time='2024-08-30',
                      device = device)

def generate_backtest_results(data, POOL_PATH):
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -6) / Ref(close, -1) - 1
    calculator = QLibStockDataCalculator(data=data, target=target, normalise = False)
    exprs, weights = load_alpha_pool_by_path(POOL_PATH)
    output_m10 = calculator.calc_pool_m10_ret(exprs, weights)
    output_ic = calculator.calc_pool_IC_ret(exprs, weights) 
    output_ric = calculator.calc_pool_rIC_ret(exprs, weights) 
    output_ir = calculator.calc_pool_IR_ret(exprs, weights)
    output_sor = calculator.calc_pool_SOR_ret(exprs, weights)
    output_q10 = calculator.calc_pool_q10_ret(exprs, weights)

    results = {'IC':output_ic,
               'Ranked IC':output_ric,
               'IR':output_ir,
               'Sortino':output_sor,
               'Q10':output_q10,
               'M10': output_m10}

    expr_ic = []
    for exp in exprs:
        expr_ic.append(calculator.calc_single_IC_ret(exp))
        
    table_alphas = pd.DataFrame(data={"alpha equation":exprs,"Weight":weights,"IC training":expr_ic})
    table_alphas.loc[len(table_alphas)] = ['Weighted combination','',output_ic]

    return results, table_alphas
    
#%%

POOL_PATH6= r"D:\results\q10\112640_steps_pool.json"
train_q10 = generate_backtest_results(data_train, POOL_PATH6)
valid_q10 = generate_backtest_results(data_valid, POOL_PATH6)

