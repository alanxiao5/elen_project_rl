from typing import List, Optional, Tuple
from torch import Tensor
import torch
from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.stock_data import StockData
import numpy as np
import pandas as pd

class QLibStockDataCalculator(AlphaCalculator):
    def __init__(self, data: StockData, target: Optional[Expression], normalise: True):
        self.data = data

        if target is None: # Combination-only mode
            self.target_value = None
        else:
            if normalise:
                self.target_value = normalize_by_day(target.evaluate(self.data))
            else:
                self.target_value = target.evaluate(self.data)

    def _calc_alpha(self, expr: Expression) -> Tensor:
        return normalize_by_day(expr.evaluate(self.data))

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()
    
    def _calc_IR(self, value1: Tensor, value2: Tensor) -> float:
        IC = batch_pearsonr(value1, value2)
        IR = (IC.mean() / (IC.std() + 1e-9)).item()
        return IR
    
    def _calc_SOR(self, value1: Tensor, value2: Tensor) -> float:
        IC = batch_pearsonr(value1, value2)
        mean_IC = IC.mean()
        down_stddev = (IC[IC<0]).std()
        sor = (mean_IC / (down_stddev  + 1e-9)).item()
        return sor
    
    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()

    def make_ensemble_alpha(self, exprs: List[Expression], weights: List[float]) -> Tensor:
        n = len(exprs)
        factors: List[Tensor] = [self._calc_alpha(exprs[i]) * weights[i] for i in range(n)]
        return sum(factors)  # type: ignore

    def calc_single_IC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.target_value)

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_rIC(value, self.target_value)

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.target_value), self._calc_rIC(value, self.target_value)

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        value1, value2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
        return self._calc_IC(value1, value2)

    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IC(ensemble_value, self.target_value)
    
    def calc_pool_IR_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IR(ensemble_value, self.target_value)
        
    def calc_pool_SOR_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_SOR(ensemble_value, self.target_value)   
        
    def calc_pool_q10_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._quantile10(ensemble_value, self.target_value, 10)      
        
    def calc_pool_m10_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return  self._mono10(ensemble_value, self.target_value, 10)     
        
    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_rIC(ensemble_value, self.target_value)

    def calc_pool_all_ret(self, exprs: List[Expression], weights: List[float]) -> Tuple[float, float]:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IC(ensemble_value, self.target_value), self._calc_rIC(ensemble_value, self.target_value)

    def _compute_quantile10_rets(self, value1: Tensor, value2: Tensor, 
                                 quantiles: int = 10):
        groups = np.array(range(quantiles)) + 1
        factor_quantiles = pd.DataFrame(value1.cpu().numpy()).rank(
            axis=1,method='first').dropna(axis=0, how='all').apply(
                pd.qcut, q=quantiles, labels = groups,axis=1)
        rets = pd.DataFrame(value2.cpu().numpy())
        return_series = {}
        for group in groups:
            returns_group = rets[factor_quantiles == group]
            #return_series[group] = (returns_group.mean(axis=1) / returns_group.count(axis=1)).mean() * annulization # scale holding to 1 ; equal weights
            returns_group = returns_group.mean(axis=1)
            return_series[group] = returns_group
        return return_series   
    
    def _quantile10(self, value1: Tensor, value2: Tensor, quantile: int = 10):
        res = self._compute_quantile10_rets(value1, value2, quantile)
        if res is None:
            return 0
        else:
            quantiles = res.keys()
            topq = res[max(quantiles)]
            bottomq = res[min(quantiles)]
            SR = (topq - bottomq).mean() / (topq - bottomq).std() 
            return SR

    def _compute_monotonicity(self, return_series):
        data = {}
        for group, values in return_series.items():
            data[group] = values.mean()/values.std()
        data = data.values()
        ranks = [sorted(data).index(x) + 1 for x in data]
        rank_differences = [ranks[i] - ranks[i-1] for i in range(1, len(ranks))]
        positive_differences = sum(1 for diff in rank_differences if diff > 0)
        negative_differences = sum(1 for diff in rank_differences if diff < 0)
        monotonicity_score = abs(positive_differences - negative_differences) / (len(data)-1)
        return monotonicity_score
    
    def _mono10(self, value1: Tensor, value2: Tensor, quantile: int =10):
        res = self._compute_quantile10_rets(value1, value2, quantile)
        if res is None:
            return 0
        else:
            return self._compute_monotonicity(res)
            
