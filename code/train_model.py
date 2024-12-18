# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:31:04 2024

@author: alanx
"""
import json
import os
from typing import Optional, Tuple
from datetime import datetime
import fire

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from alphagen.data.calculator import AlphaCalculator

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool, AlphaPoolBase
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.random import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.calculator import QLibStockDataCalculator

class CustomCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 show_freq: int,
                 save_path: str,
                 valid_calculator: AlphaCalculator,
                 test_calculator: AlphaCalculator,
                 name_prefix: str = 'rl_model',
                 timestamp: Optional[str] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        self.valid_calculator = valid_calculator
        self.test_calculator = test_calculator

        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        else:
            self.timestamp = timestamp

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        assert self.logger is not None
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        ic_test, rank_ic_test = self.pool.test_ensemble(self.test_calculator)
        self.logger.record('test/ic', ic_test)
        self.logger.record('test/rank_ic', rank_ic_test)
        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.name_prefix}_{self.timestamp}', f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_dict(), f)

    def show_pool_state(self):
        state = self.pool.state
        n = len(state['exprs'])
        print('---------------------------------------------')
        for i in range(n):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    @property
    def pool(self) -> AlphaPoolBase:
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore

#%%
def main(reward_shaping, pool_capacity: int =10, steps: int = 200000, seed: int = 5):
    instruments = 'all'
    reseed_everything(seed)
    
    #device = torch.device('cuda:0')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -6) / Ref(close, -1) - 1
    normalise = True
    #reward_shaping = 'IR'
    # You can re-implement AlphaCalculator instead of using QLibStockDataCalculator.
    
    data_train = StockData(instrument=instruments,
                           start_time='2010-01-02',
                           end_time='2019-12-31',
                           device = device)
    
    data_valid = StockData(instrument=instruments,
                           start_time='2019-12-31',
                           end_time='2021-12-30',
                           device = device)
    
    data_test = StockData(instrument=instruments,
                          start_time='2021-12-30',
                          end_time='2024-08-30',
                          device = device)
    
    calculator_train = QLibStockDataCalculator(data_train, target, normalise)
    calculator_valid = QLibStockDataCalculator(data_valid, target, normalise)
    calculator_test = QLibStockDataCalculator(data_test, target, normalise)
    
    pool = AlphaPool(
        capacity=pool_capacity,
        calculator=calculator_train,
        ic_lower_bound=None,
        l1_alpha=5e-3,
        device = device,
        reward_shaping = reward_shaping,
    )
    env = AlphaEnv(pool=pool, device=device, print_expr=True)
    
    name_prefix = f"new_{instruments}_{pool_capacity}_{seed}"
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    checkpoint_callback = CustomCallback(
        save_freq=1,
        show_freq=1,
        save_path=f'/content/drive/MyDrive/Colab Notebooks/elen_project/rw_{reward_shaping}/{name_prefix}/checkpoints',
        valid_calculator=calculator_valid,
        test_calculator=calculator_test,
        name_prefix=name_prefix,
        timestamp=timestamp,
        verbose=1,
    )
    #%%
    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=128,
        tensorboard_log=f'/content/drive/MyDrive/Colab Notebooks/elen_project/rw_{reward_shaping}/{name_prefix}/tb/log',
        device=device,
        verbose=1,
    )
    
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=f'{name_prefix}_{timestamp}',
        progress_bar = True,
    )
    
if __name__ == '__main__':
    main('IR',10,100, 5)