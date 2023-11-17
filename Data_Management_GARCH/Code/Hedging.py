#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:44:12 2023

@author: timothee
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib as plt
import pickle

os.chdir('/Users/timothee/Documents/IESEG/4a/Master Thesis/Data_Management_GARCH')

df_op = pd.read_csv('ready_for_ANN.csv')
C0K_pred = pd.read_csv('C0K_pred.csv', index_col=0)
jacobian_df = pd.read_csv('jacobian_df.csv', index_col=0)


sys.path.append('/Users/timothee/Documents/IESEG/4a/Master Thesis/Data_Management_GARCH/Code')
from Hedging_func import denormalize, choose_hedging_period, terminal_value_ANN, terminal_value_BS, append_hedging_metrics, append_pricing_metrics, tracking_errors

#hedge_df = choose_hedging_period(df_op)

hedge_df = df_op.merge(C0K_pred, left_index=True, right_index=True).merge(jacobian_df, left_index=True, right_index=True)


hedge_df = denormalize(hedge_df)

columns_to_check = ['C0', 'C_0_hat', 'C0_BS_garch_vol', 'C0_BS_impl_volatility']
hedge_df.dropna(subset=columns_to_check, inplace=True)


hedge_df = terminal_value_ANN(hedge_df)#please repeat the command as much as you need
hedge_df = terminal_value_BS(hedge_df)#please repeat the command as much as you need

pricing_metrics = append_pricing_metrics(hedge_df)
hedging_metrics = append_hedging_metrics(hedge_df)
tracking_errors = tracking_errors(hedge_df)
hedge_df.to_csv('hedge_df.csv', index = False)
tracking_errors.to_csv('tracking_errors.csv', index = False)


with open('pricing_metrics.pkl', 'wb') as file:
    pickle.dump(pricing_metrics, file)

with open('hedging_metrics.pkl', 'wb') as file:
    pickle.dump(hedging_metrics, file)













