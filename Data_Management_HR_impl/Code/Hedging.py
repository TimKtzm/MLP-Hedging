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

os.chdir('/Users/timothee/Documents/IESEG/4a/Master Thesis/Data_Management_HR_impl')

df_op = pd.read_csv('ready_for_ANN_full.csv')
delta_pred = pd.read_csv('delta_pred.csv', index_col=0)



sys.path.append('/Users/timothee/Documents/IESEG/4a/Master Thesis/Data_Management_HR_impl/Code')
from Hedging_func import denormalize, choose_hedging_period, terminal_value_ANN, terminal_value_BS, append_hedging_metrics, tracking_errors

#hedge_df = choose_hedging_period(df_op)

hedge_df = df_op.merge(delta_pred, left_index=True, right_index=True)
hedge_df.rename(columns={'delta_pred': 'delta_nn'}, inplace=True)


hedge_df = terminal_value_ANN(hedge_df)#please repeat the command as much as you need
hedge_df = terminal_value_BS(hedge_df)#please repeat the command as much as you need

hedging_metrics = append_hedging_metrics(hedge_df)
tracking_errors = tracking_errors(hedge_df)
hedge_df.to_csv('hedge_df.csv', index = False)
tracking_errors.to_csv('tracking_errors.csv', index = False)

with open('hedging_metrics.pkl', 'wb') as file:
    pickle.dump(hedging_metrics, file)











