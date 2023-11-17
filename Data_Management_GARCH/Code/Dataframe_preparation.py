#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 18:14:45 2023

@author: timothee
"""
import sys
import os
import datetime
import numpy as np
import pandas as pd

sys.path.append('/Users/timothee/Documents/IESEG/4a/Master Thesis/Data_Management/Code')

from Dataframe_preparation_func import interpolate 
from BS_func import calculate_implied_volatility
from BS_func import compute_and_append_black_scholes_columns
from Dataframe_preparation_func import app_garch_volatility
from Dataframe_preparation_func import app_histo_vol
from Dataframe_preparation_func import cleaning
from initset import OFFSET_DICT
from Dataframe_preparation_func import end_of_hedging_p_data
from BS_func import compute_and_append_greeks
from Dataframe_preparation_func import append_short_term_rate
from Dataframe_preparation_func import normalize_prices


os.chdir('/Users/timothee/Documents/IESEG/4a/Master Thesis/Data_Management')

df_op = pd.read_csv('option_file_full.csv')
yield_curve = pd.read_csv('yield_curve.csv')
overnight_libor = pd.read_csv('Overnight_Libor.csv')
df_op['date'] = pd.to_datetime(df_op['date'], errors='coerce')
yield_curve['date'] = pd.to_datetime(yield_curve['date'], errors='coerce')
overnight_libor['date'] = pd.to_datetime(overnight_libor['date'], errors='coerce')
histo_vol = pd.read_csv('SPXOptions_Historical_Vol.csv')
garch_vol = pd.read_csv('garch_vol.csv')


cols_to_drop = ['secid', 'forward_price', 'open_interest', 'cfadj', 'contract_size', 'ss_flag', 'expiry_indicator', 'ticker', 'sic', 'index_flag', 'issuer','exercise_style','symbol', 'symbol_flag', 'exdate','am_settlement', 'root', 'suffix', 'cusip', 'issue_type', 'industry_group', 'div_convention', 'exchange_d', 'am_set_flag']
df_op = df_op.drop(columns=cols_to_drop)

#Append Historical Volatility
df_op = app_histo_vol(histo_vol, df_op)
 
#Rate Interpolation
interpolate(yield_curve, overnight_libor, df_op)


#We also need the short term rate for discounting
df_op = append_short_term_rate(df_op, overnight_libor)

#Append GARCH volatility
df_op = app_garch_volatility(garch_vol, df_op)

#Compute implied volbis
calculate_implied_volatility(df_op)
#Add BS
compute_and_append_black_scholes_columns(df_op, 'impl_volatility')
compute_and_append_black_scholes_columns(df_op, 'garch_vol')
compute_and_append_black_scholes_columns(df_op, 'histo_vol')


#Append end-of-hedging information
for key, value in OFFSET_DICT.items():
        df_op = end_of_hedging_p_data(
        df_op, 
        shift_day=value[0], shift_key=value[1], next_volume=True
    )



#Append greeks
compute_and_append_greeks(df_op, vol_column='garch_vol')
compute_and_append_greeks(df_op, vol_column='impl_volatility')
#Normalize prices
normalize_prices(df_op)

#Clean the dataframe 
df_op = cleaning(df_op)



#set the index
df_op.sort_values(by='date', ascending=True, inplace=True)
df_op.reset_index(drop=True, inplace=True)
#Export the dataframe
df_op.to_csv('ready_for_ANN.csv', index=False)











