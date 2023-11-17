#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:56:42 2023

@author: timothee
"""

import pandas as pd
from scipy.interpolate import interp1d
import numpy as np




def interpolate(yield_curve, overnight_libor, df_op):
    prev_date = None
    new_rows = []

    for index, row in yield_curve.iterrows():
        if prev_date is not None and row['date'] != prev_date:
            prev_row = yield_curve.loc[index - 1]
            if prev_row['days'] != 1:
                new_row = {'date': prev_date, 'days': 1, 'rate': None}
                new_rows.append(new_row)
        prev_date = row['date']

    if new_rows:
        yield_curve = pd.concat([yield_curve, pd.DataFrame(new_rows)], ignore_index=True)

    nan_mask = yield_curve['rate'].isna()

    yield_curve.loc[nan_mask, 'rate'] = yield_curve.loc[nan_mask, 'date'].map(
        overnight_libor.set_index('date')['ON']
    )

    yield_curve['rate'] = yield_curve.apply(
        lambda row: overnight_libor.loc[overnight_libor['date'] == row['date'], 'ON'].values[0]
        if pd.isna(row['rate']) and (row['date'] in overnight_libor['date'].values)
        else row['rate'],
        axis=1,
    )

    yield_curve = yield_curve.sort_values(['date', 'days'], ascending=[True, True])
    rates_curve = yield_curve.reset_index(drop=True)

    nan_count = rates_curve['rate'].isna().sum()
    print("Number of NaN values in 'rate' column:", nan_count)

    # Rate Interpolation
    rates_curve = rates_curve.sort_values(['days', 'date'])

    interp_func = rates_curve.groupby('date').apply(lambda group: interp1d(group['days'], group['rate'], kind='linear'))

    # Apply the interpolation function based on 'date' to the appropriate column in df_op
    def interpolate_row(row):
        date = row['date']
        if date in interp_func:
            tte = row['Time_to_expiry']
            if tte <= min(interp_func[date].x):
                return interp_func[date](min(interp_func[date].x)) / 100  
            elif tte >= max(interp_func[date].x):
                return interp_func[date](max(interp_func[date].x)) / 100  
            else:
                return interp_func[date](tte) / 100
        else:
            return None  

    df_op['Rf'] = df_op.apply(interpolate_row, axis=1)

    # We need to annualize the TTE after the interpolation function since it cannot handle extremely low values
    df_op['Time_to_expiry'] = df_op['Time_to_expiry'] / 252

    return df_op






#Append Short term rate 
def append_short_term_rate(df_op, overnight_libor):
    overnight_libor['ON']/=100
    df_op = df_op.merge(overnight_libor[['date', 'ON']], on='date', how='left')
    return df_op
    


#Append Garch_volatility

def app_garch_volatility(garch_vol, df_op):
  garch_vol['date'] = pd.to_datetime(garch_vol['date'])
  df_op = df_op.merge(garch_vol[['date', 'garch_vol']], on='date', how='left')
  return df_op

def app_histo_vol(histo_vol, df_op):
   histo_vol['date'] = pd.to_datetime(histo_vol['date'])
   histo_vol.rename(columns = {'days': 'Time_to_expiry'}, inplace=True)
   histo_vol.rename(columns = {'volatility': 'histo_vol'}, inplace=True)
   histo_vol.sort_values(['Time_to_expiry', 'date'], inplace=True)
   df_op.sort_values(['Time_to_expiry', 'date'], inplace=True)
   df_op = pd.merge_asof(df_op, histo_vol[['date', 'Time_to_expiry', 'histo_vol']], on='Time_to_expiry', by='date', direction='nearest')
 
 
   return df_op






#This snippet is inspired from Ruf & Wang (2020)
def end_of_hedging_p_data(
        df,
        shift_day,
        shift_key='_1D',
        next_volume=None
    ):

    list = ['date', 'optionid', 'S0', 'C0', 'impl_volatility']
    new_cols = {
        'S0': f'S{shift_key}', 
        'C0': f'V{shift_key}', 
        'impl_volatility': f'implvol{shift_key}'}
    if 'Rf' in df.columns:
        list += ['Rf']
        new_cols['Rf'] = f'Rf{shift_key}'

    if next_volume:
        list += ['volume']
        new_cols['volume'] = f'volume{shift_key}'
        
    df_list = df[list].copy()
    df_list['date'] -= shift_day

    df_list.rename(columns=new_cols, inplace=True) 

    df = df.join(df_list.set_index(['date', 'optionid']), on=['date', 'optionid'])
    return df


def cleaning(df_op):
    df_op['Rf'] = pd.to_numeric(df_op['Rf'], errors='coerce')
    df_op = df_op[df_op['volume'] >= 1]
    df_op = df_op[df_op['Time_to_expiry'] > (1/252)]
    df_op = df_op[(df_op['impl_volatility'] >= 0.01) & (df_op['impl_volatility'] <= 1)]
    df_op = df_op[df_op['best_offer'] <= 2 * df_op['best_bid']]
    df_op = df_op[df_op['best_bid'] >= 0.05]
    mk = (df_op['cp_flag'] == 'C') & (df_op['S0'] - np.exp(-df_op['Rf'] * df_op['Time_to_expiry']) * df_op['K'] >= df_op['C0'])#negative time value
    df_op = df_op.loc[~mk]
    mk = (df_op['cp_flag'] == 'P') & (np.exp(-df_op['Rf'] * df_op['Time_to_expiry']) * df_op['K'] - df_op['S0'] >= df_op['C0'])
    df_op = df_op.loc[~mk]
    mk =  (df_op['cp_flag'] == 'C') & (df_op['M'] > 2.0)
    df_op = df_op[~mk]
    mk =  (df_op['cp_flag'] == 'P') & (df_op['M'] < 0.5)
    df_op = df_op[~mk]
    
    
    df_op['Code_dataissue'] = 0

    bl = (df_op['date'] != df_op['last_date'])
    df_op.loc[bl, 'Code_dataissue'] = 1
    print('Warning: Number of samples with last_trade_date issues: {}'.format(bl.sum()))
    df_op = df_op[~bl]
    
    
    bl_c = df_op['cp_flag'] == 'C'
    bl_p = df_op['cp_flag'] == 'P'
    df_op.loc[bl_c, 'cp_int'] = 0
    df_op.loc[bl_p, 'cp_int'] = 1
    return df_op

def normalize_prices(df):
    df['C0/K'] = df['C0']/df['K']
    df['K/K'] = df['K']/df['K']
    return df







