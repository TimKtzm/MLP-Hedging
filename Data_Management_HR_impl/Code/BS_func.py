#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:07:05 2023

@author: timothee
"""

import pandas as pd
import numpy as np
import scipy
from scipy.optimize import root_scalar
import math

#Compute Black & Scholes formula
def BS (S0, K, sigma, Time_to_expiry, Rf, cp_flag):
    d1  = (np.log(S0/K)+(Rf+1/2*sigma**2)*Time_to_expiry)/(sigma*np.sqrt(Time_to_expiry))
    d2 = (np.log(S0/K)+(Rf-1/2*sigma**2)*Time_to_expiry)/(sigma*np.sqrt(Time_to_expiry))
    
    if cp_flag == "C":
     Call_price = S0*scipy.stats.norm.cdf(d1)-K*np.exp(-Rf*Time_to_expiry)*scipy.stats.norm.cdf(d2)
     return Call_price
    elif cp_flag == "P":
     Put_price = K*np.exp(-Rf*Time_to_expiry)*scipy.stats.norm.cdf(-d2)-S0*scipy.stats.norm.cdf(-d1)
     return Put_price
   



def compute_and_append_black_scholes_columns(df_op, *args):
    for arg in args:
        col_name = f'C0_BS_{arg}'
        df_op[col_name] = df_op.apply(
            lambda row: BS(row['S0'], row['K'], row[arg], row['Time_to_expiry'], row['Rf'], row['cp_flag']),
            axis=1
        )
    return df_op







#Calculate and append implied volatility

def calculate_implied_volatility(df_op):
    def implied_volatility(row):
        option_price = row['C0']
        S0 = row['S0']
        K = row['K']
        Time_to_expiry = row['Time_to_expiry']
        Rf = row['Rf']
        cp_flag = row['cp_flag']

        #Define a function that calculates the Black-Scholes option price
        def bs_option_price(sigma):
            d1 = (np.log(S0/K)+(Rf+0.5*sigma**2)*Time_to_expiry)/(sigma*np.sqrt(Time_to_expiry))
            d2 = d1 - sigma * np.sqrt(Time_to_expiry)

            if cp_flag == "C":
                return S0 * scipy.stats.norm.cdf(d1) - K * np.exp(-Rf * Time_to_expiry) * scipy.stats.norm.cdf(d2)
            elif cp_flag == "P":
                return K * np.exp(-Rf * Time_to_expiry) * scipy.stats.norm.cdf(-d2) - S0 * scipy.stats.norm.cdf(-d1)


        def price_difference(sigma):
            return bs_option_price(sigma) - option_price


        bracket = [0.001, 2.0]  

        try:
            #Uses a numerical root-finding method to solve for implied volatility
            result = root_scalar(price_difference, bracket=bracket)
            return result.root if result.converged else None
        except ValueError:
 
            return None

    # Create a mask for missing values in the 'impl_volatility' column
    missing_impl_volatility_mask = pd.isna(df_op['impl_volatility'])

    # Apply the implied_volatility function only to rows with missing values
    df_op.loc[missing_impl_volatility_mask, 'impl_volatility'] = df_op[missing_impl_volatility_mask].apply(implied_volatility, axis=1)

    return df_op




#Compute greeks

def calculate_d1(row, vol_column):
    S0 = row['S0']
    K = row['K']
    Time_to_expiry = row['Time_to_expiry']
    Rf = row['Rf']
    sigma = row[vol_column]

    d1 = (np.log(S0/K) + (Rf + 0.5 * sigma**2) * Time_to_expiry) / (sigma * np.sqrt(Time_to_expiry))
    return d1

def calculate_d2(row, vol_column):
    d1 = calculate_d1(row, vol_column)
    Time_to_expiry = row['Time_to_expiry']
    sigma = row[vol_column]
    d2 = d1 - sigma * np.sqrt(Time_to_expiry)
    return d2

def calculate_pdf_snd(row, vol_column):
    d1 = calculate_d1(row, vol_column)
    pdf_snd = (1 / (np.sqrt(2 * np.pi))) * np.exp(-(d1**2) / 2)
    return pdf_snd

def calculate_delta_BS(row, vol_column):
    d1 = calculate_d1(row, vol_column)
    cp_flag = row['cp_flag']
    
    if cp_flag == "C":
        delta_BS = scipy.stats.norm.cdf(d1)
    elif cp_flag == "P":
        delta_BS = scipy.stats.norm.cdf(d1) - 1
    
    return delta_BS

def calculate_gamma_BS(row, vol_column):
    S0 = row['S0']
    Time_to_expiry = row['Time_to_expiry']
    pdf_snd = calculate_pdf_snd(row, vol_column)
    sigma = row[vol_column]
    gamma_BS = pdf_snd / (S0 * sigma * np.sqrt(Time_to_expiry))
    return gamma_BS

def calculate_vega_BS(row, vol_column):
    S0 = row['S0']
    Time_to_expiry = row['Time_to_expiry']
    sigma = row[vol_column]
    pdf_snd = calculate_pdf_snd(row, vol_column)
    vega_BS = (S0 * np.sqrt(Time_to_expiry) * pdf_snd) / 100
    return vega_BS

def calculate_theta_BS(row, vol_column):
    S0 = row['S0']
    sigma = row[vol_column]
    Time_to_expiry = row['Time_to_expiry']
    Rf = row['Rf']
    K = row['K']
    pdf_snd = calculate_pdf_snd(row, vol_column)
    d2 = calculate_d2(row, vol_column)
    cp_flag = row['cp_flag']
    
    if cp_flag == "C":
       theta_BS = ((-S0 * pdf_snd * sigma) / (2 * np.sqrt(Time_to_expiry)) - Rf * K * np.exp(-Rf * Time_to_expiry) * scipy.stats.norm.cdf(d2))/252
       return theta_BS
    elif cp_flag == "P":
       theta_BS = ((-S0 * pdf_snd * sigma) / (2 * np.sqrt(Time_to_expiry)) + Rf * K * np.exp(-Rf * Time_to_expiry) * scipy.stats.norm.cdf(-d2))/252
       return theta_BS

def calculate_rho_BS(row, vol_column):
    K = row['K']
    Time_to_expiry = row['Time_to_expiry']
    Rf = row['Rf']
    d2 = calculate_d2(row, vol_column)
    cp_flag = row['cp_flag']    
    if cp_flag == "C":
       rho_BS = (K * Time_to_expiry * np.exp(-Rf*Time_to_expiry) * scipy.stats.norm.cdf(d2))/100
       return rho_BS
    elif cp_flag == "P":
         rho_BS = (-K * Time_to_expiry * np.exp(-Rf * Time_to_expiry) * scipy.stats.norm.cdf(-d2))/100
         return rho_BS
    

def compute_and_append_greeks(df, vol_column):
    deltacol = f'delta_BS_{vol_column}'
    gammacol = f'gamma_BS_{vol_column}'
    vegacol = f'vega_BS_{vol_column}'
    thetacol = f'theta_BS_{vol_column}'
    rhocol = f'rho_BS_{vol_column}'
    
    
    df[deltacol] = df.apply(calculate_delta_BS, vol_column=vol_column, axis=1)
    df[gammacol] = df.apply(calculate_gamma_BS, vol_column=vol_column, axis=1)
    df[vegacol] = df.apply(calculate_vega_BS, vol_column=vol_column, axis=1)
    df[thetacol] = df.apply(calculate_theta_BS, vol_column=vol_column, axis=1)
    df[rhocol] = df.apply(calculate_rho_BS, vol_column=vol_column, axis=1)
    return df