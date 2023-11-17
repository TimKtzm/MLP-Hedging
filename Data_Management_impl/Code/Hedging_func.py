#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 14:33:37 2023

@author: timothee
"""

import sys
import os
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import scipy


def denormalize(df):
   df['C_0_hat'] = df['C0K_hat'] * df['K']
   return df



def choose_hedging_period(df):
    hedging_periods = ['1D', '2D', '5D']
    while True:
        print("Available hedging periods:", ", ".join(hedging_periods))
        selected_period = input("Enter the hedging period (e.g., 1D, 2D, 5D): ").strip()
        if selected_period in hedging_periods:
            break
        else:
            print("Invalid hedging period. Please choose from the available options.")
    column_name = f'V_{selected_period}'
    hedge_df = df.dropna(subset=[column_name])
    return hedge_df



def append_pricing_metrics(df):
        y_ANN = df['C_0_hat']
        y_BS = df['C0_BS_impl_volatility']
        y = df['C0']
   
        stats = dict()
        diff_ANN = y - y_ANN
        diff_BS = y - y_BS

        stats['mse_ANN'] = np.mean(diff_ANN**2)
        stats['mse_BS'] = np.mean(diff_BS**2)
  
        stats['rmse_ANN'] = np.sqrt(stats['mse_ANN'])
        stats['rmse_BS'] = np.sqrt(stats['mse_BS'])

        stats['mae_ANN'] = np.mean(abs(diff_ANN))
        stats['mae_BS'] = np.mean(abs(diff_BS))

        return stats





def terminal_value_ANN(df):
    hedging_periods = ['1D', '2D', '5D']
    while True:
        print("Available hedging periods:", ", ".join(hedging_periods))
        selected_period = input("Enter the hedging period (e.g., 1D, 2D, 5D): ").strip()
        if selected_period in hedging_periods:
            break
        else:
            print("Invalid hedging period. Please choose from the available options.")    
    
    def hedging_position(row, selected_period):
        cp_flag = row['cp_flag']
        delta_nn = row['delta_nn']
        S_t = f'S_{selected_period}'
        if cp_flag == "C":
            return delta_nn * row[S_t]
        elif cp_flag == "P":
            return delta_nn * row[S_t]
    

    
    def bond_position(row, selected_period):
        cp_flag = row['cp_flag']
        onr = row['ON']
        C0 = row['C0']
        S0 = row['S0']
        delta_nn = row['delta_nn']
        
        if selected_period == '1D':
            tau = 1/252
        elif selected_period == '2D':
            tau = 2/252
        elif selected_period == '5D':
            tau = 5/252
        
        if cp_flag == "C":
            return np.exp(onr * tau) * (C0 - delta_nn * S0)
        elif cp_flag == "P":
            return np.exp(onr * tau) * (C0 - delta_nn * S0)
    

    
    def option_position(row, selected_period):
        V_t = row[f'V_{selected_period}']
        V_C_t = -V_t
        return V_C_t
    

    
    df[f'V_P_{selected_period}'] = df.apply(
        lambda row: hedging_position(row, selected_period) + bond_position(row, selected_period) + option_position(row, selected_period),
        axis=1
    )
    
    return df














def terminal_value_BS(df):
    hedging_periods = ['1D', '2D', '5D']
    while True:
        print("Available hedging periods:", ", ".join(hedging_periods))
        selected_period = input("Enter the hedging period (e.g., 1D, 2D, 5D): ").strip()
        if selected_period in hedging_periods:
            break
        else:
            print("Invalid hedging period. Please choose from the available options.")    
    
    def hedging_position(row, selected_period):
        cp_flag = row['cp_flag']
        delta_BS_impl_volatility = row['delta_BS_impl_volatility']
        S_t = f'S_{selected_period}'
        if cp_flag == "C":
            return delta_BS_impl_volatility * row[S_t]
        elif cp_flag == "P":
            return delta_BS_impl_volatility * row[S_t]
    

    
    def bond_position(row, selected_period):
        cp_flag = row['cp_flag']
        onr = row['ON']
        C0 = row['C0']
        S0 = row['S0']
        delta_BS_impl_volatility = row['delta_BS_impl_volatility']
        
        if selected_period == '1D':
            tau = 1/252
        elif selected_period == '2D':
            tau = 2/252
        elif selected_period == '5D':
            tau = 5/252
        
        if cp_flag == "C":
            return np.exp(onr * tau) * (C0 - delta_BS_impl_volatility * S0)
        elif cp_flag == "P":
            return np.exp(onr * tau) * (C0 - delta_BS_impl_volatility * S0)
    

    
    def option_position(row, selected_period):
        V_t = row[f'V_{selected_period}']
        V_C_t = -V_t
        return V_C_t
    

       
    df[f'V_P_{selected_period}_BS'] = df.apply(
        lambda row: hedging_position(row, selected_period) + bond_position(row, selected_period) + option_position(row, selected_period),
        axis=1
    )
    
    return df



def join_hedging_metrics(df, pricing_metrics):
    stats = dict()
    
    onr = df['ON']
    tau1 = 1 / 252
    tau2 = 2 / 252
    tau5 = 5 / 252
    
    V_P_1D = df['V_P_1D']
    V_P_2D = df['V_P_2D']
    V_P_5D = df['V_P_5D']
    V_P_1D_BS = df['V_P_1D_BS']
    V_P_2D_BS = df['V_P_2D_BS']
    V_P_5D_BS = df['V_P_5D_BS']

    TE_1D = np.exp(-onr * tau1) * V_P_1D
    TE_2D = np.exp(-onr * tau2) * V_P_2D
    TE_5D = np.exp(-onr * tau5) * V_P_5D
    TE_1D_BS = np.exp(-onr * tau1) * V_P_1D_BS
    TE_2D_BS = np.exp(-onr * tau2) * V_P_2D_BS
    TE_5D_BS = np.exp(-onr * tau5) * V_P_5D_BS
    
    stats['MTE_1D'] = np.mean(TE_1D)
    stats['MTE_2D'] = np.mean(TE_2D)
    stats['MTE_5D'] = np.mean(TE_5D)
    stats['MTE_1D_BS'] = np.mean(TE_1D_BS)
    stats['MTE_2D_BS'] = np.mean(TE_2D_BS)
    stats['MTE_5D_BS'] = np.mean(TE_5D_BS)

    stats['MATE_1D'] = np.mean(np.abs(TE_1D))
    stats['MATE_2D'] = np.mean(np.abs(TE_2D))
    stats['MATE_5D'] = np.mean(np.abs(TE_5D))
    stats['MATE_1D_BS'] = np.mean(np.abs(TE_1D_BS))
    stats['MATE_2D_BS'] = np.mean(np.abs(TE_2D_BS))
    stats['MATE_5D_BS'] = np.mean(np.abs(TE_5D_BS))

    stats['PE_1D'] = np.sqrt(stats['MTE_1D']**2 + np.var(TE_1D))
    stats['PE_2D'] = np.sqrt(stats['MTE_2D']**2 + np.var(TE_2D))
    stats['PE_5D'] = np.sqrt(stats['MTE_5D']**2 + np.var(TE_5D))
    stats['PE_1D_BS'] = np.sqrt(stats['MTE_1D_BS']**2 + np.var(TE_1D_BS))
    stats['PE_2D_BS'] = np.sqrt(stats['MTE_2D_BS']**2 + np.var(TE_2D_BS))
    stats['PE_5D_BS'] = np.sqrt(stats['MTE_5D_BS']**2 + np.var(TE_5D_BS))
    
    stats.update(pricing_metrics)
    
    return stats



def append_hedging_metrics(df):
    stats = dict()
    
    onr = df['ON']
    tau1 = 1 / 252
    tau2 = 2 / 252
    tau5 = 5 / 252
    
    V_P_1D = df['V_P_1D']
    V_P_2D = df['V_P_2D']
    V_P_5D = df['V_P_5D']
    V_P_1D_BS = df['V_P_1D_BS']
    V_P_2D_BS = df['V_P_2D_BS']
    V_P_5D_BS = df['V_P_5D_BS']

    TE_1D = np.exp(-onr * tau1) * V_P_1D
    TE_2D = np.exp(-onr * tau2) * V_P_2D
    TE_5D = np.exp(-onr * tau5) * V_P_5D
    TE_1D_BS = np.exp(-onr * tau1) * V_P_1D_BS
    TE_2D_BS = np.exp(-onr * tau2) * V_P_2D_BS
    TE_5D_BS = np.exp(-onr * tau5) * V_P_5D_BS
    


    ATE_1D = np.abs(TE_1D)
    ATE_2D = np.abs(TE_2D)
    ATE_5D = np.abs(TE_5D)
    ATE_1D_BS = np.abs(TE_1D_BS)
    ATE_2D_BS = np.abs(TE_2D_BS)
    ATE_5D_BS = np.abs(TE_5D_BS)
    
    
    
    stats['MTE_1D'] = np.mean(TE_1D)
    stats['MTE_2D'] = np.mean(TE_2D)
    stats['MTE_5D'] = np.mean(TE_5D)
    stats['MTE_1D_BS'] = np.mean(TE_1D_BS)
    stats['MTE_2D_BS'] = np.mean(TE_2D_BS)
    stats['MTE_5D_BS'] = np.mean(TE_5D_BS)

    stats['MATE_1D'] = np.mean(np.abs(TE_1D))
    stats['MATE_2D'] = np.mean(np.abs(TE_2D))
    stats['MATE_5D'] = np.mean(np.abs(TE_5D))
    stats['MATE_1D_BS'] = np.mean(np.abs(TE_1D_BS))
    stats['MATE_2D_BS'] = np.mean(np.abs(TE_2D_BS))
    stats['MATE_5D_BS'] = np.mean(np.abs(TE_5D_BS))

    stats['PE_1D'] = np.sqrt(stats['MTE_1D']**2 + np.var(TE_1D))
    stats['PE_2D'] = np.sqrt(stats['MTE_2D']**2 + np.var(TE_2D))
    stats['PE_5D'] = np.sqrt(stats['MTE_5D']**2 + np.var(TE_5D))
    stats['PE_1D_BS'] = np.sqrt(stats['MTE_1D_BS']**2 + np.var(TE_1D_BS))
    stats['PE_2D_BS'] = np.sqrt(stats['MTE_2D_BS']**2 + np.var(TE_2D_BS))
    stats['PE_5D_BS'] = np.sqrt(stats['MTE_5D_BS']**2 + np.var(TE_5D_BS))
    
    #standard deviation for each measure
    stats['TE_1D_std_dev'] = np.std(TE_1D)
    stats['TE_2D_std_dev'] = np.std(TE_2D)
    stats['TE_5D_std_dev'] = np.std(TE_5D)
    stats['TE_1D_BS_std_dev'] = np.std(TE_1D_BS)
    stats['TE_2D_BS_std_dev'] = np.std(TE_2D_BS)
    stats['TE_5D_BS_std_dev'] = np.std(TE_5D_BS)


    stats['ATE_1D_std_dev'] = np.std(ATE_1D)
    stats['ATE_2D_std_dev'] = np.std(ATE_2D)
    stats['ATE_5D_std_dev'] = np.std(ATE_5D)
    stats['ATE_1D_BS_std_dev'] = np.std(ATE_1D_BS)
    stats['ATE_2D_BS_std_dev'] = np.std(ATE_2D_BS)
    stats['ATE_5D_BS_std_dev'] = np.std(ATE_5D_BS)
    
    def calculate_confidence_interval(data, confidence=0.95):
         n = len(data)
         mean = np.mean(data)
         std_dev = np.std(data, ddof=1)  
         margin_of_error = std_dev * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
         lower_bound = mean - margin_of_error
         upper_bound = mean + margin_of_error
         return lower_bound, upper_bound

    
    confidence_interval_1D = calculate_confidence_interval(TE_1D)
    confidence_interval_2D = calculate_confidence_interval(TE_2D)
    confidence_interval_5D = calculate_confidence_interval(TE_5D)
    confidence_interval_1D_BS = calculate_confidence_interval(TE_1D_BS)
    confidence_interval_2D_BS = calculate_confidence_interval(TE_2D_BS)
    confidence_interval_5D_BS = calculate_confidence_interval(TE_5D_BS)
      
    
    stats['MTE_1D_ci'] = confidence_interval_1D
    stats['MTE_2D_ci'] = confidence_interval_2D
    stats['MTE_5D_ci'] = confidence_interval_5D
    stats['MTE_1D_BS_ci'] = confidence_interval_1D_BS
    stats['MTE_2D_BS_ci'] = confidence_interval_2D_BS
    stats['MTE_5D_BS_ci'] = confidence_interval_5D_BS    
    
    return stats





def tracking_errors(df):
    tracking_errors = pd.DataFrame()
    onr = df['ON']
    tau1 = 1 / 252
    tau2 = 2 / 252
    tau5 = 5 / 252
    
    V_P_1D = df['V_P_1D']
    V_P_2D = df['V_P_2D']
    V_P_5D = df['V_P_5D']
    V_P_1D_BS = df['V_P_1D_BS']
    V_P_2D_BS = df['V_P_2D_BS']
    V_P_5D_BS = df['V_P_5D_BS']

    tracking_errors['TE_1D'] = np.exp(-onr * tau1) * V_P_1D
    tracking_errors['TE_2D'] = np.exp(-onr * tau2) * V_P_2D
    tracking_errors['TE_5D'] = np.exp(-onr * tau5) * V_P_5D
    tracking_errors['TE_1D_BS'] = np.exp(-onr * tau1) * V_P_1D_BS
    tracking_errors['TE_2D_BS'] = np.exp(-onr * tau2) * V_P_2D_BS
    tracking_errors['TE_5D_BS'] = np.exp(-onr * tau5) * V_P_5D_BS
    
    tracking_errors['TTE'] = df['Time_to_expiry']
    tracking_errors['M'] = df['M']
    tracking_errors['cpflag'] = df['cp_flag']  
    tracking_errors['optionid'] = df['optionid']  
    
    return tracking_errors






    