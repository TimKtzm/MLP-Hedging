#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 10:59:38 2023

@author: timothee
"""

from tabulate import tabulate
import os
import pandas as pd
import pickle
import numpy as np
import scipy

os.chdir('/Users/timothee/Documents/IESEG/4a/Master Thesis/Data_Management')

hedge_df = pd.read_csv('hedge_df.csv')

AA = (hedge_df['Time_to_expiry'] < 0.02) & (((hedge_df['cp_flag'] == 'P') & (
    hedge_df['M'] < 1)) | ((hedge_df['cp_flag'] == 'C') & (hedge_df['M'] > 1)))
AB = ((hedge_df['Time_to_expiry'] >= 0.02) & (hedge_df['Time_to_expiry'] < 0.06)) & ((((hedge_df['cp_flag'] == 'P') & (
    hedge_df['M'] < 1)) | ((hedge_df['cp_flag'] == 'C') & (hedge_df['M'] > 1))))
AC = (hedge_df['Time_to_expiry'] >= 0.06) & (((hedge_df['cp_flag'] == 'P') & (
    hedge_df['M'] < 1)) | ((hedge_df['cp_flag'] == 'C') & (hedge_df['M'] > 1)))

BA = (hedge_df['Time_to_expiry'] < 0.02) & (((hedge_df['cp_flag'] == 'P') & (hedge_df['M'] >= 0.8) & (
    hedge_df['M'] <= 1.2)) | ((hedge_df['cp_flag'] == 'C') & (hedge_df['M'] >= 0.8) & (hedge_df['M'] <= 1.2)))
BB = ((hedge_df['Time_to_expiry'] >= 0.02) & (hedge_df['Time_to_expiry'] < 0.06)) & (((hedge_df['cp_flag'] == 'P') & (hedge_df['M'] >= 0.8) & (
    hedge_df['M'] <= 1.2)) | ((hedge_df['cp_flag'] == 'C') & (hedge_df['M'] >= 0.8) & (hedge_df['M'] <= 1.2)))
BC = (hedge_df['Time_to_expiry'] >= 0.06) & (((hedge_df['cp_flag'] == 'P') & (hedge_df['M'] >= 0.8) & (
    hedge_df['M'] <= 1.2)) | ((hedge_df['cp_flag'] == 'C') & (hedge_df['M'] >= 0.8) & (hedge_df['M'] <= 1.2)))

CA = (hedge_df['Time_to_expiry'] < 0.02) & (((hedge_df['cp_flag'] == 'P') & (
    hedge_df['M'] > 1)) | ((hedge_df['cp_flag'] == 'C') & (hedge_df['M'] < 1)))
CB = ((hedge_df['Time_to_expiry'] >= 0.02) & (hedge_df['Time_to_expiry'] < 0.06)) & ((((hedge_df['cp_flag'] == 'P') & (
    hedge_df['M'] > 1)) | ((hedge_df['cp_flag'] == 'C') & (hedge_df['M'] < 1))))
CC = (hedge_df['Time_to_expiry'] >= 0.06) & (((hedge_df['cp_flag'] == 'P') & (
    hedge_df['M'] > 1)) | ((hedge_df['cp_flag'] == 'C') & (hedge_df['M'] < 1)))



y_ANN = hedge_df['C_0_hat']
y_BS = hedge_df['C0_BS_garch_vol']
y = hedge_df['C0']

conditions = [AA, AB, AC, BA, BB, BC, CA, CB, CC]  

results = {}  

for condition_label, condition in zip(["AA", "AB", "AC", "BA", "BB", "BC", "CA", "CB", "CC"], conditions):
    y_ANN_condition = y_ANN[condition]
    y_BS_condition = y_BS[condition]
    y_condition = y[condition]

    mae_ANN = np.mean(np.abs(y_condition - y_ANN_condition))
    mae_BS = np.mean(np.abs(y_condition - y_BS_condition))
    mse_ANN = np.mean((y_condition-y_ANN_condition)**2)
    mse_BS = np.mean((y_condition-y_BS_condition)**2)
    rmse_ANN = np.sqrt(np.mean((y_condition-y_ANN_condition)**2))
    rmse_BS = np.sqrt(np.mean((y_condition-y_BS_condition)**2))
    
    results[condition_label + "1"] = mae_ANN
    results[condition_label + "2"] = mae_BS
    results[condition_label + "3"] = mse_ANN
    results[condition_label + "4"] = mse_BS
    results[condition_label + "5"] = rmse_ANN
    results[condition_label + "6"] = rmse_BS


for condition, mae in results.items():
    print(f"{condition} = np.round(results['{condition}'], 3)")

N1 = np.round(np.mean(abs(y - y_ANN)), 3)
N2 = np.round(np.mean(abs(y - y_BS)), 3)
N3 = np.round(np.mean((y - y_ANN)**2), 3)
N4 = np.round(np.mean((y - y_BS)**2), 3)
N5 = np.round(np.sqrt(np.mean((y - y_ANN)**2)) ,3)
N6 = np.round(np.sqrt(np.mean((y - y_BS)**2)) ,3)

AA1 = np.round(results['AA1'], 3)
AA2 = np.round(results['AA2'], 3)
AA3 = np.round(results['AA3'], 3)
AA4 = np.round(results['AA4'], 3)
AA5 = np.round(results['AA5'], 3)
AA6 = np.round(results['AA6'], 3)
AB1 = np.round(results['AB1'], 3)
AB2 = np.round(results['AB2'], 3)
AB3 = np.round(results['AB3'], 3)
AB4 = np.round(results['AB4'], 3)
AB5 = np.round(results['AB5'], 3)
AB6 = np.round(results['AB6'], 3)
AC1 = np.round(results['AC1'], 3)
AC2 = np.round(results['AC2'], 3)
AC3 = np.round(results['AC3'], 3)
AC4 = np.round(results['AC4'], 3)
AC5 = np.round(results['AC5'], 3)
AC6 = np.round(results['AC6'], 3)
BA1 = np.round(results['BA1'], 3)
BA2 = np.round(results['BA2'], 3)
BA3 = np.round(results['BA3'], 3)
BA4 = np.round(results['BA4'], 3)
BA5 = np.round(results['BA5'], 3)
BA6 = np.round(results['BA6'], 3)
BB1 = np.round(results['BB1'], 3)
BB2 = np.round(results['BB2'], 3)
BB3 = np.round(results['BB3'], 3)
BB4 = np.round(results['BB4'], 3)
BB5 = np.round(results['BB5'], 3)
BB6 = np.round(results['BB6'], 3)
BC1 = np.round(results['BC1'], 3)
BC2 = np.round(results['BC2'], 3)
BC3 = np.round(results['BC3'], 3)
BC4 = np.round(results['BC4'], 3)
BC5 = np.round(results['BC5'], 3)
BC6 = np.round(results['BC6'], 3)
CA1 = np.round(results['CA1'], 3)
CA2 = np.round(results['CA2'], 3)
CA3 = np.round(results['CA3'], 3)
CA4 = np.round(results['CA4'], 3)
CA5 = np.round(results['CA5'], 3)
CA6 = np.round(results['CA6'], 3)
CB1 = np.round(results['CB1'], 3)
CB2 = np.round(results['CB2'], 3)
CB3 = np.round(results['CB3'], 3)
CB4 = np.round(results['CB4'], 3)
CB5 = np.round(results['CB5'], 3)
CB6 = np.round(results['CB6'], 3)
CC1 = np.round(results['CC1'], 3)
CC2 = np.round(results['CC2'], 3)
CC3 = np.round(results['CC3'], 3)
CC4 = np.round(results['CC4'], 3)
CC5 = np.round(results['CC5'], 3)
CC6 = np.round(results['CC6'], 3)    

AA7 = sum(AA)
AB7 = sum(AB)
AC7 = sum(AC)
BA7 = sum(BA)
BB7 = sum(BB)
BC7 = sum(BC)
CA7 = sum(CA)
CB7 = sum(CB)
CC7 = sum(CC)
Total = AA7 + AB7 + AC7 + BA7 + BB7 + BC7 + CA7 + CB7 + CC7
N7 = len(y)

rows = [
    ['Moneyness','Expiry', "", 'MAE', "", 'MSE', "", 'RMSE','Observations'],
    ["", "",'ANN', 'BS', 'ANN', 'BS', 'ANN', 'BS',Total],
    ['All','All',N1,N2,N3,N4,N5,N6, N7],   
    ['', 'Short', AA1, AA2, AA3, AA4, AA5, AA6, AA7],
    ['ITM', 'Medium', AB1, AB2, AB3, AB4, AB5, AB6, AB7],
    ['', 'Long', AC1, AC2, AC3, AC4, AC5, AC6,AC7],

    ["", "",'ANN', 'BS', 'ANN', 'BS', 'ANN', 'BS',''],
    ['', 'Short', BA1, BA2, BA3, BA4, BA5, BA6, BA7],
    ['NTM', 'Medium', BB1, BB2, BB3, BB4, BB5, BB6, BB7],
    ['', 'Long', BC1, BC2, BC3, BC4, BC5, BC6, BC7],

    ["", "",'ANN', 'BS', 'ANN', 'BS', 'ANN', 'BS',''],
    ['', 'Short', CA1, CA2, CA3, CA4, CA5, CA6, CA7],
    ['OTM', 'Medium', CB1, CB2, CB3, CB4, CB5, CB6, CB7],
    ['', 'Long', CC1, CC2, CC3, CC4, CC5, CC6,CC7],


]


table_pricing = tabulate(rows, headers='firstrow', tablefmt="latex")
    
