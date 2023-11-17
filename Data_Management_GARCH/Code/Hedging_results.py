#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:02:21 2023

@author: timothee
"""
from tabulate import tabulate
import os
import pandas as pd
import pickle
import numpy as np
import scipy

os.chdir('/Users/timothee/Documents/IESEG/4a/Master Thesis/Data_Management_GARCH')
tracking_errors = pd.read_csv('tracking_errors.csv')



with open('hedging_metrics.pkl', 'rb') as file:
    hedging_metrics = pickle.load(file)


column_pairs = [('TE_1D_BS', 'TE_1D'), ('TE_2D_BS',
                                        'TE_2D'), ('TE_5D_BS', 'TE_5D')]

T_test = {}

for col1, col2 in column_pairs:

    data1 = np.asarray(tracking_errors[col1])
    data2 = np.asarray(tracking_errors[col2])


    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]


    T_value = scipy.stats.ttest_rel(data1, data2)


    T_test[(col1, col2)] = T_value


for column_pair, t_value in T_test.items():
    col1, col2 = column_pair
    print(f"T-Test for {col1} and {col2}:")
    print(f"T-Statistic: {t_value[0]}")
    print(f"P-Value: {t_value[1]}")
    print()

T1 = np.round(T_test[('TE_1D_BS', 'TE_1D')][0],3)
T2 = np.round(T_test[('TE_2D_BS', 'TE_2D')][0],3)
T3 = np.round(T_test[('TE_5D_BS', 'TE_5D')][0],3)
P1 = np.round(T_test[('TE_1D_BS', 'TE_1D')][1],3)
P2 = np.round(T_test[('TE_2D_BS', 'TE_2D')][1],3)
P3 = np.round(T_test[('TE_5D_BS', 'TE_5D')][1],3)


data = [
        ['1 day', '', '2 days', '', '5 days', ''],
        ['T-stat', 'P-value', 'T-stat', 'P-value', 'T-stat', 'P-value'],
        [T1      ,    P1    ,    T2   ,    P2    ,    T3   ,    P3    ]
                        
]

table_Ttest = tabulate(data, headers='firstrow', tablefmt="latex")



# We define conditions for the table
AA = (tracking_errors['TTE'] < 0.02) & (((tracking_errors['cpflag'] == 'P') & (
    tracking_errors['M'] < 1)) | ((tracking_errors['cpflag'] == 'C') & (tracking_errors['M'] > 1)))
AB = ((tracking_errors['TTE'] >= 0.02) & (tracking_errors['TTE'] < 0.06)) & ((((tracking_errors['cpflag'] == 'P') & (
    tracking_errors['M'] < 1)) | ((tracking_errors['cpflag'] == 'C') & (tracking_errors['M'] > 1))))
AC = (tracking_errors['TTE'] >= 0.06) & (((tracking_errors['cpflag'] == 'P') & (
    tracking_errors['M'] < 1)) | ((tracking_errors['cpflag'] == 'C') & (tracking_errors['M'] > 1)))

BA = (tracking_errors['TTE'] < 0.02) & (((tracking_errors['cpflag'] == 'P') & (tracking_errors['M'] >= 0.8) & (
    tracking_errors['M'] <= 1.2)) | ((tracking_errors['cpflag'] == 'C') & (tracking_errors['M'] >= 0.8) & (tracking_errors['M'] <= 1.2)))
BB = ((tracking_errors['TTE'] >= 0.02) & (tracking_errors['TTE'] < 0.06)) & (((tracking_errors['cpflag'] == 'P') & (tracking_errors['M'] >= 0.8) & (
    tracking_errors['M'] <= 1.2)) | ((tracking_errors['cpflag'] == 'C') & (tracking_errors['M'] >= 0.8) & (tracking_errors['M'] <= 1.2)))
BC = (tracking_errors['TTE'] >= 0.06) & (((tracking_errors['cpflag'] == 'P') & (tracking_errors['M'] >= 0.8) & (
    tracking_errors['M'] <= 1.2)) | ((tracking_errors['cpflag'] == 'C') & (tracking_errors['M'] >= 0.8) & (tracking_errors['M'] <= 1.2)))

CA = (tracking_errors['TTE'] < 0.02) & (((tracking_errors['cpflag'] == 'P') & (
    tracking_errors['M'] > 1)) | ((tracking_errors['cpflag'] == 'C') & (tracking_errors['M'] < 1)))
CB = ((tracking_errors['TTE'] >= 0.02) & (tracking_errors['TTE'] < 0.06)) & ((((tracking_errors['cpflag'] == 'P') & (
    tracking_errors['M'] > 1)) | ((tracking_errors['cpflag'] == 'C') & (tracking_errors['M'] < 1))))
CC = (tracking_errors['TTE'] >= 0.06) & (((tracking_errors['cpflag'] == 'P') & (
    tracking_errors['M'] > 1)) | ((tracking_errors['cpflag'] == 'C') & (tracking_errors['M'] < 1)))

# We now calculate and assign hedging_metrics based on the conditions
# 1st the stdev
AA1 = np.round(np.std(tracking_errors.loc[AA, 'TE_1D']),3)
AA2 = np.round(np.std(tracking_errors.loc[AA, 'TE_1D_BS']),3)
AA3 = np.round(np.std(np.abs(tracking_errors.loc[AA, 'TE_1D'])),3)
AA4 = np.round(np.std(np.abs(tracking_errors.loc[AA, 'TE_1D_BS'])),3)
AB1 = np.round(np.std(tracking_errors.loc[AB, 'TE_1D']),3)
AB2 = np.round(np.std(tracking_errors.loc[AB, 'TE_1D_BS']),3)
AB3 = np.round(np.std(np.abs(tracking_errors.loc[AB, 'TE_1D'])),3)
AB4 = np.round(np.std(np.abs(tracking_errors.loc[AB, 'TE_1D_BS'])),3)
AC1 = np.round(np.std(tracking_errors.loc[AC, 'TE_1D']),3)
AC2 = np.round(np.std(tracking_errors.loc[AC, 'TE_1D_BS']),3)
AC3 = np.round(np.std(np.abs(tracking_errors.loc[AC, 'TE_1D'])),3)
AC4 = np.round(np.std(np.abs(tracking_errors.loc[AC, 'TE_1D_BS'])),3)

BA1 = np.round(np.std(tracking_errors.loc[BA, 'TE_1D']),3)
BA2 = np.round(np.std(tracking_errors.loc[BA, 'TE_1D_BS']),3)
BA3 = np.round(np.std(np.abs(tracking_errors.loc[BA, 'TE_1D'])),3)
BA4 = np.round(np.std(np.abs(tracking_errors.loc[BA, 'TE_1D_BS'])),3)
BB1 = np.round(np.std(tracking_errors.loc[BB, 'TE_1D']),3)
BB2 = np.round(np.std(tracking_errors.loc[BB, 'TE_1D_BS']),3)
BB3 = np.round(np.std(np.abs(tracking_errors.loc[BB, 'TE_1D'])),3)
BB4 = np.round(np.std(np.abs(tracking_errors.loc[BB, 'TE_1D_BS'])),3)
BC1 = np.round(np.std(tracking_errors.loc[BC, 'TE_1D']),3)
BC2 = np.round(np.std(tracking_errors.loc[BC, 'TE_1D_BS']),3)
BC3 = np.round(np.std(np.abs(tracking_errors.loc[BC, 'TE_1D'])),3)
BC4 = np.round(np.std(np.abs(tracking_errors.loc[BC, 'TE_1D_BS'])),3)

CA1 = np.round(np.std(tracking_errors.loc[CA, 'TE_1D']),3)
CA2 = np.round(np.std(tracking_errors.loc[CA, 'TE_1D_BS']),3)
CA3 = np.round(np.std(np.abs(tracking_errors.loc[CA, 'TE_1D'])),3)
CA4 = np.round(np.std(np.abs(tracking_errors.loc[CA, 'TE_1D_BS'])),3)
CB1 = np.round(np.std(tracking_errors.loc[CB, 'TE_1D']),3)
CB2 = np.round(np.std(tracking_errors.loc[CB, 'TE_1D_BS']),3)
CB3 = np.round(np.std(np.abs(tracking_errors.loc[CB, 'TE_1D'])),3)
CB4 = np.round(np.std(np.abs(tracking_errors.loc[CB, 'TE_1D_BS'])),3)
CC1 = np.round(np.std(tracking_errors.loc[CC, 'TE_1D']),3)
CC2 = np.round(np.std(tracking_errors.loc[CC, 'TE_1D_BS']),3)
CC3 = np.round(np.std(np.abs(tracking_errors.loc[CC, 'TE_1D'])),3)
CC4 = np.round(np.std(np.abs(tracking_errors.loc[CC, 'TE_1D_BS'])),3)

# 2nd step the mean
AA7 = np.round(np.mean(tracking_errors.loc[AA, 'TE_1D']),3)
AA8 = np.round(np.mean(tracking_errors.loc[AA, 'TE_1D_BS']),3)
AA9 = np.round(np.mean(np.abs(tracking_errors.loc[AA, 'TE_1D'])),3)
AA10 = np.round(np.mean(np.abs(tracking_errors.loc[AA, 'TE_1D_BS'])),3)
AA11 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[AA, 'TE_1D'])**2 + np.var(tracking_errors.loc[AA, 'TE_1D'])),3)
AA12 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[AA, 'TE_1D_BS'])**2 + np.var(tracking_errors.loc[AA, 'TE_1D_BS'])),3)
AB7 = np.round(np.mean(tracking_errors.loc[AB, 'TE_1D']),3)
AB8 = np.round(np.mean(tracking_errors.loc[AB, 'TE_1D_BS']),3)
AB9 = np.round(np.mean(np.abs(tracking_errors.loc[AB, 'TE_1D'])),3)
AB10 = np.round(np.mean(np.abs(tracking_errors.loc[AB, 'TE_1D_BS'])),3)
AB11 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[AB, 'TE_1D'])**2 + np.var(tracking_errors.loc[AB, 'TE_1D'])),3)
AB12 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[AB, 'TE_1D_BS'])**2 + np.var(tracking_errors.loc[AB, 'TE_1D_BS'])),3)
AC7 = np.round(np.mean(tracking_errors.loc[AC, 'TE_1D']),3)
AC8 = np.round(np.mean(tracking_errors.loc[AC, 'TE_1D_BS']),3)
AC9 = np.round(np.mean(np.abs(tracking_errors.loc[AC, 'TE_1D'])),3)
AC10 = np.round(np.mean(np.abs(tracking_errors.loc[AC, 'TE_1D_BS'])),3)
AC11 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[AC, 'TE_1D'])**2 + np.var(tracking_errors.loc[AC, 'TE_1D'])),3)
AC12 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[AC, 'TE_1D_BS'])**2 + np.var(tracking_errors.loc[AC, 'TE_1D_BS'])),3)

BA7 = np.round(np.mean(tracking_errors.loc[BA, 'TE_1D']),3)
BA8 = np.round(np.mean(tracking_errors.loc[BA, 'TE_1D_BS']),3)
BA9 = np.round(np.mean(np.abs(tracking_errors.loc[BA, 'TE_1D'])),3)
BA10 = np.round(np.mean(np.abs(tracking_errors.loc[BA, 'TE_1D_BS'])),3)
BA11 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[BA, 'TE_1D'])**2 + np.var(tracking_errors.loc[BA, 'TE_1D'])),3)
BA12 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[BA, 'TE_1D_BS'])**2 + np.var(tracking_errors.loc[BA, 'TE_1D_BS'])),3)
BB7 = np.round(np.mean(tracking_errors.loc[BB, 'TE_1D']),3)
BB8 = np.round(np.mean(tracking_errors.loc[BB, 'TE_1D_BS']),3)
BB9 = np.round(np.mean(np.abs(tracking_errors.loc[BB, 'TE_1D'])),3)
BB10 = np.round(np.mean(np.abs(tracking_errors.loc[BB, 'TE_1D_BS'])),3)
BB11 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[BB, 'TE_1D'])**2 + np.var(tracking_errors.loc[BB, 'TE_1D'])),3)
BB12 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[BB, 'TE_1D_BS'])**2 + np.var(tracking_errors.loc[BB, 'TE_1D_BS'])),3)
BC7 = np.round(np.mean(tracking_errors.loc[BC, 'TE_1D']),3)
BC8 = np.round(np.mean(tracking_errors.loc[BC, 'TE_1D_BS']),3)
BC9 = np.round(np.mean(np.abs(tracking_errors.loc[BC, 'TE_1D'])),3)
BC10 = np.round(np.mean(np.abs(tracking_errors.loc[BC, 'TE_1D_BS'])),3)
BC11 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[BC, 'TE_1D'])**2 + np.var(tracking_errors.loc[BC, 'TE_1D'])),3)
BC12 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[BC, 'TE_1D_BS'])**2 + np.var(tracking_errors.loc[BC, 'TE_1D_BS'])),3)

CA7 = np.round(np.mean(tracking_errors.loc[CA, 'TE_1D']),3)
CA8 = np.round(np.mean(tracking_errors.loc[CA, 'TE_1D_BS']),3)
CA9 = np.round(np.mean(np.abs(tracking_errors.loc[CA, 'TE_1D'])),3)
CA10 = np.round(np.mean(np.abs(tracking_errors.loc[CA, 'TE_1D_BS'])),3)
CA11 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[CA, 'TE_1D'])**2 + np.var(tracking_errors.loc[CA, 'TE_1D'])),3)
CA12 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[CA, 'TE_1D_BS'])**2 + np.var(tracking_errors.loc[CA, 'TE_1D_BS'])),3)
CB7 = np.round(np.mean(tracking_errors.loc[CB, 'TE_1D']),3)
CB8 = np.round(np.mean(tracking_errors.loc[CB, 'TE_1D_BS']),3)
CB9 = np.round(np.mean(np.abs(tracking_errors.loc[CB, 'TE_1D'])),3)
CB10 = np.round(np.mean(np.abs(tracking_errors.loc[CB, 'TE_1D_BS'])),3)
CB11 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[CB, 'TE_1D'])**2 + np.var(tracking_errors.loc[CB, 'TE_1D'])),3)
CB12 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[CB, 'TE_1D_BS'])**2 + np.var(tracking_errors.loc[CB, 'TE_1D_BS'])),3)
CC7 = np.round(np.mean(tracking_errors.loc[CC, 'TE_1D']),3)
CC8 = np.round(np.mean(tracking_errors.loc[CC, 'TE_1D_BS']),3)
CC9 = np.round(np.mean(np.abs(tracking_errors.loc[CC, 'TE_1D'])),3)
CC10 = np.round(np.mean(np.abs(tracking_errors.loc[CC, 'TE_1D_BS'])),3)
CC11 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[CC, 'TE_1D'])**2 + np.var(tracking_errors.loc[CC, 'TE_1D'])),3)
CC12 = np.round(np.sqrt(np.mean(
    tracking_errors.loc[CC, 'TE_1D_BS'])**2 + np.var(tracking_errors.loc[CC, 'TE_1D_BS'])),3)

# 3rd step the condidence interval


def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  
    margin_of_error = std_dev * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound


AA13 = np.round(calculate_confidence_interval(tracking_errors.loc[AA, 'TE_1D']),3)
AA14 = np.round(calculate_confidence_interval(tracking_errors.loc[AA, 'TE_1D_BS']),3)
AB13 = np.round(calculate_confidence_interval(tracking_errors.loc[AB, 'TE_1D']),3)
AB14 = np.round(calculate_confidence_interval(tracking_errors.loc[AB, 'TE_1D_BS']),3)
AC13 = np.round(calculate_confidence_interval(tracking_errors.loc[AC, 'TE_1D']),3)
AC14 = np.round(calculate_confidence_interval(tracking_errors.loc[AC, 'TE_1D_BS']),3)

BA13 = np.round(calculate_confidence_interval(tracking_errors.loc[BA, 'TE_1D']),3)
BA14 = np.round(calculate_confidence_interval(tracking_errors.loc[BA, 'TE_1D_BS']),3)
BB13 = np.round(calculate_confidence_interval(tracking_errors.loc[BB, 'TE_1D']),3)
BB14 = np.round(calculate_confidence_interval(tracking_errors.loc[BB, 'TE_1D_BS']),3)
BC13 = np.round(calculate_confidence_interval(tracking_errors.loc[BC, 'TE_1D']),3)
BC14 = np.round(calculate_confidence_interval(tracking_errors.loc[BC, 'TE_1D_BS']),3)

CA13 = np.round(calculate_confidence_interval(tracking_errors.loc[CA, 'TE_1D']),3)
CA14 = np.round(calculate_confidence_interval(tracking_errors.loc[CA, 'TE_1D_BS']),3)
CB13 = np.round(calculate_confidence_interval(tracking_errors.loc[CB, 'TE_1D']),3)
CB14 = np.round(calculate_confidence_interval(tracking_errors.loc[CB, 'TE_1D_BS']),3)
CC13 = np.round(calculate_confidence_interval(tracking_errors.loc[CC, 'TE_1D']),3)
CC14 = np.round(calculate_confidence_interval(tracking_errors.loc[CC, 'TE_1D_BS']),3)


AA15 = sum(AA)
AB15 = sum(AB)
AC15 = sum(AC)
BA15 = sum(BA)
BB15 = sum(BB)
BC15 = sum(BC)
CA15 = sum(CA)
CB15 = sum(CB)
CC15 = sum(CC)
Total = AA15 + AB15 + AC15 + BA15 + BB15 + BC15 + CA15 + CB15 + CC15



rows = [
    ['Moneyness','Expiry', "", "", 'TE', "", 'ATE', "", 'PE','Observations'],
    ["", "", "", 'ANN', 'BS', 'ANN', 'BS', 'ANN', 'BS',Total],
    ['', '', 'Stdev', AA1, AA2, AA3, AA4, '', '',''],
    ['', 'Short', 'Mean', AA7, AA8, AA9, AA10, AA11, AA12, AA15],
    ['', '', 'CI', AA13,  AA14, '', '', '', '',''],
    ['', '', 'Stdev', AB1, AB2, AB3, AB4, '', '',''],
    ['ITM', 'Medium', 'Mean', AB7, AB8, AB9, AB10, AB11, AB12, AB15],
    ['', '', 'CI', AB13,  AB14, '', '', '', '', ''],
    ['', '', 'Stdev', AC1, AC2, AC3, AC4, '', '',''],
    ['', 'Long', 'Mean', AC7, AC8, AC9, AC10, AC11, AC12,AC15],
    ['', '', 'CI', AC13,  AC14, '', '', '', ''],




    ["", "", "", 'ANN', 'BS', 'ANN', 'BS', 'ANN', 'BS',''],
    ['', '', 'Stdev', BA1, BA2, BA3, BA4, '', '',''],
    ['', 'Short', 'Mean', BA7, BA8, BA9, BA10, BA11, BA12, BA15],
    ['', '', 'CI', BA13,  BA14, '', '', '', '',''],
    ['', '', 'Stdev', BB1, BB2, BB3, BB4, '', ''],
    ['NTM', 'Medium', 'Mean', BB7, BB8, BB9, BB10, BB11, BB12, BB15],
    ['', '', 'CI', BB13,  BB14, '', '', '', '',''],
    ['', '', 'Stdev', BC1, BC2, BC3, BC4, '', '',''],
    ['', 'Long', 'Mean', BC7, BC8, BC9, BC10, BC11, BC12, BC15],
    ['', '', 'CI', BC13,  BC14, '', '', '', '', ''],






    ["", "", "", 'ANN', 'BS', 'ANN', 'BS', 'ANN', 'BS', ''],
    ['', '', 'Stdev', CA1, CA2, CA3, CA4, '', '',''],
    ['', 'Short', 'Mean', CA7, CA8, CA9, CA10, CA11, CA12, CA15],
    ['', '', 'CI', CA13,  CA14, '', '', '', '',''],
    ['', '', 'Stdev', CB1, CB2, CB3, CB4, '', '',''],
    ['OTM', 'Medium', 'Mean', CB7, CB8, CB9, CB10, CB11, CB12, CB15],
    ['', '', 'CI', CB13,  CB14, '', '', '', '',''],
    ['', '', 'Stdev', CC1, CC2, CC3, CC4, '', '',''],
    ['', 'Long', 'Mean', CC7, CC8, CC9, CC10, CC11, CC12, CC15],
    ['', '', 'CI', CC13,  CC14, '', '', '', '', ''],




]

table_hedge = tabulate(rows, headers='firstrow', tablefmt="latex")

