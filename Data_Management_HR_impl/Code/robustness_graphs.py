#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:28:55 2023

@author: timothee
"""


import os
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

os.chdir('/Users/timothee/Documents/IESEG/4a/Master Thesis/Data_Management_HR_impl')
hedge_df = pd.read_csv('hedge_df.csv')
hedge_df = hedge_df[hedge_df['cp_flag'] == 'P']
tracking_errors = pd.read_csv('tracking_errors.csv')






fig = plt.figure(figsize=(7, 7))
x1 = hedge_df['M']
x2 = hedge_df['Time_to_expiry']
y = np.abs(hedge_df['V_P_1D'])-np.abs(hedge_df['V_P_1D_BS'])
x1 = np.clip(x1, 0.5, 2)
x2 = np.clip(x2, 0, 1)
y1 = np.clip(y, 0, 20)
y2 = y
negative_mask = y < 0

marker_size = 2  

ax1 = fig.add_subplot(111, projection='3d')
sc1 = ax1.scatter(x1, x2, y1, c=y1, cmap=cm.magma_r, marker='o', s=marker_size, alpha=0.50, label='Value of the relative hedging error using hedging optimization criterion (ANN>BS)')
sc2 = ax1.scatter(x1[negative_mask], x2[negative_mask], np.abs(y2[negative_mask]), c='lime', marker='x', s=marker_size, label='Value of the BS hedging error (ANN<BS)')
ax1.set_xlabel('Moneyness')
ax1.set_ylabel('Time-to-expiry')
ax1.set_zlabel('Error')

ax1.set_xlim(0.5, 2)
ax1.set_ylim(0, 1)

fig.subplots_adjust(right=0.95, bottom=0.1)

ax1.grid(True)
ax1.legend()
ax1.view_init(elev=20, azim=-25)

plt.savefig('hedgedANN_impl.png', dpi=800)




