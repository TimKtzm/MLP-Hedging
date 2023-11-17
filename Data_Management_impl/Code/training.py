#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:20:16 2023

@author: timothee
"""

import sys
import os
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns



sys.path.append('/Users/timothee/Documents/IESEG/4a/Master Thesis/Data_Management_impl/Code')
from df_ANN_Prep_func import split_X_Y, split_data
from ANN_func import build_ANN, history, create_huber, pred_X ,compute_jacobian

os.chdir('/Users/timothee/Documents/IESEG/4a/Master Thesis/Data_Management_impl')

df_op = pd.read_csv('ready_for_ANN.csv')
df_op = df_op[['C0/K', 'M', 'Time_to_expiry', 'impl_volatility', 'cp_int']]

columns_to_check = ['C0/K', 'M', 'Time_to_expiry', 'impl_volatility', 'cp_int']
df_op.dropna(subset=columns_to_check, inplace=True)

#Split X and y
X, y = split_X_Y(data=df_op)

#Cross_validation
X_test, y_test, X_train, y_train, X_valid, y_valid, X_index = split_data(X, y)



#Descriptive statistics


ITM_call = ((X_train['cp_int'] == 0) & (X_train['M'] > 1)).sum() + ((X_valid['cp_int'] == 0) & (X_valid['M'] > 1)).sum()
OTM_call = ((X_train['cp_int'] == 0) & (X_train['M'] < 1)).sum() + ((X_valid['cp_int'] == 0) & (X_valid['M'] < 1)).sum()
ITM_put = ((X_train['cp_int'] == 1) & (X_train['M'] < 1)).sum() + ((X_valid['cp_int'] == 1) & (X_valid['M'] < 1)).sum()
OTM_put = ((X_train['cp_int'] == 1) & (X_train['M'] > 1)).sum() + ((X_valid['cp_int'] == 1) & (X_valid['M'] > 1)).sum()
OS_ITM_call = ((X_test['cp_int'] == 0) & (X_test['M'] > 1)).sum()
OS_OTM_call = ((X_test['cp_int'] == 0) & (X_test['M'] < 1)).sum()
OS_ITM_put = ((X_test['cp_int'] == 1) & (X_test['M'] < 1)).sum()
OS_OTM_put = ((X_test['cp_int'] == 1) & (X_test['M'] > 1)).sum()

total_IS_call = ITM_call + OTM_call
total_IS_put = ITM_put + OTM_put
total_OS_call = OS_ITM_call + OS_ITM_put
total_OS_put = OS_OTM_call + OS_OTM_put

# Plot the stacked bar chart for ITM and OTM counts
colors = ['#1f77b4', '#ff7f0e']

plt.bar(
    ['IS Calls', 'IS Puts', 'OS Calls', 'OS Puts'],
    [ITM_call, ITM_put, OS_ITM_call, OS_ITM_put],
    color=colors[0]
)

plt.bar(
    ['IS Calls', 'IS Puts', 'OS Calls', 'OS Puts'],
    [OTM_call, OTM_put, OS_OTM_call, OS_OTM_put],
    bottom=[ITM_call, ITM_put, OS_ITM_call, OS_ITM_put],
    color=colors[1]
)


plt.xlabel('Sample Type')
plt.ylabel('Count')
plt.title('Total Options Frequency')
plt.xticks(rotation=45, ha='right')  
plt.legend(["ITM", "OTM"])
plt.tight_layout()  
plt.savefig('total_options_frequency.png', dpi=800)



#IV surface
fig = plt.figure(figsize=(7, 7))
x1 = df_op['M']
x2 = df_op['Time_to_expiry']
y = df_op['impl_volatility']
x1 = np.clip(x1, 0.5, 2)
x2 = np.clip(x2, 0, 1)
y1 = np.clip(y, 0, 1)
marker_size = 2  
ax1 = fig.add_subplot(111, projection='3d')
sc1 = ax1.scatter(x1, x2, y1, c=y1, cmap=cm.PuOr, marker='o', s=marker_size, alpha=0.50, label='Implied Volatility surface')
ax1.set_xlabel('Moneyness')
ax1.set_ylabel('Time-to-expiry')
ax1.set_zlabel('Implied Volatility')
ax1.set_xlim(0.5, 2)
ax1.set_ylim(0, 1)
fig.subplots_adjust(right=0.95, bottom=0.1)
ax1.grid(True)
ax1.legend()
ax1.view_init(elev=30, azim=-230)
plt.savefig('IV_surface', dpi=800)


#PDF Time-to-expiry
plt.figure(figsize=(8, 6))
ax = sns.kdeplot(df_op['Time_to_expiry'], shade=True, color="skyblue", cbar=True, cbar_kws={'label': 'Density'})
plt.xlabel('Time-to-Expiry')
y_ticks = ax.get_yticks()
probabilities = y_ticks / y_ticks.sum()
ax.set_yticklabels(['{:,.2%}'.format(p) for p in probabilities])
plt.ylabel('Density')
plt.xlim(0, 1.5)
plt.savefig('PDF-expiry', dpi=800)



#CDF Time-to-expiry
ax = sns.kdeplot(df_op['Time_to_expiry'], shade=True, color="skyblue", cumulative=True)
plt.xlabel('Time-to-Expiry')
plt.ylabel('Cumulative Probability')
ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
plt.xlim(0, 2)
plt.ylim(0, 1.5)
plt.savefig('CDF-expiry', dpi=800)







#MLP model
X_test = tf.constant(X_test, dtype=tf.float32)
y_test = tf.constant(y_test, dtype=tf.float32)
X_train = tf.constant(X_train, dtype=tf.float32)
y_train = tf.constant(y_train, dtype=tf.float32)
X_valid = tf.constant(X_valid, dtype=tf.float32)
y_valid = tf.constant(y_valid, dtype=tf.float32)

model = build_ANN(X_test, num_units = 40, dropout_rate = 0.25, loss= create_huber(1))


history, learning_curves = history(model, X_train, X_valid, y_train, y_valid)

eval = model.evaluate(X_test, y_test)

C0K_pred = pred_X(model, X_test, X_index)
C0K_pred.to_csv('C0K_pred.csv', index=True)  

jacobian_df = compute_jacobian(X_test, model, X_index)
jacobian_df.to_csv('jacobian_df.csv', index=True)