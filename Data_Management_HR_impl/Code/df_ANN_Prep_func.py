#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:15:19 2023

@author: timothee
"""
import sys
import os
import pandas as pd
import scipy
import matplotlib 
import numpy as np
from sklearn.model_selection import train_test_split



def split_X_Y(data):
    X = data.copy()
    y = X.pop('delta')
    return X, y


def split_data(X, y):
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=None, shuffle=False)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state= None, shuffle=False)
    X_index = X_test.index.to_frame(index=True)
    return X_test, y_test, X_train, y_train, X_valid, y_valid, X_index


    
    

