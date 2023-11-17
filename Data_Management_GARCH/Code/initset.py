#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:53:36 2023

@author: timothee
"""


#This snippet is inspired from Ruf & Wang (2020)
from pandas.tseries.offsets import BDay

OFFSET_DICT = {
    '1D': [BDay(1), '_1D'],
    '2D': [BDay(2), '_2D'],
    '5D': [BDay(5), '_5D']
}