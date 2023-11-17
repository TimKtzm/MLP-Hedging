#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:53:26 2023

@author: timothee
"""

import os
import sys
import tensorflow as tf
os.chdir('/Users/timothee/Documents/IESEG/4a/Master Thesis/Data_Management_GARCH')
sys.path.append('/Users/timothee/Documents/IESEG/4a/Master Thesis/Data_Management_GARCH/Code')
from ANN import load_custom_model
from keras_visualizer import visualizer



model = load_custom_model('my_FFNN_model.h5')


visualizer(model, file_name='network_architecture', file_format='pdf', view=True, settings = {
    # ALL LAYERS
    'MAX_NEURONS': 10,
    'ARROW_COLOR': 'steelblue',
    # INPUT LAYERS
    'INPUT_DENSE_COLOR': 'red',
    'INPUT_EMBEDDING_COLOR': 'black',
    'INPUT_EMBEDDING_FONT': 'purple',
    'INPUT_GRAYSCALE_COLOR': 'black:white',
    'INPUT_GRAYSCALE_FONT': 'white',
    'INPUT_RGB_COLOR': '#e74c3c:#3498db',
    'INPUT_RGB_FONT': 'white',
    'INPUT_LAYER_COLOR': 'black',
    'INPUT_LAYER_FONT': 'white',
    # HIDDEN LAYERS
    'HIDDEN_DENSE_COLOR': 'sienna',
    'HIDDEN_CONV_COLOR': '#5faad0',
    'HIDDEN_CONV_FONT': 'black',
    'HIDDEN_POOLING_COLOR': '#8e44ad',
    'HIDDEN_POOLING_FONT': 'white',
    'HIDDEN_FLATTEN_COLOR': '#2c3e50',
    'HIDDEN_FLATTEN_FONT': 'white',
    'HIDDEN_DROPOUT_COLOR': '#f39c12',
    'HIDDEN_DROPOUT_FONT': 'black',
    'HIDDEN_ACTIVATION_COLOR': '#00b894',
    'HIDDEN_ACTIVATION_FONT': 'black',
    'HIDDEN_LAYER_COLOR': 'black',
    'HIDDEN_LAYER_FONT': 'white',
    # OUTPUT LAYER
    'OUTPUT_DENSE_COLOR': '#e74c3c',
    'OUTPUT_LAYER_COLOR': 'black',
    'OUTPUT_LAYER_FONT': 'white',
})


