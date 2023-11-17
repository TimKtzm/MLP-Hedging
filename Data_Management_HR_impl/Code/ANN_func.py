#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:30:02 2023

@author: timothee
"""
import sys
import os
import pandas as pd
import scipy
import matplotlib 
import numpy as np
import keras
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend
from keras.models import Sequential
tf.data.experimental.enable_debug_mode()
from tensorflow.python.keras.callbacks import TensorBoard


def custom_activation(x):
    return backend.exp(x)


def create_huber(threshold=0.05):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn


root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard = keras.callbacks.TensorBoard(run_logdir)


def build_ANN(X, num_units, dropout_rate, loss = 'mse'):

    model = Sequential()
    
    model.add(Dense(units = num_units, input_dim = X.shape[1], activation= LeakyReLU()))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(num_units, activation='elu'))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(num_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(num_units, activation='elu'))
    model.add(Dropout(dropout_rate))
    
    #model.add(Dense(1))
    #model.add(Activation(custom_activation))
    model.add(Dense(1, activation='tanh'))
    
    model.compile(loss=loss, optimizer='Nadam', metrics=["mae"])
    
    
    model.summary()
    return model
    
    
    
    

def history (model, X_train, X_valid, y_train, y_valid):
    checkpoint_cb = keras.callbacks.ModelCheckpoint("my_FFNN_model.h5",
                                                    save_best_only=True)    
    
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                      restore_best_weights=True)
    
    
    history = model.fit(X_train, y_train, epochs=100,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard])
    
    learning_curves = pd.DataFrame(history.history).plot(figsize=(8, 5))
    
    return history, learning_curves
        

def load_custom_model(filename):
    model = keras.models.load_model(filename, custom_objects={"huber_fn": create_huber(1.0), "custom_activation": custom_activation})
    return model  




def pred_X(model, X, X_index):
    M_pred = model.predict(X)
    M_pred = pd.DataFrame(M_pred)
    M_pred.index = X_index.index
    M_pred.rename(columns={M_pred.columns[0]: 'C0K_hat'}, inplace=True)
    return M_pred



