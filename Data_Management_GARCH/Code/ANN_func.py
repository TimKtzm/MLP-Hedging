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
    model.add(Dense(1, activation=custom_activation))
    
    model.compile(loss=loss, optimizer='Nadam', metrics=["mae"])
    
    
    model.summary()
    return model
    
    
    
    

def history (model, X_train, X_valid, y_train, y_valid):
    checkpoint_cb = keras.callbacks.ModelCheckpoint("my_FFNN_model.h5",
                                                    save_best_only=True)    
    
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=20,
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



import resource

# Function to check if memory usage exceeds the threshold
def is_memory_limit_exceeded(soft_limit_mb):
    soft_limit_bytes = soft_limit_mb * 1024 * 1024  #=bites
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage > soft_limit_bytes


def compute_jacobian(X, model, X_index):
    soft_memory_limit_mb = 800  

    try:
        with tf.GradientTape(persistent=True) as tape:

            tape.watch(X)

            y_pred = model(X)

        # Check if memory usage exceeds the threshold
        if is_memory_limit_exceeded(soft_memory_limit_mb):
            raise MemoryError("Memory usage exceeded soft limit during tape.watch")

        # Computes the gradients of predictions with respect to input features
        jacobian = tape.jacobian(y_pred, X)
        del tape
        jacobian_reshaped = jacobian.numpy().reshape(X.shape[0], -1)
        column_names = [f'Jacobian_{i}' for i in range(jacobian_reshaped.shape[1])]
        jacobian_df = pd.DataFrame(jacobian_reshaped, columns=column_names)

        def shift_non_zero(row):
            non_zero_values = [value for value in row if value != 0]
            return non_zero_values + [0] * (len(row) - len(non_zero_values))
        
        jacobian_df = jacobian_df.apply(shift_non_zero, axis=1, result_type='expand')
        jacobian_df = jacobian_df.loc[:, (jacobian_df != 0).any()]
        jacobian_df = jacobian_df.iloc[:, [0]]
        jacobian_df.rename(columns={jacobian_df.columns[0]: 'delta_nn'}, inplace=True)
        jacobian_df.index = X_index.index

    except MemoryError:
        jacobian_df = compute_jacobian_alternative(X, model, X_index)

    return jacobian_df


def compute_jacobian_alternative(X, model, X_index):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)
        y_pred = model(X)
    gradients2 = tape.gradient(y_pred, X)
    del tape

    gradient_np = gradients2.numpy()
    jacobian_df = pd.DataFrame(gradient_np)
    jacobian_df = jacobian_df.iloc[:, [0]]
    jacobian_df.rename(columns={jacobian_df.columns[0]: 'delta_nn'}, inplace=True)
    jacobian_df.index = X_index.index

    return jacobian_df









