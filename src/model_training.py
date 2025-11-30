# -*- coding: utf-8 -*-
"""
Created on Fri May  9 03:39:03 2025

@author: 36955
"""


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score

def train_models(P, Ra_s):
    # Initialize metrics storage
    rmse_list = []
    r2_list = []
    smape_list = []
    histories = []
    models = []

    # Main training loop
    for i in range(1, 21):
        print(f"Training model with {i} neurons")
        model = Sequential([
            Dense(i, activation='sigmoid', input_shape=(4,)),
            Dense(i, activation='sigmoid'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(P.T, Ra_s, epochs=1000, verbose=0)
        histories.append(history)
        models.append(model)
        
        Ra_p = model.predict(P.T).flatten()
        e = Ra_s - Ra_p
        
        rmse = np.sqrt(mean_squared_error(Ra_s, Ra_p))
        r2 = r2_score(Ra_s, Ra_p)
        smape = np.mean(2 * np.abs(e) / (np.abs(Ra_s) + np.abs(Ra_p))) * 100
        
        rmse_list.append(rmse)
        r2_list.append(r2)
        smape_list.append(smape)

    # Find best neuron counts
    best_r2_idx = np.argmax(r2_list)
    best_rmse_idx = np.argmin(rmse_list)
    best_smape_idx = np.argmin(smape_list)
    best_model = models[best_rmse_idx]
    
    return (rmse_list, r2_list, smape_list,
            histories, models,
            best_r2_idx, best_rmse_idx, best_smape_idx,
            best_model)
