# -*- coding: utf-8 -*-
"""
Created on Fri May  9 03:57:10 2025

@author: 36955
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

def compare_algorithms(P, Ra_s, best_neurons, optimizers):
    optim_models = []
    smape_values = []
    r2_values = []

    for opt in optimizers:
        print(f"Training with {opt} algorithm")
        model = Sequential([
            Dense(best_neurons, activation='sigmoid', input_shape=(4,)),
            Dense(best_neurons, activation='sigmoid'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer=opt, loss='mse')
        model.fit(P.T, Ra_s, epochs=1000, verbose=0)
        optim_models.append(model)
        
        Ra_p = model.predict(P.T).flatten()
        e = Ra_s - Ra_p
        
        smape = np.mean(2 * np.abs(e) / (np.abs(Ra_s) + np.abs(Ra_p))) * 100
        r2 = r2_score(Ra_s, Ra_p)
        
        smape_values.append(smape)
        r2_values.append(r2)

    best_optim_idx = np.argmax(r2_values)
    best_optim_model = optim_models[best_optim_idx]
    
    return (smape_values, r2_values, optim_models,
            best_optim_idx, best_optim_model,
            )
