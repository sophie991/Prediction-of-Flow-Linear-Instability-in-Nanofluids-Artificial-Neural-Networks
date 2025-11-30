# -*- coding: utf-8 -*-
"""
Created on Fri May  9 04:02:07 2025

@author: 36955
"""


import pandas as pd

def export_weights(best_model):
    # Extract weights from best model
    w1 = best_model.layers[0].get_weights()[0]
    b1 = best_model.layers[0].get_weights()[1]
    w2 = best_model.layers[1].get_weights()[0]
    b2 = best_model.layers[1].get_weights()[1]
    w3 = best_model.layers[2].get_weights()[0]
    b3 = best_model.layers[2].get_weights()[1]

    # Save best model weights
    pd.DataFrame(w1).to_excel('w1.xlsx', index=False, header=False)
    pd.DataFrame(b1).to_excel('b1.xlsx', index=False, header=False)
    pd.DataFrame(w2).to_excel('w2.xlsx', index=False, header=False)
    pd.DataFrame(b2).to_excel('b2.xlsx', index=False, header=False)
    pd.DataFrame(w3).to_excel('w3.xlsx', index=False, header=False)
    pd.DataFrame(b3).to_excel('b3.xlsx', index=False, header=False)

  
