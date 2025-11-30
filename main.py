import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
import os
import time
from data_loader import load_and_preprocess_data
from model_training import train_models
from plottingg import generate_plots
from plot_validation_performance import plot_validation_performance
from algorithm_comparison import compare_algorithms
from weight_exporter import export_weights
from algorithm_plots import plot_algorithm_comparison
from results_saver import save_results
from regression_plots import (generate_combined_plots, 
                             save_metrics, 
                             generate_enhanced_grid)

# START Execution time
start_time = time.time() 

# Set all seeds and deterministic configurations
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

# Configure TensorFlow determinism
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Load and normalize dataset
P, Ra_s = load_and_preprocess_data()

# Train models with different neuron counts and collect metrics
(rmse_list, r2_list, smape_list,
 histories, models,
 best_r2_idx, best_rmse_idx, best_smape_idx,
 best_model) = train_models(P, Ra_s)

# Generate metric plots and error analysis
generate_plots(
    N=np.arange(1, 21),
    r2_list=r2_list,
    rmse_list=rmse_list,
    smape_list=smape_list,
    best_r2_idx=best_r2_idx,
    best_rmse_idx=best_rmse_idx,
    best_smape_idx=best_smape_idx,
    histories=histories,
    Ra_s=Ra_s,
    P=P,
    best_model=best_model
)


# Split data and evaluate model 
(val_model, val_history, test_loss, 
 X_train, X_valtest, y_train, y_valtest,
 X_val, X_test, y_val, y_test, train_loss,
 val_loss) = plot_validation_performance(
    P=P,
    Ra_s=Ra_s,
    N=np.arange(1, 21),
    best_rmse_idx=best_rmse_idx
)


# Test different optimization algorithms
best_neurons = np.arange(1, 21)[best_rmse_idx]  # Convert index to neuron count
optimizers = ['adam', 'sgd', 'rmsprop', 'adamax', 'nadam', 'adagrad']

(smape_values, r2_values, optim_models,
 best_optim_idx, best_optim_model) = compare_algorithms(
    P=P,
    Ra_s=Ra_s,
    best_neurons=best_neurons,
    optimizers=optimizers)

# Maintain original best model reference
best_model = models[best_rmse_idx]

# Save trained weights and algorithm comparison results
export_weights(best_model)
plot_algorithm_comparison(optimizers, smape_values, r2_values)

# Save key metrics
save_results(
    best_r2_idx=best_r2_idx,
    best_smape_idx=best_smape_idx,
    best_rmse_idx=best_rmse_idx,
    r2_list=r2_list,
    rmse_list=rmse_list,
    smape_list=smape_list
)

# Generate predictions for all datasets
train_pred = val_model.predict(X_train).flatten()
val_pred = val_model.predict(X_val).flatten()
test_pred = val_model.predict(X_test).flatten()
all_pred = val_model.predict(P.T).flatten()

# Create combined regression plots
generate_combined_plots(
    y_train, train_pred, y_val, val_pred, y_test, test_pred,
    X_train, X_val, X_test, P, val_model, train_loss,
    val_loss, test_loss, Ra_s
)

# Save mse every epoch
epoch_mse = save_metrics(train_loss, val_loss, test_loss)

# Create 2x2 grid of enhanced regression plots
generate_enhanced_grid(
    y_train, train_pred, y_val, val_pred,
    y_test, test_pred, Ra_s, val_model.predict(P.T).flatten()
)

# End execution time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")

