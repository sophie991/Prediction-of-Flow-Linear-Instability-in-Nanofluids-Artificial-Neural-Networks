import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from enhanced_regression_plot_one import enhanced_regression_plot_one
from plot_regression_one import plot_regression_one
import random
import os
import time

start_time = time.time() 
# Set all seeds
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

# Configure determinism
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Set seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Load and preprocess data
data1 = pd.read_excel('Data.xlsx', header=None).values
data = data1.T

# Extract columns
K1 = data[:, 2]  # 3rd column
K2 = data[:, 3]  # 4th column
K3 = data[:, 4]  # 5th column
K4 = data[:, 5]  # 6th column
K6 = data[:, 1]  # 2nd column

# Normalize features
scaler = MinMaxScaler()
K1 = scaler.fit_transform(K1.reshape(-1, 1)).flatten()
K2 = scaler.fit_transform(K2.reshape(-1, 1)).flatten()
K3 = scaler.fit_transform(K3.reshape(-1, 1)).flatten()
K4 = scaler.fit_transform(K4.reshape(-1, 1)).flatten()
Ra_s = scaler.fit_transform(K6.reshape(-1, 1)).flatten()

P = np.vstack([K1, K2, K3, K4])

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
w1 = best_model.layers[0].get_weights()[0]  # IW{1,1}
b1 = best_model.layers[0].get_weights()[1]  # b{1}
w2 = best_model.layers[1].get_weights()[0]  # LW{2,1}
b2 = best_model.layers[1].get_weights()[1]  # b{2}

# Plotting
plt.figure(figsize=(12, 8))
N = np.arange(1, 21)

# R² plot
plt.subplot(3, 1, 1)
plt.plot(N, r2_list, '-<', markersize=4, linewidth=1.5)
plt.xlabel('Neurons (Nₙ)')
plt.ylabel('R²')
plt.annotate(f'Max: {N[best_r2_idx]}', (N[best_r2_idx], r2_list[best_r2_idx]))

# RMSE plot
plt.subplot(3, 1, 2)
plt.plot(N, rmse_list, '-<', markersize=4, linewidth=1.5)
plt.xlabel('Neurons (Nₙ)')
plt.ylabel('RMSE')
plt.annotate(f'Min: {N[best_rmse_idx]}', (N[best_rmse_idx], rmse_list[best_rmse_idx]))

# sMAPE plot
plt.subplot(3, 1, 3)
plt.plot(N, smape_list, '-<', markersize=4, linewidth=1.5)
plt.xlabel('Neurons (Nₙ)')
plt.ylabel('sMAPE (%)')
plt.annotate(f'Min: {N[best_smape_idx]}', (N[best_smape_idx], smape_list[best_smape_idx]))

plt.tight_layout()

# Error histogram
plt.figure()
errors = Ra_s - best_model.predict(P.T).flatten()
plt.hist(errors, bins=20, edgecolor='black')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')

# Loss curve
plt.figure()
for i, history in enumerate(histories[:]):
    plt.plot(history.history['loss'], label=f'{i+1} neurons')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    # plt.legend()

# Regression plot
plt.figure()
Ra_p = best_model.predict(P.T).flatten()
plt.scatter(Ra_s, Ra_p, alpha=0.6)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('Actual (Normalized)')
plt.ylabel('Predicted (Normalized)')

# Validation performance plot
X_train, X_valtest, y_train, y_valtest = train_test_split(P.T, Ra_s, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=0)

val_model = Sequential([
    Dense(N[best_rmse_idx], activation='sigmoid', input_shape=(4,)),
    Dense(1, activation='linear')
])
val_model.compile(optimizer='adam', loss='mse')
val_history = val_model.fit(X_train, y_train,
                          epochs=1000,
                          validation_data=(X_val, y_val),
                          verbose=0)

test_loss = val_model.evaluate(X_test, y_test, verbose=0)
val_loss = val_history.history['val_loss']
train_loss = val_history.history['loss']
best_epoch = np.argmin(val_loss)

plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.axvline(best_epoch, color='r', linestyle='--', label='Best Model')
plt.axhline(test_loss, color='g', linestyle='-.', label='Test Loss')
plt.title('Validation Performance')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

# Algorithm comparison
best_neurons = N[best_rmse_idx]
optimizers = ['adam', 'sgd', 'rmsprop', 'adamax', 'nadam', 'adagrad']
optim_models = []

smape_values = []
r2_values = []

for opt in optimizers:
    print(f"Training with {opt} algorithm")
    model = Sequential([
        Dense(best_neurons, activation='sigmoid', input_shape=(4,)),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer=opt, loss='mse')
    history = model.fit(P.T, Ra_s, epochs=1000, verbose=0)
    optim_models.append(model)

    Ra_p = model.predict(P.T).flatten()
    e = Ra_s - Ra_p

    smape = np.mean(2 * np.abs(e) / (np.abs(Ra_s) + np.abs(Ra_p))) * 100
    r2 = r2_score(Ra_s, Ra_p)

    smape_values.append(smape)
    r2_values.append(r2)

best_optim_idx = np.argmax(r2_values)
best_optim_model = optim_models[best_optim_idx]

ow1 = best_optim_model.layers[0].get_weights()[0]
ob1 = best_optim_model.layers[0].get_weights()[1]
ow2 = best_optim_model.layers[1].get_weights()[0]
ob2 = best_optim_model.layers[1].get_weights()[1]

best_model = models[best_rmse_idx]

# Extract weights and biases from all layers
w1 = best_model.layers[0].get_weights()[0]  # First hidden layer weights
b1 = best_model.layers[0].get_weights()[1]  # First hidden layer biases
w2 = best_model.layers[1].get_weights()[0]  # Output layer weights
b2 = best_model.layers[1].get_weights()[1]  # Output layer biases
pd.DataFrame(w1).to_excel('one_w1.xlsx', index=False, header=False)
pd.DataFrame(b1).to_excel('one_b1.xlsx', index=False, header=False)
pd.DataFrame(w2).to_excel('one_w2.xlsx', index=False, header=False)
pd.DataFrame(b2).to_excel('one_b2.xlsx', index=False, header=False)
# Algorithm comparison plot
plt.figure(figsize=(15, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# sMAPE subplot
plt.subplot(1, 2, 1)
bars = plt.bar(optimizers, smape_values, color=colors)
plt.ylabel('sMAPE (%)', fontsize=12)
plt.title('Algorithm Comparison - sMAPE', fontsize=14)
plt.xticks(rotation=45)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom',
             fontsize=10)

# R² subplot
plt.subplot(1, 2, 2)
bars = plt.bar(optimizers, r2_values, color=colors)
plt.ylabel('R² Value', fontsize=12)
plt.title('Algorithm Comparison - R²', fontsize=14)
plt.ylim(0, 1.1)  # Set consistent scale for R²
plt.xticks(rotation=45)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom',
             fontsize=10)

plt.tight_layout()

# Save results
np.savetxt('DDD.txt', [best_r2_idx+1, best_smape_idx+1, best_rmse_idx+1])
results_table = pd.DataFrame({
    'Neurons': [best_r2_idx+1],
    'R²': [r2_list[best_r2_idx]],
    'RMSE': [rmse_list[best_rmse_idx]],
    'sMAPE': [smape_list[best_smape_idx]]
})
# results_table.to_excel('table_data_adagrad.xlsx', index=False)

plt.show()

# Get predictions for all sets
train_pred = val_model.predict(X_train).flatten()
val_pred = val_model.predict(X_val).flatten()
test_pred = val_model.predict(X_test).flatten()
all_pred = val_model.predict(P.T).flatten()

# Individual plots
plot_regression_one(y_train, train_pred, 'Training Set Regression')
plot_regression_one(y_val, val_pred, 'Validation Set Regression')
plot_regression_one(y_test, test_pred, 'Test Set Regression')

# Combined plot
plt.figure(figsize=(7, 7))
plt.scatter(y_train, train_pred, alpha=0.6, label='Training',
            edgecolor='w', linewidth=0.5)
plt.scatter(y_val, val_pred, alpha=0.6, label='Validation',
            marker='s', edgecolor='w', linewidth=0.5)
plt.scatter(y_test, test_pred, alpha=0.6, label='Test',
            marker='^', edgecolor='w', linewidth=0.5)
plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
plt.title('Combined Regression Plot', fontsize=14)
plt.xlabel('Actual (Normalized)', fontsize=12)
plt.ylabel('Predicted (Normalized)', fontsize=12)
plt.legend()

# Add metrics
combined_actual = np.concatenate([y_train, y_val, y_test])
combined_pred = np.concatenate([train_pred, val_pred, test_pred])
r2 = r2_score(combined_actual, combined_pred)
plt.text(0.05, 0.85, f'Overall R² = {r2:.3f}', transform=plt.gca().transAxes,
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()

# All data regression
plot_regression_one(Ra_s, all_pred, 'Complete Dataset Regression')

epoch_mse = pd.DataFrame({
    'Epoch': np.arange(1, len(train_loss)+1),
    'Train_MSE': train_loss,
    'Val_MSE': val_loss
})

# Add test MSE (constant across epochs)
epoch_mse['Test_MSE'] = test_loss

# Create combined fitness figure
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Training set
enhanced_regression_plot_one(y_train, train_pred, 'Training Set', ax=axs[0, 0])

# Validation set
enhanced_regression_plot_one(y_val, val_pred, 'Validation Set', ax=axs[0, 1])

# Test set
enhanced_regression_plot_one(y_test, test_pred, 'Test Set', ax=axs[1, 0])

# All data
enhanced_regression_plot_one(Ra_s, all_pred, 'Complete Dataset', ax=axs[1, 1])

plt.tight_layout()
end_time = time.time()  # Record end time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.4f} seconds")


# 633.3341 seconds
