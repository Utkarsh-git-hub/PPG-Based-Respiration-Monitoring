import torch
import numpy as np
import os
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from scipy.signal import resample


data_dir = '/content/drive/MyDrive/bidmc_csv'
num_patients = 53


all_ppg_data = []
all_resp_data = []


for patient_id in range(1, num_patients + 1):

    file_path = os.path.join(data_dir, f'bidmc_{patient_id:02d}_Signals.csv')


    signal_data = pd.read_csv(file_path)


    all_ppg_data.append(signal_data[' PLETH'])
    all_resp_data.append(signal_data[' RESP'])

all_ppg_data = pd.concat(all_ppg_data)
all_resp_data = pd.concat(all_resp_data)

train_ppg, test_ppg = train_test_split(all_ppg_data, test_size=0.2, random_state=42)
train_resp, test_resp = train_test_split(all_resp_data, test_size=0.2, random_state=42)

print(np.shape(all_resp_data))

#data downsampling

desired_sample_rate = 30
original_sample_rate = 125


resampling_factor = desired_sample_rate / original_sample_rate

train_ppg = resample(train_ppg, int(len(train_ppg) * resampling_factor))
test_ppg = resample(test_ppg, int(len(test_ppg) * resampling_factor))
train_resp = resample(train_resp, int(len(train_resp) * resampling_factor))
test_resp = resample(test_resp, int(len(test_resp) * resampling_factor))

#Normalization

train_ppg = (train_ppg - np.min(train_ppg)) / (np.max(train_ppg) - np.min(train_ppg))
test_ppg = (test_ppg - np.min(test_ppg)) / (np.max(test_ppg) - np.min(test_ppg))
train_resp = (train_resp - np.min(train_resp)) / (np.max(train_resp) - np.min(train_resp))
test_resp = (test_resp - np.min(test_resp)) / (np.max(test_resp) - np.min(test_resp))
print(np.shape(train_ppg))

window_size = 30 * desired_sample_rate

# For train_ppg
frames_train_ppg = len(train_ppg) // window_size
train_ppg_reshaped = train_ppg[:frames_train_ppg * window_size].reshape((frames_train_ppg, window_size, 1))
train_ppg_reshaped = (train_ppg_reshaped - np.min(train_ppg_reshaped)) / (np.max(train_ppg_reshaped) - np.min(train_ppg_reshaped))

# For test_ppg
frames_test_ppg = len(test_ppg) // window_size
test_ppg_reshaped = test_ppg[:frames_test_ppg * window_size].reshape((frames_test_ppg, window_size, 1))
test_ppg_reshaped = (test_ppg_reshaped - np.min(test_ppg_reshaped)) / (np.max(test_ppg_reshaped) - np.min(test_ppg_reshaped))

# For train_resp
frames_train_resp = len(train_resp) // window_size
train_resp_reshaped = train_resp[:frames_train_resp * window_size].reshape((frames_train_resp, window_size, 1))
train_resp_reshaped = (train_resp_reshaped - np.min(train_resp_reshaped)) / (np.max(train_resp_reshaped) - np.min(train_resp_reshaped))

# For test_resp
frames_test_resp = len(test_resp) // window_size
test_resp_reshaped = test_resp[:frames_test_resp * window_size].reshape((frames_test_resp, window_size, 1))
test_resp_reshaped = (test_resp_reshaped - np.min(test_resp_reshaped)) / (np.max(test_resp_reshaped) - np.min(test_resp_reshaped))

