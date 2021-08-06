from mne.time_frequency import psd_welch
from mne import pick_types

import torch

import numpy as np

from scipy.signal import butter
from scipy.signal import lfilter
from joblib import Parallel, delayed


freqs = [7, 30]
b,a = butter(5,np.array(freqs)/250.0,btype='bandpass')

def raw_preprocessing(raw):
    print("Raw_preprocessing()")
    picks = pick_types(raw.info, eeg=True)
    raw._data[picks] = np.array(Parallel(n_jobs=-1)(delayed(lfilter)(b, a, raw._data[i]) for i in picks))
    return raw


def max_step(raw, time_step=1/500, window_size=0.256):
    points_num = raw.n_times
    points_num = points_num - window_size*500 # windows_size[s] times frequency Hz
    return points_num


def get_psd_tensor(raw, step, time_step=1/500, window_size=0.256):
    picks = pick_types(raw.info, eeg=True)
    psds, freq = psd_welch(raw, n_fft=64,
                           tmin=step*time_step, tmax=step*time_step+window_size,
                           verbose=False, picks=picks)

    # psds = psds - np.mean(psds)
    # psds = psds/np.std(psds)
    tensor = torch.randn(1, 32, 33)
    tensor[0] = torch.from_numpy(psds)
    return tensor

def get_output_tensor(raw, step):
    picks = pick_types(raw.info, stim=True)
    return torch.from_numpy(raw._data[picks][:, step])
