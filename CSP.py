from glob import glob
import pandas as pd
import numpy as np

import os

import mne
from mne import create_info
from mne import concatenate_raws
from mne import find_events
from mne import Epochs
from mne import pick_types
from mne.io import RawArray
from mne.io import read_raw
from mne.channels import make_standard_montage
from mne.epochs import concatenate_epochs
from mne.decoding import CSP
from mne.time_frequency import psd_welch

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from pathlib import Path


from scipy.signal import convolve
from scipy.signal import boxcar

from joblib import Parallel, delayed

# To plot in separate window - not in Pycharm's SciView
import matplotlib
matplotlib.use('Qt5Agg')

from networks import SimpleNet
from networks import max_batch_num
from networks import get_batch
from networks import train_net
from networks import get_simple_net_model

import torch

from utils.utils import get_psd_tensor
from utils.utils import get_output_tensor
from utils.utils import raw_preprocessing
from utils.utils import max_step

from scipy.signal import butter, lfilter, convolve, boxcar

submission_file = 'beat_the_benchmark.csv'

subsample = 10

subjects = range(1,13)
# design a butterworth bandpass filter
freqs = [7, 30]
b,a = butter(5,np.array(freqs)/250.0,btype='bandpass')

# CSP parameters
# Number of spatial filter to use
nfilters = 5

# window for smoothing features
nwin = 250

cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

from sklearn.metrics import accuracy_score
from sklearn import metrics

def create_mne_raw_object(fname, read_events=True):
    print("Create_mne_raw_object...")
    print("Filename:", fname)

    data = pd.read_csv(fname)

    ch_names = list(data.columns[1:])

    ch_type = ['eeg']*len(ch_names)
    data = 1e-6*np.array(data[ch_names]).T

    if read_events:
        ev_fname = fname.replace('_data', '_events')

        events = pd.read_csv(ev_fname)
        events_names = events.columns[1:]
        events_data = np.array(events[events_names]).T

        ch_type.extend(['stim']*6)
        ch_names.extend(events_names)

        data = np.concatenate((data, events_data))

    info = create_info(ch_names, sfreq=500.0, ch_types=ch_type)

    raw = RawArray(data, info, verbose=False)

    return raw

if __name__ == "__main__":

    for subject in subjects:
        epochs_tot = []
        y = []

        fnames = glob('./data/train/subj%d_series*_data.csv' % (subject))


        raw = concatenate_raws([create_mne_raw_object(fname) for fname in fnames])

        picks = pick_types(raw.info, eeg=True)
        raw._data[picks] = np.array(Parallel(n_jobs=-1)(delayed(lfilter)(b, a, raw._data[i]) for i in picks))


        events = find_events(raw, stim_channel='HandStart', verbose=False)

        epochs = Epochs(raw, events, {'during': 1}, 0, 2, proj=False,
                        picks=picks, baseline=None, preload=True, verbose=False)

        epochs_tot.append(epochs)
        y.extend([1] * len(epochs))

        epochs_rest = Epochs(raw, events, {'before': 1}, -2, 0, proj=False,
                             picks=picks, baseline=None, preload=True, verbose=False)

        epochs_rest.shift_time(2.0)

        y.extend([-1] * len(epochs_rest))
        epochs_tot.append(epochs_rest)

        epochs = concatenate_epochs(epochs_tot)

        X = epochs.get_data()
        y = np.array(y)

        csp = CSP(n_components=nfilters, reg='shrunk')
        csp.fit(X, y)

        feat = np.dot(csp.filters_[0:nfilters], raw._data[picks]) ** 2

        feattr = np.array(
            Parallel(n_jobs=-1)(delayed(convolve)(feat[i], boxcar(nwin), 'full') for i in range(nfilters)))
        feattr = np.log(feattr[:, 0:feat.shape[1]])
        print(feattr.shape)

        labels = raw._data[32:]

        fnames = glob('./data/test/subj%d_series*_data.csv' % (subject))
        raw = concatenate_raws([create_mne_raw_object(fname, read_events=False) for fname in fnames])
        raw._data[picks] = np.array(Parallel(n_jobs=-1)(delayed(lfilter)(b, a, raw._data[i]) for i in picks))

        ids = np.concatenate([np.array(pd.read_csv(fname)['id']) for fname in fnames])

        feat = np.dot(csp.filters_[0:nfilters], raw._data[picks]) ** 2
        featte = np.array(
            Parallel(n_jobs=-1)(delayed(convolve)(feat[i], boxcar(nwin), 'full') for i in range(nfilters)))
        featte = np.log(featte[:, 0:feat.shape[1]])

        print("Len: ", len(featte))

        lr = LogisticRegression()
        pred = np.empty((len(ids), 6))
        for i in range(6):
            print('Train subject %d, class %s' % (subject, cols[i]))
            lr.fit(feattr[:, ::subsample].T, labels[i, ::subsample])
            pred[:, i] = lr.predict_proba(featte.T)[:, 1]

        subject_prediction = pd.DataFrame(index = ids,
                                          columns= cols,
                                          data=pred)

        subject_prediction.to_csv('./CSP/results/subject' + str(subject) + '.csv')

    results_tot = []
    for subject in subjects:
        pred = pd.read_csv('./CSP/results/subject'+str(subject)+'.csv')
        true = pd.read_csv('./data/test/subj'+str(subject)+'_series8_events.csv')
        results = []
        for col in cols:
            results.append(metrics.roc_auc_score(true[col].values, pred[col].values))

        print("Subject:", subject)
        print(results)

        results_tot.append(results)

    print(np.concatenate(results_tot))