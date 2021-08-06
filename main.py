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
    # Learning models
    for subject in subjects:
        fnames = glob('./data/train/subj%d_series*_data.csv' % (subject))
        raw = concatenate_raws([create_mne_raw_object(fname) for fname in fnames])

        raw = raw_preprocessing(raw)

        net = SimpleNet()
        train_net(net, raw, out_dir="./SimpleNet/Models/subject"+str(subject) +"/")

    # Results
    for subject in subjects:
        fnames =  glob('./SimpleNet/Models/subject%d/sn-epoch-*.pth' % (subject))

        test_filename = glob('./data/test/subj%d_series8_data.csv' % (subject))
        test = create_mne_raw_object(test_filename[0])
        test = raw_preprocessing(test)

        print("Test file:", test_filename)

        simplenet_output_dir = "./SimpleNet/Results/subject" + str(subject) + "/"

        if not os.path.exists(simplenet_output_dir):
            os.mkdir(simplenet_output_dir)

        for fname in fnames:
            print("Model file:", fname)

            out_filename = simplenet_output_dir + Path(fname).stem + '-result.pt'
            model = get_simple_net_model(fname)
            eeg_picks = pick_types(test.info, eeg=True)
            stim_picks = pick_types(test.info, stim=True)

            input = test._data[eeg_picks]
            output = test._data[stim_picks]

            max_steps = int(max_step(test))

            input = torch.randn(max_steps, 1, 32, 33)
            output = torch.randn(max_steps, 6)

            for step in range(max_steps):
                input[step] = get_psd_tensor(test, step)
                output[step] = get_output_tensor(test, step)

            out = model(input)

            torch.save(out, out_filename)



