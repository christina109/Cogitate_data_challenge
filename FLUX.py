# MEG preprocessing using FLUX pipeline
# adapted from: https://github.com/Neuronal-Oscillations/FLUX/tree/main
# correspondance: cyj.sci@gmail.com

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import scipy as sp
import os
import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import annotate_muscle_zscore, ICA
import matplotlib.pyplot as plt
from shutil import copy2  # preserve the timestamp
import pickle as pkl


# path management
bids_root  = os.path.join('raw', 'batch_1', 'meg')
deriv_root = os.path.join('derivatives', 'batch_1')
meg_suffix = 'meg'
max_suffix = 'raw_sss'
ann_suffix = 'ann'
ica_suffix = 'ica'
epo_suffix = 'epo'

# parameters
srate = 200         # downsample when ICA
low_hz = 1          # high pass
high_hz = 40        # low pass
epo_win = [-0.2, 2]

# Time-frequency analysis
freqs = np.arange(2, 40, 1)
# n_cycles = freqs / 2
n_cycles = np.linspace(3, 7, freqs.shape[0])
time_bandwidth = 2.0

# test only
subject = 'CA103'
session = '1'
task    = 'dur'
run     = '01'
runs    = ['01', '02', '03', '04', '05']


def get_subjects():
    files = os.listdir(bids_root)
    subjects_ex = ['CA123', 'CB999']
    sub_files = [f for f in files if f.startswith('sub-')]
    subjects = [s.split('-')[-1] for s in sub_files]
    for s in subjects_ex:
        subjects.remove(s)
    return subjects


if False:
    subjects = get_subjects()
    for si, s in enumerate(subjects):
        print('Processing data ({}/{})'.format(si+1, len(subjects)))
        max_filter(s, session, task, runs)


def max_filter(subject, session, task, runs):
    # flag bad channels
    # max fitler

    noisy_chs = []
    flat_chs  = []
    bad_chs   = []
    data_list = []
    deriv_file_list = []
    for ri, run in enumerate(runs):
        print('Reading data ({}/{})...'.format(ri+1, len(runs)))
        bids_path  = BIDSPath(subject=subject, session=session, task=task, run=run,
                             suffix=meg_suffix, datatype='meg', extension='.fif', root=bids_root)

        deriv_path = BIDSPath(subject=subject, session=session, task=task, run=run,
                              suffix=max_suffix, datatype='meg', extension='.fif',
                              root=os.path.join(deriv_root, 'Preprocessing'), check=False).mkdir()

        deriv_fname = bids_path.basename.replace(meg_suffix, max_suffix)  # output filename
        deriv_file = os.path.join(deriv_path.directory, deriv_fname)
        deriv_file_list.append(deriv_file)

        crosstalk_file = bids_path.meg_crosstalk_fpath
        calibration_file = bids_path.meg_calibration_fpath
        #print(calibration_file)

        data = read_raw_bids(bids_path=bids_path,
                             extra_params={'preload':True},
                             verbose=True)
        bad_chs.extend(data.info['bads']) # some dataset has bad channels marked in the raw file. e.g, CA123

        # identify the faulty sensors
        data_list.append(data)
        data_check = data.copy()
        print('Detecting noisy channels...', end = '')
        auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(data_check,
                                                                                                 cross_talk=crosstalk_file,
                                                                                                 calibration=calibration_file,
                                                                                                 return_scores=True,
                                                                                                 verbose=False)
        print('Done.')
        noisy_chs.extend(auto_noisy_chs)
        flat_chs.extend(auto_flat_chs)
        del data, data_check

    # drop duplicates
    bad_chs   = list(dict.fromkeys(bad_chs))
    noisy_chs = list(dict.fromkeys(noisy_chs))
    flat_chs  = list(dict.fromkeys(flat_chs))
    print('raw bad = ', bad_chs)
    print('noisy =', noisy_chs)
    print('flat =', flat_chs)

    # plot time course of a few sensors
    if False:
        data_tmp = data.copy()
        data_tmp.pick(["MEG2311", "MEG2321" ])
        data_tmp.plot(proj = False)

    # flag bad sensors
    for ri in range(len(runs)):
        data = data_list[ri].copy()
        data.info['bads'] = []
        data.info['bads'].extend(noisy_chs + flat_chs + bad_chs)
        print('bads =', data.info['bads'])

        # Change MEGIN magnetometer coil types (type 3022 and 3023 to 3024) to ensure compatibility across systems.
        data.fix_mag_coil_types()

        # pick one run as a “reference” run and then use Maxwell filter to transform the head positions of the other runs
        if ri == 0:
            destination = data.info["dev_head_t"]
        # Apply the Maxfilter and calibration
        print('Applying max filter ({}/{})...'.format(ri+1, len(runs)), end='')
        data_sss = mne.preprocessing.maxwell_filter(data,
                                                    destination = destination,
                                                    cross_talk=crosstalk_file,
                                                    calibration=calibration_file,
                                                    verbose=False)
        print('Done.')

        # compare data before and after noise reduction
        if False:
            data1.compute_psd(fmax=60, n_fft=1000).plot()
            data1_sss.compute_psd(fmax=60, n_fft=1000).plot()

        if False:
            data1_sss.info['subject_info']['weight'] = None
            data1_sss.info['subject_info']['height'] = None
        print(data_sss.info['bads'])
        data_sss.save(deriv_file_list[ri], overwrite=True)
        if False:
            bids_path = BIDSPath(subject=subject, session=session, task=task, run=runs[ri],
                                 suffix=max_suffix, datatype='meg', extension='.fif',
                                 root=os.path.join(deriv_root, 'Preprocessing'), check=False)
            sss_reload = read_raw_bids(bids_path=bids_path,
                                       extra_params={'preload': True},
                                       verbose=False)
            print(sss_reload.info['bads'])  # note for CA123, reloaded data still have the bad channels?
        del data_sss
    del data_list


if False:
    subjects = ['CB999']
    for si, s in enumerate(subjects):
        print('Processing data ({}/{})'.format(si+1, len(subjects)))
        for ri, run in enumerate(runs):
            artifact_annotation(s, session, task, run)


def artifact_annotation(subject, session, task, run):

    bids_path = BIDSPath(subject=subject, session=session, task=task, run=run,
                         suffix=max_suffix, datatype='meg', extension='.fif',
                         root=os.path.join(deriv_root, 'Preprocessing'), check=False)
    ann_path  = BIDSPath(subject=subject, session=session, task=task, run=run,
                         suffix=ann_suffix, datatype='meg', extension='.fif',
                         root=os.path.join(deriv_root, 'Preprocessing'), check=False)

    ann_fname = bids_path.basename.replace(max_suffix, ann_suffix) # output filename
    ann_fname = ann_fname.replace('fif', 'csv')                    # output filename
    ann_fname = os.path.join(deriv_root, 'Preprocessing', 'sub-{}'.format(subject), 'ses-{}'.format(session), 'meg', ann_fname)

    # requires _channels.ts, _events.ts, _meg.json in the bids_path
    #           participants.tsc, participants.json in the root
    for suffix in ['channels.tsv', 'events.json', 'events.tsv', 'meg.json']:
        f_copy = 'sub-{}_ses-{}_task-{}_run-{}_{}'.format(subject, session, task, run, suffix)
        src = os.path.join( bids_root, 'sub-{}'.format(subject), 'ses-{}'.format(session), 'meg', f_copy )
        dst = os.path.join( deriv_root, 'Preprocessing', 'sub-{}'.format(subject), 'ses-{}'.format(session), 'meg', f_copy )
        copy2(src, dst)

    data = read_raw_bids(bids_path=bids_path,
                         extra_params={'preload':True},
                         verbose=False)
    #print(data.info['bads'])

    # finding muscle artifacts
    threshold_muscle = 10
    annotations_muscle, scores_muscle = annotate_muscle_zscore(data, ch_type="mag", threshold=threshold_muscle,
                                                               min_length_good=0.2, filter_freq=[110, 140])
    data.set_annotations(annotations_muscle)

    # Save the artifact annotations
    data.annotations.save(ann_fname, overwrite=True)
    data.save(ann_path, overwrite=True)
    #print(data.info['bads'])


def run_ica(subject, session, task, runs):

    # Resampling and filtering of the raw dat
    for ri, run in enumerate(runs):
        print('Loading data ({}/{})...'.format(ri+1, len(runs)), end = '')
        bids_path = BIDSPath(subject=subject, session=session, task=task, run=run,
                             suffix=ann_suffix, datatype='meg', extension='.fif',
                             root=os.path.join(deriv_root, 'Preprocessing'), check=False)

        raw = read_raw_bids(bids_path=bids_path,
                            extra_params={'preload':True},
                            verbose=False)
        #print(raw.info['bads'])
        raw_resmpl = raw.copy().pick('meg')
        raw_resmpl.resample(srate)
        raw_resmpl.filter(low_hz, high_hz)
        if ri == 0:
            raw_resmpl_all = mne.io.concatenate_raws([raw_resmpl])
        else:
            raw_resmpl_all = mne.io.concatenate_raws([raw_resmpl_all, raw_resmpl])
        print('Done.')
    del raw_resmpl

    # Apply ICA
    ica = ICA(method='fastica', random_state=97, n_components=30, verbose=True)
    ica.fit(raw_resmpl_all, verbose=True)

    return ica, raw_resmpl_all, runs


if False:
    import matplotlib
    matplotlib.use('TkAgg')  # shift back between tk and qt to solve the "fig unable to show" problem
    from code_pyfile.FLUX import *
    subjects = get_subjects()
    subject = subjects[0]  # update the number
    print('Subject {}...'.format(subject))
    ica, raw_resample_all, runs = run_ica(subject, session, task, runs)

    plot_components(ica)

    plot_sources(ica, raw_resample_all)

    ics_ex = [0,2,16]  # change according to inspection
    ica = update_ica(ica, ics_ex)
    for ri, run in enumerate(runs):
        if ri == 0:
            remove_ics(ica, subject, session, task, run, plotOn=True)
        else:
            remove_ics(ica, subject, session, task, run, plotOn=False)


def plot_sources(ica, raw_resmpl_all):
    ica.plot_sources(raw_resmpl_all, title='ICA')

def plot_components(ica):
    ica.plot_components()

def update_ica(ica, ics_ex):
    print('Will exclude IC {}'.format(ics_ex))
    ica.exclude = ics_ex
    return ica

def remove_ics(ica, subject, session, task, run, plotOn = True):

    bids_path = BIDSPath(subject=subject, session=session, task=task, run=run,
                         suffix=ann_suffix, datatype='meg', extension='.fif',
                         root=os.path.join(deriv_root, 'Preprocessing'), check=False)

    deriv_path = BIDSPath(subject=subject, session=session, task=task, run=run,
                          suffix=ica_suffix, datatype='meg', extension='.fif',
                          root=os.path.join(deriv_root, 'Preprocessing'), check=False)

    deriv_fname = bids_path.basename.replace(ann_suffix, ica_suffix)  # output filename
    deriv_file = os.path.join(deriv_path.directory, deriv_fname)

    raw = read_raw_bids(bids_path=bids_path,
                        extra_params={'preload':True},
                        verbose=True)
    raw_ica = raw.copy()
    raw_ica = ica.apply(raw_ica)

    raw_ica.save(deriv_file, overwrite=True)

    # Plotting the data to check the artifact reduction
    if plotOn:
        chs = ['MEG0311', 'MEG0121', 'MEG1211', 'MEG1411']
        chan_idxs = [raw.ch_names.index(ch) for ch in chs]
        raw.plot(order=chan_idxs, duration=5, title='before')
        raw_ica.plot(order=chan_idxs, duration=5, title='after')

if False:
    subjects = get_subjects()
    for si, s in enumerate(subjects):
        print('Extracting trials ({}/{})...'.format(si+1, len(subjects)))
        extract_trials(s, session, task)

def extract_trials(subject, session, task):

    raw_list = []
    events_list = []
    for ri, run in enumerate(runs):
        print('Loading data ({}/{})...'.format(ri + 1, len(runs)))
        bids_path_preproc = BIDSPath(subject=subject, session=session,
                                     task=task, run=run, suffix=ica_suffix, datatype='meg',
                                     root=os.path.join(deriv_root, 'Preprocessing'), extension='.fif', check=False)

        bids_path = BIDSPath(subject=subject, session=session,
                             task=task, run=run, suffix=epo_suffix, datatype='meg',
                             root=os.path.join(deriv_root, 'Analysis'), extension='.fif', check=False).mkdir()
        if ri == 0:
            deriv_file = bids_path.basename.replace('run-01', 'run-99')  # run-99 the concatenated runs
            deriv_fname = os.path.join(bids_path.directory, deriv_file)
            print(deriv_fname)

        # reading the events from the stimulus channels
        raw = read_raw_bids(bids_path=bids_path_preproc,
                            extra_params={'preload': False},
                            verbose=True)
        ann_fname = bids_path_preproc.basename.replace(ica_suffix, ann_suffix)  # output filename
        ann_fname = ann_fname.replace('fif', 'csv')  # output filename
        ann_fname = os.path.join(deriv_root, 'Preprocessing', 'sub-{}'.format(subject), 'ses-{}'.format(session), 'meg',
                                 ann_fname)

        # Reading the events from the  raw file
        events, events_id = mne.events_from_annotations(raw, event_id='auto')

        ann = mne.read_annotations(ann_fname)
        print(ann)
        raw.set_annotations(ann)

        raw_list.append(raw)
        events_list.append(events)

    events_picks_id = {k: v for k, v in events_id.items() if k.startswith('task') or k.endswith('ms')}

    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    del raw_list
    if False:
        raw.plot(start=50)

    # Set the peak-to-peak amplitude thresholds for trial rejection.
    # These values may change depending on the quality of the data.
    reject = dict(grad=5000e-13,  # unit: T / m (gradiometers)
                  mag=4e-12,     # unit: T (magnetometers)
                  # eeg=40e-6,      # unit: V (EEG channels)
                  # eog=250e-6      # unit: V (EOG channels)
                  )

    # Make epochs (4.5 seconds centered on stim onset)
    epochs = mne.Epochs(raw,
                        events, events_picks_id,
                        tmin=epo_win[0], tmax=epo_win[1],
                        baseline=None,
                        proj=False,
                        picks='all',
                        detrend=1,
                        reject=reject,
                        reject_by_annotation=True,
                        preload=True,
                        verbose=True)
    #epochs.plot_drop_log()

    epochs.save(deriv_fname, overwrite=True)


if False:
    subjects = get_subjects()
    for si, s in enumerate(subjects):
        print('Export TF data ({}/{})...'.format(si+1, len(subjects)))
        time_frequency(s, session, task)


def time_frequency(subject, session, task):

    run = '99'
    bids_path = BIDSPath(subject=subject, session=session,
                         task=task, run=run, suffix=epo_suffix, datatype='meg',
                         root=os.path.join(deriv_root, 'Analysis'), extension='.fif', check=False)
    epochs = mne.read_epochs(bids_path.fpath,
                             proj = False,
                             preload=True,
                             verbose=True)

    conditions = ['task relevant target', 'task relevant non target', 'task irrelevant']
    conds  = ['T', 'R', 'N']
    for ci, condition in enumerate(conditions):
        print('Time-frequency for {}...'.format(condition))
        deriv_path = BIDSPath(subject=subject, session=session,
                              task=task, run=run, suffix='tf_{}'.format(conds[ci]), datatype='meg',
                              root=os.path.join(deriv_root, 'Analysis'), extension='.fif', check=False)
        tf = mne.time_frequency.tfr_multitaper(epochs[condition],
                                               freqs=freqs,
                                               n_cycles=n_cycles,
                                               time_bandwidth=time_bandwidth,
                                               picks='mag',  # picks=None, # all channels
                                               use_fft=True,
                                               return_itc=False,
                                               average=False,
                                               decim=2,
                                               n_jobs=-1,
                                               verbose=True)
        # tf.data (n_epoch, n_chan, n_freq, n_pnt)

        # output tfr
        f_main = os.path.join('data_tfr', 'batch_1', 'sub-{}'.format(subject), 'ses-{}'.format(session))
        if not os.path.exists(f_main): os.makedirs(f_main)
        print('Saving...', end = '')
        with open(os.path.join(f_main, 'tfr_{}.npy'.format(conds[ci])), 'wb') as f:
            np.save(f, tf.data)
            np.save(f, tf.times)
            np.save(f, tf.freqs)
        if ci == 0:
            chans = pd.DataFrame({'channel': tf.ch_names,
                                  'type': tf.get_channel_types()})
            chans.to_csv(os.path.join(f_main, 'channels.csv'), index = False)
        print('Done.')
        if False:
            with open(os.path.join(f_main, 'tfr_{}.npy'.format(conds[ci])), 'rb') as f:
                data = np.load(f)
                times = np.load(f)
                frex = np.load(f)
            chans = pd.read_csv(os.path.join(f_main, 'channels.csv'))
            print( np.array_equal(tf.data, data) )
            print( np.array_equal(tf.times, times) )
            print( np.array_equal(tf.freqs, frex) )
        del tf
    del epochs






