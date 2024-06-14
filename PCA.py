# transform the tfr data to s/f components (as input to neural decoders)
# correspondance: cyj.sci@gmail.com

import matplotlib
matplotlib.use('TkAgg')
import os
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import mne
import pickle as pkl
import time

def get_subjects():
    bids_root = os.path.join('raw', 'batch_1', 'meg')
    files = os.listdir(bids_root)
    subjects_ex = ['CA123', 'CB999']
    sub_files = [f for f in files if f.startswith('sub-')]
    subjects = [s.split('-')[-1] for s in sub_files]
    for s in subjects_ex:
        subjects.remove(s)
    return subjects


def load_tfr(subject, condition):
    f_main = os.path.join('data_tfr', 'batch_1', 'sub-{}'.format(subject), 'ses-1')
    with open(os.path.join(f_main, 'tfr_{}.npy'.format(condition)), 'rb') as f:
        data = np.load(f)
        times = np.load(f)
        frex = np.load(f)
    chans = pd.read_csv(os.path.join(f_main, 'channels.csv'))
    return data, chans, frex, times


if False: # example of tfr data
    data, chans, frex, times = load_tfr('CA103', 'T')
    data = data[0,:,:,:] # one epoch/trial
    f, axes = plt.subplots(nrows=1, ncols = 10)
    for ci in range(10):
        axes[ci].pcolormesh(data[ci,:,:])
        axes[ci].set_axis_off()

def test_try_ipca_one_dataset():
    subject = 'CA103'
    condition = 'T'
    data, chans, frex, times = load_tfr(subject, condition)
    fi = 0
    tp = data[:, :, fi, :].copy()
    tp2 = transpose_X(tp)
    if False:
        ei = 2
        ci = 5
        npnt = len(times)
        np.array_equal(tp2[npnt*ei:npnt*(ei+1),ci], tp[ei,ci,:])

    transformer_0 = IncrementalPCA(n_components=10, batch_size=1000)
    transformer_0.fit(tp2)
    X_trans_0 = transformer_0.transform(tp2)

    transformer_1 = IncrementalPCA(n_components=10, batch_size=1000)
    for bi in range(64):
        transformer_1.partial_fit(tp2[bi*1000:(bi+1)*1000,:])
    X_trans_1 = transformer_1.transform(tp2)
    print(np.array_equal(X_trans_1, X_trans_0))


def transpose_X(X):
    # transpose X (n_epoch, n_chan, n_pnt) to X (n_pnt x n_epoch, n_chan)
    X = X.transpose([0,2,1])
    X_trans = []
    for feati in range(X.shape[-1]):
        X_trans.append(X[:,:,feati].flatten().copy())
    X_trans = np.array(X_trans).T
    if False:
        ei = 2
        ci = 5
        npnt = X.shape[1]
        print(np.array_equal(X_trans[npnt*ei:npnt*(ei+1),ci], X[ei,:,ci]))
    return X_trans


def ipca_frequency(freq):
    # running ipca with group data for each frequency
    t0 = time.time()
    subjects = get_subjects()
    conditions = ['T', 'R', 'N']
    _, chans, frex, times = load_tfr(subjects[0], conditions[0])
    fi = np.where(frex == freq)[0][0]
    tp_record = []
    counti = 0
    transformer = IncrementalPCA(n_components=10, batch_size=500)
    print('PCA decomposition for frequency ({}/{})...'.format(fi+1, len(frex)))
    for condi, c in enumerate(conditions):
        for si, s in enumerate(subjects):
            data,_,_,_ = load_tfr(s,c)
            X = transpose_X(data[:,:,fi,:])
            del data
            transformer.partial_fit(X)
            del X
            counti += 1
            tp_record.append([condi, c, si, s])
            df_record = pd.DataFrame(np.array(tp_record), columns = ['condi', 'condition', 'si', 'subject'])
            df_record.to_csv(os.path.join('results_pca', 'tp_record_{}_Hz.csv'.format(freq)), index=False)
            pkl.dump(transformer, open(os.path.join('results_pca', 'pca_{}_Hz'.format(freq)), 'wb'))
            print('{:3.2f}%'.format(counti / len(conditions) / len(subjects) * 100), end='\r')
    print('Done. Time cost {:3.2f}hrs'.format( (time.time()-t0) / 3600))
    if False:
        data, _, _, _ = load_tfr(s, c)
        X = transpose_X(data[:, :, 0, :])
        del data
        X_trans = transformer.transform(X[:1000,:])
        tp = pkl.load( open(os.path.join('results_pca', 'pca_{}_Hz'.format(freq)), 'rb'))
        tp2 = tp.transform(X[:1000,:])
        print(np.array_equal(X_trans, tp2))


if False:
    _, _, frex, _ = load_tfr('CA103', 'T')
    for fi, f in enumerate(frex):
        ipca_frequency(f)


def export_pc(subject):
    # export the & top 3 & components
    conditions = ['T', 'R', 'N']
    counti = 0
    t0 = time.time()
    for condi, c in enumerate(conditions):
        print('Exporting condition ({}/{})...'.format(condi+1, len(conditions)))
        print('Loading data...', end = '')
        data, _, frex, _ = load_tfr(subject,c)
        print('Done.')
        print('Transforming...', end = '')
        for fi, f in enumerate(frex):
            transformer = pkl.load(open(os.path.join('results_pca', 'pca_{}_Hz'.format(f)), 'rb'))
            X = data[:,:,fi,:].copy()
            X_pc = np.zeros((X.shape[0], transformer.explained_variance_ratio_.shape[0], X.shape[-1]))
            for ei in range(X_pc.shape[0]):
                X_pc[ei,:,:] = transformer.transform(X[ei,:,:].T).T
            if fi == 0:
                data_pc = X_pc[:,:3,:].copy()  # top three pcs
            else:
                data_pc = np.concatenate([data_pc, X_pc[:,:3,:].copy()], axis = 1)  # concatenate along fi/pci
            del X_pc
            counti += 1
        del data
        print('Done.')
        print('Saving...', end='')
        f_main = os.path.join('data_pc', 'batch_1', 'sub-{}'.format(subject), 'ses-1')
        if not os.path.exists(f_main): os.makedirs(f_main)
        with open(os.path.join(f_main, 'pc_{}.npy'.format(c)), 'wb') as f:
            np.save(f, data_pc)
        if False:
            with open(os.path.join(f_main, 'pc_{}.npy'.format(c)), 'rb') as f:
                tp = np.load(f)
            print(np.array_equal(tp, data_pc))
        del data_pc
        print('Done.')
    print('Total time cost {:3.2f}hrs'.format((time.time() - t0) / 3600))


if False:
    subjects = get_subjects()
    for si, s in enumerate(subjects):
        print('Export pc data for subject ({}/{})...'.format(si+1, len(subjects)))
        export_pc(s)


###  when having the feature selection results   ###

def export_selected_features():
    df = pd.read_csv(os.path.join('results_idv_decoder', 'batch_1', 'feature_idx.csv'))
    fidx = df.fid.tolist()
    _, _, frex_raw, _ = load_tfr('CA103', 'T')
    frex = []
    pc = []
    for fi, fid in enumerate(fidx):
        frex.append(frex_raw[np.floor(fid/3).astype(int)])
        pc.append(np.mod(fid,3))
    df = pd.DataFrame({'fid': fidx,
                       'frequency': frex,
                       'pci': pc})
    df.to_csv(os.path.join('results_idv_decoder', 'batch_1', 'features.csv'), index=False)
    return None


def plot_freq_pc(frequency, pci, return_loading = False):
    transformer = pkl.load(open(os.path.join('results_pca', 'pca_{:1.1f}_Hz'.format(frequency)), 'rb'))
    coefs = transformer.components_.copy()[pci,:]

    info = pkl.load(open('channel_info.pkl', 'rb'))

    if not return_loading:
        matplotlib.rcParams.update({'font.size': 18})
        f, ax = plt.subplots()
        img, _ = mne.viz.plot_topomap(data=coefs, pos=info, axes=ax)
        cbar = plt.colorbar(ax=ax, shrink=0.8, orientation='vertical', mappable=img)
        #cbar.set_label('PC loading')
    else:
        return coefs, info


def plot_feature_loadings():
    feats = pd.read_csv(os.path.join('results_idv_decoder', 'batch_1', 'features.csv'))

    nrows = 10
    ncols = 6
    matplotlib.rcParams.update({'font.size': 14})
    f, axes = plt.subplots(nrows=nrows, ncols=ncols)
    for fi in range(feats.shape[0]):
        freq = feats.frequency[fi]
        pci = feats.pci[fi]
        ri = np.floor(fi/ncols).astype(int)
        ci = np.mod(fi,ncols)
        ax = axes[ri,ci]
        loading, info = plot_freq_pc(freq, pci, return_loading=True)
        img,_ = mne.viz.plot_topomap(data=loading, pos=info, axes=ax)
        cbar = plt.colorbar(ax=ax, shrink=0.8, orientation='vertical', mappable=img)
        ax.set_title('{}Hz PC {}'.format(freq, pci+1))
    while fi < nrows * ncols - 1:
        fi += 1
        ri = np.floor(fi/ncols).astype(int)
        ci = np.mod(fi,ncols)
        ax = axes[ri,ci]
        ax.set_axis_off()




