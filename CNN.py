# Training neural decoders
# Correspondence: Christina Jin (cyj.sci@gmail.com)
# tf==2.8 or keras==2.6, tf-gpu==2.6
# numpy==1.19.5 pandas==1.3.4 scipy==1.10

# device number
gpu2use = 0 # -1 for cpu

import sys
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu2use)
#os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'


from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.regularizers import l2, l1
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import scipy as sp
import os
import time
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
sns.set_theme(style="white", font_scale=2)
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import reset_default_graph
from multiprocessing import Process, Manager
from matplotlib import animation


# randomization
randseed = 1234

# cross validation
n_split = 5

# lower neurons/batch size if out of memory
# hyperparameters
if True: # idv decoder
    batch_size = 50
    n_epoch = 100
else:
    batch_size = 100
    n_epoch = 100


batch_norm = True
dropout = 0.2  # 0 if off

neurons_t   = 16
neurons_sf  = 32
neurons_cnn = [32,32,64,64]
neurons_fc  = [200, 20]
n_class = 3

lr = 5e-4
reg_type = 'l2'
reg_val = 0.01


def get_condition_code():
    return {'T': 2, 'R': 1, 'N': 0}

def get_subjects():
    bids_root = os.path.join('raw', 'batch_1', 'meg')
    files = os.listdir(bids_root)
    subjects_ex = ['CA123', 'CB999']
    sub_files = [f for f in files if f.startswith('sub-')]
    subjects = [s.split('-')[-1] for s in sub_files]
    for s in subjects_ex:
        subjects.remove(s)
    return subjects


def load_data(subject, condition):
    f_main = os.path.join('data_pc', 'batch_1', 'sub-{}'.format(subject), 'ses-1')
    with open(os.path.join(f_main, 'pc_{}.npy'.format(condition)), 'rb') as f:
        data = np.load(f)
    return data


if False: # example of input
    X = load_data('CA103', 'T')
    f = plt.figure()
    plt.pcolormesh(X[10,:,:])



def load_dataset(subject):
    conditions = get_condition_code()
    for ci, c in enumerate(['T', 'R', 'N']):
        tp = load_data(subject, c)
        #print(tp.shape[0])
        if ci == 0:
            X = tp.copy()
            y = [conditions[c]]*tp.shape[0]
        else:
            X = np.concatenate([X, tp], axis=0)
            y.extend([conditions[c]]*tp.shape[0])
        del tp
    return X, np.array(y)
    # X(n, s/f, t)


###   modelling   ###

def encode_y(y, target=None):
    if target is None:
        y_trans = np.zeros([y.shape[0], n_class])
        for c in range(n_class):
            y_trans[y==c,c] = 1
    else:
        y_trans = np.zeros([y.shape[0], 2])
        y_trans[:,0] = 1
        y_trans[y==target,0] = 0
        y_trans[y==target,1] = 1
    return y_trans


def normalize_X(X):
    return (X - X.mean())/X.std()

def scaling_X(X, level = 'dataset'):
    if level == 'sf':
        X_min = X.min(axis=(0, 2), keepdims=True)
        X_max = X.max(axis=(0, 2), keepdims=True)
        X_norm = (X-X_min)/(X_max-X_min)
    else:  # whole dataset
        X_norm = (X - X.min())/ (X.max()-X.min())
    if False:
        print(X_norm.min())
        print(X_norm.max())
    return X_norm

def check_folder(f_folder, clear_folder = False):
    if not os.path.exists(f_folder):
        os.makedirs(f_folder)
    else:
        if clear_folder:
            for f in os.listdir(f_folder):
                os.remove(os.path.join(f_folder, f))
    return None

def save_model_hist(model, hist, f_hist, cv_i):
    if model is not None:
        model.save(os.path.join(f_hist, 'model_lnpo{}.h5'.format(cv_i)))
        print('Model saved.')
    pd.DataFrame(hist.history).to_csv(os.path.join(f_hist, 'history_lnpo{}.csv').format(cv_i), index=False)
    print('History saved.')
    return None

def export_cv_perf(f_save_main):
    loss_train = []
    loss_val = []
    acc_train = []
    acc_val = []
    for cvi in range(n_split):
        tp = pd.read_csv(os.path.join(f_save_main, 'history_lnpo{}.csv'.format(cvi)))
        loss_train.append(tp['loss'].tolist()[-1])
        loss_val.append(tp['val_loss'].tolist()[-1])
        acc_train.append(tp['categorical_accuracy'].tolist()[-1])
        acc_val.append(tp['val_categorical_accuracy'].tolist()[-1])
    df = pd.DataFrame({'loss_train':  loss_train,
                       'loss_val': loss_val,
                       'acc_train': acc_train,
                       'acc_val': acc_val,
                       'cvi': range(n_split)})
    df.to_csv(os.path.join(f_save_main, 'perf_cv.csv'), index=False)
    return None


def cnn_decoder(X_train, y_train, X_val, y_val,
                f_save_main, cvi, save_model = True,
                hypar = None, hypar_val = None, verbose = 1):

    if y_train.ndim == 1:
        y_train = encode_y(y_train)
    if y_val is not None and y_val.ndim == 1:
        y_val = encode_y(y_val)

    if hypar == 'reg_val':
        if reg_type == 'l1':
            reg = l1(hypar_val)
        elif reg_type == 'l2':
            reg = l2(hypar_val)
        else:
            reg = None
    else:
        if reg_type == 'l1':
            reg = l1(reg_val)
        elif reg_type == 'l2':
            reg = l2(reg_val)
        else:
            reg = None

    # model design
    keras.backend.clear_session()

    inputs = Input(shape = X_train.shape[1:], name = 'input')

    hidden_layers = Sequential(name='hidden')
    hidden_layers.add(Conv2D(neurons_t, (1, 5),  strides = (1,2),                         # 1000/200*5=25ms
                             activation='relu', kernel_regularizer=reg,
                             name='conv_t'))
    if batch_norm: hidden_layers.add(BatchNormalization())
    if dropout > 0: hidden_layers.add(Dropout(dropout))
    hidden_layers.add(Conv2D(neurons_sf, (X_train.shape[1], 1),
                             activation='relu', kernel_regularizer=reg,
                             name='conv_sf'))
    if batch_norm: hidden_layers.add(BatchNormalization())
    if dropout > 0: hidden_layers.add(Dropout(dropout))

    for i, n_neuron in enumerate(neurons_cnn):
        for j in range(2): # layers per block
            hidden_layers.add(Conv2D(n_neuron, (1,5),
                                     activation = 'relu', kernel_regularizer = reg,
                                     name = 'conv_{}_{}'.format(i,j)))
            if batch_norm: hidden_layers.add(BatchNormalization())
            if dropout > 0: hidden_layers.add(Dropout(dropout))
        hidden_layers.add(MaxPooling2D(pool_size=(1,2),name='pool{}'.format(i)))

    hidden_layers.add(Flatten())
    for i, n_neuron in enumerate(neurons_fc):
        hidden_layers.add(Dense(n_neuron, activation='ReLU', kernel_regularizer=reg, name='fc_{}'.format(i)))
    if dropout>0: hidden_layers.add(Dropout(dropout))
    last_hidden = hidden_layers(inputs)
    outputs = Dense(n_class, activation='softmax', name='classifier')(last_hidden)
    model = Model(inputs=inputs, outputs=outputs)

    if False:
        hidden_layers.summary()
        model.summary()

    # optimizer
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['categorical_accuracy'])
    if X_val is not None:
        #callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)  # in fact not used
        hist = model.fit(X_train,
                         y_train,
                         validation_data=(X_val, y_val),
                         batch_size=batch_size, epochs=n_epoch,
                         verbose = verbose)
                         #callbacks = [callback])
    else:
        #callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        hist = model.fit(X_train,
                         y_train,
                         batch_size=batch_size, epochs=n_epoch,
                         verbose = verbose)
                         #callbacks = [callback])
    if save_model:
        save_model_hist(model, hist, f_save_main, cvi)
    else:
        save_model_hist(None, hist, f_save_main, cvi)
    #return model, hist


def cross_validation(X, y, f_save_main, cv = True, save_model = True,
                     test_features = False, use_features = None,
                     hypar = None, hypar_val = None):
    # cross validation for individual decoder
    # cv - set False for the final model
    # hypar, hypar_val - name and value of the hyperparameters; None for the default
    X = scaling_X(X)

    if use_features is None:
        feats = range(X.shape[1])
    else:
        feats = use_features
    X = X[:,feats,:]

    if cv:
        kf = KFold(n_splits=n_split, shuffle=True, random_state=randseed)

    if False:
        for i, (train_idx, val_idx) in enumerate(kf.split(X[:,0,0])):
            print('Fold {}'.format(i))
            print('Train index={}'.format(train_idx))
            print('Val index={}'.format(val_idx))

    if X.ndim == 3:
        X = np.expand_dims(X, axis=-1)

    if cv:
        iters = kf.split(X[:,0,0,0])
        n_iter = n_split
    else:
        iters = [[range(X.shape[0]),range(X.shape[0])]]
        n_iter = 1

    if test_features:
        acc_ratio = []

    for i, (train_idx, val_idx) in enumerate(iters):
        X_train = X[train_idx,:,:,:]
        y_train = y[train_idx]
        if cv:
            X_val = X[val_idx,:,:,:]
            y_val = y[val_idx]
        else:
            X_val = None
            y_val = None

        if cv:
            cvi = i
        else:
            cvi = 99

        print('Training model...')
        #reset_default_graph()
        if hypar is None:
            p1 = Process(target=cnn_decoder, args=(X_train, y_train, X_val, y_val,
                                                   f_save_main, cvi, save_model, None, None))
        else:
            p1 = Process(target=cnn_decoder, args=(X_train, y_train, X_val, y_val,
                                                   f_save_main, cvi, save_model, hypar, hypar_val))
        p1.start()
        p1.join()
        #model, hist = modelling(X_train, X_val, n_feature)
        #save_model_hist(model, hist, f_save_main, cvi)
        print('Cross-validation ({}/{}) done.'.format(i+1, n_iter))

        if test_features:
            model = load_model(os.path.join(f_save_main, 'model_lnpo{}.h5'.format(cvi)))
            if False:
                y_train_hat = np.argmax(model.predict(X_train), axis=1)
                np.mean(y_train_hat == y_train)
            y_val_hat = np.argmax(model.predict(X_val), axis=1)
            acc_val = np.mean(y_val_hat == y_val)
            tp_X_val = X_val.copy()
            acc_ratio.append([])
            print('Testing features...')
            for fi in range(X.shape[1]):  # mute each s/f
                tp_X_val[:, fi, :, :] = 0.5
                y_val_hat = np.argmax(model.predict(tp_X_val), axis=1)
                tp_acc = np.mean(y_val_hat == y_val)
                acc_ratio[-1].append(tp_acc/acc_val)
                print('{:3.2f}%'.format(fi/X.shape[1]*100), end = '\r')
            print('Done.')
        del X_train, y_train, X_val, y_val

    if test_features:
        acc_ratio = pd.DataFrame(np.array(acc_ratio), columns = feats)
        acc_ratio.to_csv(os.path.join(f_save_main, 'feature_testing.csv'), index=False)
    del X, y
    if cv: export_cv_perf(f_save_main)
    return None


def feature_testing_run(run, use_features=None):
    subjects = get_subjects()
    for si, s in enumerate(subjects):
        print('Cross-validation for subject ({}/{})...'.format(si+1, len(subjects)))
        t0 = time.time()
        f_save_main = os.path.join('results_idv_decoder', 'batch_1', 'sub-{}'.format(s), 'ses-1',
                                   'run-{}'.format(run))
        check_folder(f_save_main)
        X, y = load_dataset(s)

        cross_validation(X, y, f_save_main, cv=True, save_model=True, test_features=True, use_features=use_features)
        del X, y
        print('Done. Time cost {:3.2} hrs'.format((time.time() - t0) / 3600))
        print('')
    return None


def eval_features(run):
    alpha = 0.01
    mu = 1
    acc_ratio = []
    for si, s in enumerate(get_subjects()):
        f_save_main = os.path.join('results_idv_decoder', 'batch_1', 'sub-{}'.format(s), 'ses-1',
                                   'run-{}'.format(run))
        tp = pd.read_csv(os.path.join(f_save_main, 'feature_testing.csv'))
        if si == 0:
            colnames = tp.columns.tolist()
        acc_ratio.append(tp.mean().tolist())
    df = pd.DataFrame(np.array(acc_ratio), columns = colnames)
    fidx = np.where(df.to_numpy().mean(axis=0) < 1)[0]  # performance dropping without the feature
    pvals = []
    for fi in fidx:
        pvals.append(sp.stats.ttest_1samp(df.iloc[:,fi].tolist(), mu)[-1])
    fidx = fidx[np.array(pvals) < alpha]
    feats = df.columns.to_numpy().astype(int)[fidx].tolist()
    return feats


def cv_session():
    run = 0
    n_feat = [114]
    feature_testing_run(run, use_features = None)
    feats = eval_features(run)
    n_feat.append(len(feats))

    while n_feat[-1] < n_feat[-2] and n_feat[-1] > 0:
        run += 1
        feature_testing_run(run, use_features=feats)
        feats = eval_features(run)
        n_feat.append(len(feats))
    return n_feat


if False: # train final models
    use_features = eval_features(run=3)
    subjects = get_subjects()
    for si, s in enumerate(subjects):
        print('Final modeling for subject ({}/{})...'.format(si + 1, len(subjects)))
        t0 = time.time()
        f_save_main = os.path.join('results_idv_decoder', 'batch_1', 'sub-{}'.format(s), 'ses-1')
        check_folder(f_save_main)
        X, y = load_dataset(s)

        cross_validation(X, y, f_save_main, cv=False, save_model=True, test_features=False, use_features=use_features)
        del X, y
        print('Done. Time cost {:3.2} hrs'.format((time.time() - t0) / 3600))
        print('')


def predict_idv_decoder(subject, cvi):
    use_features = eval_features(run=3)
    if not cvi == 99:
        model = load_model(os.path.join('results_idv_decoder', 'batch_1', 'sub-{}'.format(subject),
                                        'ses-1', 'run-4', 'model_lnpo{}.h5'.format(cvi)))
    else:
        model = load_model(os.path.join('results_idv_decoder', 'batch_1', 'sub-{}'.format(subject),
                                        'ses-1', 'model_lnpo{}.h5'.format(cvi)))
    X, y = load_dataset(subject)
    X = scaling_X(X)
    X = X[:, use_features,:]
    X = np.expand_dims(X, axis=-1)
    y_hat = np.argmax(model.predict(X), axis=1)
    del X
    return y, y_hat


def get_perf_metrics(y, y_hat):
    acc = np.mean(y==y_hat)
    acc_target = []
    for target in range(3):
        acc_target.append(np.mean(y_hat[y==target]==target))
    return acc, acc_target


def summarize_performance_idv_decoder():
    subjects = get_subjects()
    acc = []
    trp = {0:[], 1:[], 2:[]}
    print('Getting performance...')
    for si, s in enumerate(subjects):
        y, y_hat = predict_idv_decoder(s, 99)
        metrics  = get_perf_metrics(y, y_hat)
        acc.append(metrics[0])
        for target in range(3):
            trp[target].append(metrics[1][target])
        print('{:3.2f}%'.format( (si+1)/len(subjects)*100), end = '\r')
    print('Done.')
    df = pd.DataFrame({'subject': subjects,
                       'accuracy': acc,
                       'acc_0': trp[0],
                       'acc_1': trp[1],
                       'acc_2': trp[2]})
    df['bacc'] = df[['acc_0', 'acc_1', 'acc_2']].mean(axis=1)
    df.to_csv(os.path.join('results_idv_decoder', 'batch_1', 'summary_performance_final.csv'), index=False)
    print(df.sort_values(by='bacc', ascending=False).round(3))
    return None


def export_feat_idx():
    feats = eval_features(run=3)
    feats = pd.DataFrame({'fid': feats})
    feats.to_csv(os.path.join('results_idv_decoder', 'batch_1', 'feature_idx.csv'), index=False)
    return None

