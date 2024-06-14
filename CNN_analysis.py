import numpy as np
import pandas as pd
import scipy as sp
import os
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


def get_subjects():
    bids_root = os.path.join('raw', 'batch_1', 'meg')
    files = os.listdir(bids_root)
    subjects_ex = ['CA123', 'CB999']
    sub_files = [f for f in files if f.startswith('sub-')]
    subjects = [s.split('-')[-1] for s in sub_files]
    for s in subjects_ex:
        subjects.remove(s)
    return subjects


def eval_performance(run, return_df=False):
    for si, s in enumerate(get_subjects()):
        f_save_main = os.path.join('results_cnn', 'batch_1', 'sub-{}'.format(s), 'ses-1',
                                   'run-{}'.format(run))
        tp = pd.read_csv(os.path.join(f_save_main, 'perf_cv.csv'))
        tp['participant'] = s
        if si == 0:
            df = tp.copy()
        else:
            df = pd.concat([df,tp])
    f_save_main = os.path.join('results_cnn', 'batch_1', 'sub-{}'.format(s), 'ses-1',
                               'run-{}'.format(run))
    tp = pd.read_csv(os.path.join(f_save_main, 'feature_testing.csv'))

    if return_df:
        return df, tp.shape[1]
    else:
        return df.mean(), tp.shape[1]


def plot_perf_over_runs(max_run):
    nfeats = []
    for run in range(max_run+1):
        perf, n = eval_performance(run, return_df=True)
        perf['nfeat'] = n
        perf['run'] = run
        if run == 0:
            df = perf.copy()
        else:
            df = pd.concat([df, perf.copy()], ignore_index=False)
        nfeats.append(n)
    df = pd.melt(df, id_vars=['participant', 'run', 'nfeat'], value_vars=['acc_train', 'acc_val'],
                 var_name = 'type', value_name = 'acc')
    sns.set_theme(style="white", font_scale=2)
    f = plt.figure()
    sns.pointplot(data = df, x = 'run', y ='acc', hue='type', palette='BuPu_r')
    plt.legend().remove()
    plt.xlabel('#feature')
    plt.xticks(ticks = range(max_run+1), labels = nfeats)
    # pairwise test
    fdr = pg.pairwise_tests(df[df.type=='acc_val'], dv = 'acc', between='nfeat', padjust='fdr_bh')
    fdr.round(3).to_csv(os.path.join('results_cnn', 'batch_1', 'acc_val_pairwise.csv'), index=False)
    print(fdr[fdr['p-corr']<0.05].round(3))
    # chose n_feat = 57, run = 4 ( eval_features(run=3) )


def plot_perf_idv_decoder_final():
    f_load = os.path.join('results_idv_decoder', 'batch_1', 'summary_performance_final.csv')
    df = pd.read_csv(f_load)
    df = df.rename({'acc_0': 'acc_N',
                    'acc_1': 'acc_R',
                    'acc_2': 'acc_T'}, axis=1)
    df.sort_values(by='bacc', ascending=False, inplace=True)
    df = pd.melt(df, id_vars=['subject'], value_vars=['accuracy', 'acc_T', 'acc_R', 'acc_N', 'bacc'],
                 var_name = 'metric', value_name = 'val')
    sns.set_theme(style="white", font_scale=1.5)
    f = plt.figure()
    ax = plt.gca()
    sns.pointplot(data=df, x='subject', y='val', hue='metric', palette='gnuplot2')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.hlines(1/3, xmin=0, xmax=45)
    plt.legend(ncol=5, loc='lower center', frameon=False)





