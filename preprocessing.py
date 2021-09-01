# -*- coding: utf-8 -*-
import os
from ctypes import *
from math import pow, floor, isnan, isinf

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lqx_SleepEEGNet.utils import dataset2name

WIN = 1   # unit on basis of second
NON_OVERLAP = 0.5   # percentage
NEIGHBORHOOD = 2


PREPRO_DIR = os.sep.join(['E:', 'EEG_dataset', 'prepro_data'])


def heatmap(data, row_labels=None, col_labels=None, ax=None, top=False, right=False, minor=True, grid=True,
            cbar_kw={}, cbarlabel=None, **kw):
    if not ax:
        ax = plt.gca()
    im = ax.imshow(data, **kw)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    if cbarlabel:
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va='bottom')

    if col_labels is not None:
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.tick_params(top=top, bottom=(not top), labeltop=top, labelbottom=(not top))
        plt.setp(ax.get_xticklabels(), rotation=(45 if top else -45), ha='left', rotation_mode='anchor')
        if minor:
            ax.set_xticks(np.arange(len(col_labels) + 1) - .5, minor=True)
            ax.tick_params(which='minor', bottom=(not top), top=top)
    if row_labels is not None:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.tick_params(right=right, left=(not right), labelright=right, labelleft=(not right))
        if minor:
            ax.set_yticks(np.arange(len(row_labels) + 1) - .5, minor=True)
            ax.tick_params(which='minor', left=(not right), right=right)
    if minor and grid:
        ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    return im, cbar


def annotate_heatmap(im, data=None, fdata=None, fmt='{: .2f}', textcolors=('k', 'w'), threshold=None,
                   fontweight=('normal', 'bold'), **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    # 将给定阈值与图像像素匹配
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2
    print(threshold)
    kw = dict(ha='center', va='center')
    kw.update(textkw)
    texts = []
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            choice = int(im.norm(data[j, i]) > threshold)
            kw.update(color=textcolors[choice], fontweight=fontweight[choice])
            #'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold',
            #  'demi', 'bold', 'heavy', 'extra bold', 'black'
            if fdata is not None:
                text = im.axes.text(i, j, str(data[j, i]) + '\n' + fmt.format(fdata[j, i]), **kw)
            else:
                text = im.axes.text(i, j, str(data[j, i]), **kw)
            texts.append(text)
    return texts


# confusion matrix records the numbers of samples, x ---- predict output, y ----- ground truth
def plot_confusion(matrix):
    percentage = matrix / np.sum(matrix, axis=-1, keepdims=True)
    labels = ['W', 'N1', 'N2', 'N3', 'REM']
    fig, ax = plt.subplots()
    im, cbar = heatmap(matrix, labels, labels, ax=ax, cmap='Purples', cbarlabel=None, top=True)
    texts = annotate_heatmap(im, fdata=percentage)
    print(texts)
    ax.set_xlabel('predict output')
    ax.set_ylabel('ground truth')
    fig.tight_layout()
    plt.show()


# data shape is possibly (epochs, chs, wins, nttf/2+1) or (chs, wins, nfft/2+1)
def plot_pectrogram(data, fs):
    if data.shape[-3] == 3:
        fig, axes = plt.subplots(3, 1, sharex='col', sharey='row', squeeze=False)
        labels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']
    elif data.shape[-3] == 4:
        fig, axes = plt.subplots(4, 1, sharex='col', sharey='row', squeeze=False)
        labels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']
    else:
        fig, axes = plt.subplots(4, 3, sharex='col', sharey='row')
        labels = ['EEG F3-A2', 'EEG F4-A1', 'EEG C3-A2', 'EEG C4-A1', 'EEG O1-A2', 'EEG O2-A1', 'EOG LOC-A2',
                  'EOG ROC-A2', 'EMG Chin', 'Leg 1', 'Leg 2', 'ECG II']
    colors = ['tomato', 'gold', 'tan', 'mediumpurple', 'deepskyblue', 'forestgreen', 'rosybrown', 'turquoise',
              'darkorange', 'slategrey', 'royalblue', 'olive']
    if len(data.shape) == 4:
        data = np.hstack(data)
    X, Y = np.mgrid[0:(data.shape[-2]-1):complex(0, data.shape[-2]), 0:fs/2:complex(0, data.shape[-1])]
    for i in range(len(labels)):
        colid = (i + axes.shape[-1]) % axes.shape[-1]
        rowid = (i + axes.shape[-1]) // axes.shape[-1] - 1
        axes[rowid, colid].set_title(labels[i], fontsize=8)
        #row_labels = np.linspace(0, fs/2, 10).astype('<U3')
        #col_label = np.arange(0, epoch_data.shape[1], 5).astype('<U2')
        #im, cbar = heatmap(epoch_data[i].transpose(), row_labels=row_labels, col_labels=col_label,
        #                  ax=axes[rowid, colid], cmap='ocean', minor=False, grid=False)
        heatmap = axes[rowid, colid].pcolor(X, Y, data[i], cmap='Spectral',
                                            norm=matplotlib.colors.Normalize(vmin=data[i].min(),
                                                                             vmax=data[i].max()))
        fig.colorbar(heatmap, ax=axes[rowid, colid])
        if rowid == axes.shape[0] - 1:
            axes[rowid, colid].set_xlabel('window', fontsize=6)
        if colid == 0:
            axes[rowid, colid].set_ylabel('Amplitude frequency', fontsize=6)
    fig.subplots_adjust(hspace=0.5)
    #plt.tight_layout()
    plt.show()


# regardless of sign, get nearest exponent in term of 2 base
def next_pow2(N):
    class F32Bits(Structure):
        _fields_ = [('M', c_uint, 23), ('E', c_uint, 8), ('F', c_uint, 1)]

    class F32(Union):
        _fields_ = [('value', c_float), ('bits', F32Bits)]

    if isnan(N):
        return 'NaN'
    if isinf(N):
        return 'inf'
    f = F32()
    f.value = N
    #print(f.bits.F, f.bits.E, f.bits.M)
    if f.value == 0.:
        return 0
    elif abs(f.value) == 1.:
        return 1
    else:
        if f.bits.M == 0:
            return f.bits.E - 127
        else:
            return f.bits.E - 127 + 1


def compute_stft_para(fs, width, non_overlap):
    # width & overlap should be well considered
    window = int(width * fs)
    stride = int(window * non_overlap)
    nfft = int(pow(2, next_pow2(window)))     # use window or fs?
    return window, stride, nfft


# t_data --- (epochs, chs, wins, window)
def win_ssa(t_data, threshold=0.):
    ori_shape = t_data.shape
    t_data = np.reshape(t_data, (-1, t_data.shape[-1]))
    row = int(t_data.shape[-1] // 4)
    col = t_data.shape[-1] - row + 1
    track = np.zeros((t_data.shape[0], row, col), dtype=t_data.dtype)
    for i in range(col):
        track[:, :, i] = t_data[:, i: i + row]
    # svd
    u, s, v = np.linalg.svd(track, full_matrices=False)
    # select singular values
    if threshold:
        flag = np.sum(s, axis=-1, keepdims=True) * threshold - s[:, 0]
        idx = 0
        while any(flag > 0):
            flag = flag - s[: idx + 1]
            idx += 1
    else:
        idx = 1
    selected = s[:, 0: idx + 1]
    u = u[:, :, 0: idx + 1]
    v = v[:, 0: idx + 1, :]
    new = np.matmul(selected[:, np.newaxis, :] * u, v)
    # reconstruction
    recon = np.zeros_like(t_data)
    for i in range(row - 1):
        for j in range(i + 1):
            recon[:, i] += new[:, i - j, j]
        recon[:, i] = recon[:, i] / (i + 1)
    for i in range(row - 1, col):
        for j in range(row):
            recon[:, i] += new[:, j, i - j]
        recon[:, i] /= row
    for i in range(col, t_data.shape[-1]):
        for j in range(t_data.shape[-1] - i):
            recon[:, i] += new[:, i - col + j + 1, col - j - 1]
        recon[:, i] = recon[:, i] / (t_data.shape[-1] - i)
    return recon.reshape(ori_shape)


# data's shape is (epochs, chs, samples_per_epoch), if data is about one epoch, reshaping first
# data should be from one sleep epoch in sequence method
# used after standardizing data needed
def STFT(std_data, window, stride, nfft):
    is_newaxis = False
    if len(std_data.shape) == 2:
        std_data = std_data[np.newaxis, :]
        is_newaxis = True
    # divide one epoch into small frames
    # stride must not be 0
    num_wins = floor((std_data.shape[-1] - window) / stride) + 1
    df_data = np.zeros(list(std_data.shape[:-1]) + [num_wins, window])
    for i in range(num_wins):
        df_data[:, :, i, :] = std_data[:, :, i * stride:i * stride + window]
    #print('There are [{}] windows in one sleep epoch\noriginal shape is {}\nafter divided into frames,the shape is {}'.
    #      format(num_wins, std_data.shape, df_data.shape))
    # window
    w_data = df_data * np.hamming(window)
    # FFT
    fft_data = np.fft.rfft(w_data, nfft, axis=-1)   # shape --> (chs, wins, nfft /2 + 1)
    #print('after FFT, the shape is {}'.format(fft_data.shape))
    # dB
    db_data = 20 * np.log10(np.clip(np.abs(fft_data), 1e-20, 1e100))
    #print('after preprocessing\nmax\n{}\nmin\n{}'.format(np.max(db_data, axis=(1, 2)), np.min(db_data, axis=(1, 2))))
    if is_newaxis:
        db_data = np.squeeze(db_data, axis=0)
        df_data = np.squeeze(df_data, axis=0)
    return db_data, df_data


# to get several epochs' data as a example, its shape is (epochs, chs, wins, nfft/2+1) after STFT
def compute_dynamics(data, theta):
    is_newaxis = False
    if len(data.shape) == 3:
        data = data[np.newaxis, :]
        is_newaxis = True
    delta = np.zeros(data.shape)
    denominator = np.sum(np.arange(1, theta+1)**2)
    for t in range(data.shape[-2]):
        if t < theta:
            delta[:, :, t, :] = data[:, :, t+1, :] - data[:, :, t, :]
        elif t >= data.shape[-2] - theta:
            delta[:, :, t, :] = data[:, :, t, :] - data[:, :, t-1, :]
        else:
            sum = 0
            for n in range(1, theta+1):
                sum += n * (data[:, :, t+n, :] - data[:, :, t-n, :])
            delta[:, :, t, :] = sum / np.clip(denominator, np.finfo(float).eps, None)
    if is_newaxis:
        delta = np.squeeze(delta, axis=0)
    return delta


# normalized by maximum in term of the whole file data
def separate_sign(delta, delta_delta):
    bs = np.sign(delta)
    pm = np.sign(delta_delta)
    sign = np.sign(delta * delta_delta)
    des_1 = 1 - np.asarray(np.logical_not((sign == 1) & (bs == -1)), dtype=np.int)  # v<0, a<0, 拉深
    inc_1 = 1 - np.asarray(np.logical_not((sign == 1) & (bs == 1)), dtype=np.int)  # v>0, a>0, 拔高
    inc_2 = 1 - np.asarray(np.logical_not((sign == -1) & (bs == -1)), dtype=np.int)    # v<0, a>0, 拔高
    des_2 = 1 - np.asarray(np.logical_not((sign == -1) & (bs == 1)), dtype=np.int)  # v>0, a<0, 拉深
    des_3 = 1 - np.asarray(np.logical_not((bs == -1) & (pm == 0)), dtype=np.int)   # v<0, a=0, 拉深
    inc_3 = 1 - np.asarray(np.logical_not((bs == 1) & (pm == 0)), dtype=np.int)   # v>0, a=0, 拔高
    s = 1 - np.asarray(np.logical_not((bs == 0) & (pm == -1)), dtype=np.int)   # v=0, a<0, 极大， 拔高
    b = 1 - np.asarray(np.logical_not((bs == 0) & (pm == 1)), dtype=np.int)   # v=0, a>0, 极小， 拉深
    aux = np.abs(delta * delta_delta)
    increase = aux * inc_1 + delta * delta * inc_3 + aux * inc_2 + delta_delta * delta_delta * s
    decline = aux * des_1 + delta * delta * des_3 + aux * des_2 + delta_delta * delta_delta * b
    maximum = max(np.max(decline), np.max(increase), np.finfo(float).eps)
    decline = decline / maximum
    increase = increase / maximum
    return decline, increase


def augment(f_data, decline, increase):
    dec = np.abs(f_data * (0.5 - decline))
    inc = np.abs(f_data * (0.5 - increase))
    return f_data - dec + inc


# file_data should be ndarray-type, shape is (epochs, chs, samples_per_epoch)
# return data shape-----fre:(epochs, chs, wins, nfft//2+1)   time:(epochs, chs, wins, per_win)
def preprocessing(fs, data, outputs, width=WIN, non_overlap=NON_OVERLAP, dynamics_theta=NEIGHBORHOOD, stat=None,
                  trunc_reserved=0, trunc_start=0):
    '''
    return corresponding preprocessed data and some information
    :param fs: sampling rate, float
    :param data:
    :param outputs: iterative object, indicates this function should return what
    :param width: seconds, int
    :param non_overlap: non-overlapping percentage between two adjacent windows
    :param dynamics_theta: used to compute dynamic features, on behalf of neighborhood size
    :param stat: some statistics need using, if not None, is 2-nested-element and each element given information
                 in order of 't', 'f', 'df'
    :param trunc_reserved: the number of data returned is truncated from original 'data'
    :param trunc_start: valid only when 'trunc_reserved' is positive integer
    :return: a 3-element iterative object in order of 't', 'f', 'df' and a nested tuple object with
              mean and variance information in corresponding of returned data
    '''
    returned = list(None for _ in range(3))
    window, stride, nfft = compute_stft_para(fs, width, non_overlap)

    def _frequence_process(_std_data, need_d=False):
        f_data, _ = STFT(_std_data, window, stride, nfft)
        if need_d:
            delta = compute_dynamics(f_data, dynamics_theta)
            delta_delta = compute_dynamics(delta, dynamics_theta)
            return f_data, delta, delta_delta
        return f_data

    start = trunc_start if trunc_reserved else 0
    end = start + trunc_reserved if trunc_reserved else data.shape[0]

    if stat is not None:
        data = data[start: end]
        std_data = (data - stat[0][0]) / np.sqrt(np.clip(stat[1][0], np.finfo(float).eps, None))
        if 'f' in outputs:
            if 'df' in outputs:
                f, d, dd = _frequence_process(std_data, True)
                d = (d - stat[0][2][0]) / np.sqrt(np.clip(stat[1][2][0], np.finfo(float).eps, None))
                dd = (dd - stat[0][2][1]) / np.sqrt(np.clip(stat[1][2][1], np.finfo(float).eps, None))
                returned[-1] = np.stack((d, dd), axis=-1)
            else:
                f = _frequence_process(std_data)
            f = (f - stat[0][1]) / np.sqrt(np.clip(stat[1][1], np.finfo(float).eps, None))
            returned[1] = f
    else:
        stat = ([], [])
        stat[0].append(np.mean(data, axis=(0, 2), keepdims=True))
        stat[1].append(np.var(data, axis=(0, 2), keepdims=True))
        std_data = (data - stat[0][0]) / np.sqrt(np.clip(stat[1][0], np.finfo(float).eps, None))
        if 'f' in outputs:
            if 'df' in outputs:
                f, d, dd = _frequence_process(std_data, True)
                stat[0].append(np.mean(f, axis=(0, 2, 3), keepdims=True))
                stat[1].append(np.var(f, axis=(0, 2, 3), keepdims=True))
                stat[0].append((np.mean(d, axis=(0, 2, 3), keepdims=True), np.mean(dd, axis=(0, 2, 3), keepdims=True)))
                stat[1].append((np.var(d, axis=(0, 2, 3), keepdims=True), np.var(dd, axis=(0, 2, 3), keepdims=True)))
                d = (d - stat[0][2][0]) / np.sqrt(np.clip(stat[1][2][0], np.finfo(float).eps, None))
                dd = (dd - stat[0][2][1]) / np.sqrt(np.clip(stat[1][2][1], np.finfo(float).eps, None))
                d = d[start: end]
                dd = dd[start: end]
                returned[-1] = np.stack((d, dd), axis=-1)
            else:
                f = _frequence_process(std_data)
                stat[0].append(np.mean(f, axis=(0, 2, 3), keepdims=True))
                stat[1].append(np.var(f, axis=(0, 2, 3), keepdims=True))
            f = (f - stat[0][1]) / np.sqrt(np.clip(stat[1][1], np.finfo(float).eps, None))
            f = f[start: end]
            returned[1] = f
        std_data = std_data[start: end]
    if 't' in outputs:
        returned[0] = std_data

    return returned, stat


DATA_INFO = 'dataset_info'


def prepro_data_store(data_dir, dataset_id, prepro_dir=PREPRO_DIR, data_info_dir=DATA_INFO):
    import ntpath
    import glob
    from lqx_SleepEEGNet.utils import get_info, get_subject_data
    from math import ceil
    import sys
    import shutil
    dataset_name = dataset2name[dataset_id]
    fs = get_info(dataset_name, data_info_dir)['fs']
    dataset_dir = os.sep.join([data_dir, dataset_name, ''])
    print(dataset_dir)
    files = glob.glob(dataset_dir + '*.npz')
    files.sort()
    print(len(files))
    dataset_dir = os.sep.join([prepro_dir, dataset_name])
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir, mode=0o777)
    window, stride, nfft = compute_stft_para(fs, WIN, NON_OVERLAP)
    for file in files:
        name = ntpath.basename(file)
        data = get_subject_data(file, 'x', 'y', 'trans_y')      
        x_mean = np.mean(data[0], axis=(0, 2), keepdims=True)
        x_var = np.var(data[0], axis=(0, 2), keepdims=True)
        std_x = (data[0] - x_mean) / np.sqrt(np.clip(x_var, np.finfo(float).eps, None))
        f_data, _ = STFT(std_x, window, stride, nfft)
        #ssa_t = []
        #for ch in range(t_data.shape[1]):
        #    ssa_t.append(win_ssa(t_data[:, ch: ch + 1]))
        #    sys.stdout.write(('channel: %d' % ch) + '  ')
        #    sys.stdout.flush()
        #print('\n')
        #ssa_t = np.concatenate(ssa_t, axis=1)
        delta = compute_dynamics(f_data, NEIGHBORHOOD)
        delta_delta = compute_dynamics(delta, NEIGHBORHOOD)
        f_mean = np.mean(f_data, axis=(0, 2, 3), keepdims=True)  # (1, chs, 1, 1)
        f_var = np.var(f_data, axis=(0, 2, 3), keepdims=True)
        delta_mean = np.mean(delta, axis=(0, 2, 3), keepdims=True)
        delta_var = np.var(delta, axis=(0, 2, 3), keepdims=True)
        delta_d_mean = np.mean(delta_delta, axis=(0, 2, 3), keepdims=True)
        delta_d_var = np.var(delta_delta, axis=(0, 2, 3), keepdims=True)
        f_data = (f_data - f_mean) / np.sqrt(np.clip(f_var, np.finfo(float).eps, None))
        delta = (delta - delta_mean) / np.sqrt(np.clip(delta_var, np.finfo(float).eps, None))
        delta_delta = (delta_delta - delta_d_mean) / np.sqrt(np.clip(delta_d_var, np.finfo(float).eps, None))
        path = os.sep.join([dataset_dir, name])
        d = {
            't': std_x,      # (epochs, chs, fs*30)
            'f': f_data,     # (epochs, chs, wins, nfft//2 + 1)
            'df': np.stack((delta, delta_delta), axis=-1),  # (epochs, chs, wins, nfft//2 + 1, 2)
            'y': data[1],
            'trans_y': data[2]
        }
        with open(path, 'wb') as f:
            np.savez(f, **d)
        print('has finished %s' % name)
        

prepro_data_store('data', 1)


# =================================================================================================================
def fetch_data(file='data\\personal\\01.npz'):
    with np.load(file) as f:
        d = {
            'x': f['x'],
            'y': f['y'],
            'fs': float(f['fs']),
            'sc': f['selected_channel']
        }
    return d


def demo():
    package = fetch_data()
    data = package['x']
    #plot_pectrogram(preprocessing_d(package['fs'], data)[0][:2], package['fs'])
    data, delta, delta_delta = preprocessing_d(package['fs'], data)
    decline, increase = separate_sign(delta, delta_delta)
    print(decline.shape, increase.shape)


#demo()
#confusion_matrix = np.random.randint(0, [4900, 1000, 4200, 2500, 1700], size=(5, 5))
#plot_confusion(confusion_matrix)


