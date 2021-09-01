# -*- coding: utf-8 -*-
import os
import numpy as np
import glob
from math import pow
import pandas as pd
from ctypes import *
import re


W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5


stage_dict = {
    'W': W,
    'N1': N1,
    'N2': N2,
    'N3': N3,
    'REM': REM,
    'UNKNOWN': UNKNOWN
}


class_dict = {
    0: 'W',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM',
    5: 'UNKNOWN'
}


sleep_EDF_label = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4,
    'Sleep stage ?': 5,
    'Movement time': 0
}


personal_label = {
    'WK': 0,
    'N1': 1,
    'N2': 2,
    'N3': 3,
    'REM': 4,
    'NS': 5
}


EPOCH_SEC_SIZE = 30


dataset2name = {
    0: 'personal',
    1: 'sleep-EDF-data_2018',
    2: 'sleep-EDF-sleep-telemetry'
}


name2dataset = {
    'personal': 0,
    'sleep-EDF-data_2018': 1,
    'sleep-EDF-sleep-telemetry': 2
}


origin_dir = os.sep.join(['E:', 'EEG_dataset', 'original'])


SC_subjects_path = os.sep.join([origin_dir, 'sleep-expanded', 'SC-subjects.xls'])
ST_subjects_path = os.sep.join([origin_dir, 'sleep-expanded', 'ST-subjects.xls'])


LABELS = list(class_dict.values())[:-1]


class ConfusionMatrix(object):
    def __init__(self, class_label=LABELS):
        super(ConfusionMatrix, self).__init__()
        if len(set(class_label)) != len(class_label):
            raise ValueError('passed parameter has some elements are not unique')
        self.class_label = class_label
        self.__n_matrix = np.zeros((len(class_label),) * 2, dtype='int32')
        self.__2_matrix = np.zeros((2, 2), dtype='int32')
        # is not confusion matrix, 0 col indicates whether right or not on non-transition set
        self.__n_for_transit = np.zeros((2, 2), dtype='int32')
        self._n_dict = {'col': ['gt_' + l for l in self.class_label], 'row': ['pre_' + l for l in self.class_label]}
        self._2_dict = {'col': ['gt_non_trans', 'gt_trans'], 'row': ['pre_non_trans', 'pre_trans']}
        self.__recording = False
        self.__configured = False
        self.multitask = False

    def record_ini(self, multitask=False):
        self.__configured = True
        if multitask:
            self.__multitask_n_matrix = np.zeros(self.__n_matrix.shape, dtype='int32')
            self.__multitask_n_for_transit = np.zeros((2, 2), dtype='int32')
            self.multitask = True
        else:
            self.__multitask_n_matrix = None
            self.__multitask_n_for_transit = None

    # passed must be tuple/list
    def record(self, true, pre, trans_y):
        if self.__configured:
            self.__recording = True
            for t, p, _t in zip(true, pre, trans_y):
                if t.shape[0] != 0:  # not placehold
                    _p = np.equal(t, p).astype('int32')  # classifiation result right or not, 0 --- not
                    for e_t, e_p, e_tr_t, e_tr_p in zip(t, p, _t, _p):
                        self.__n_matrix[e_p, e_t] += 1
                        self.__n_for_transit[e_tr_p, e_tr_t] += 1
            self.__recording = False
        else:
            raise Exception('please configure first')

    # passed must be tuple/list
    # branch1 has values returned
    def record_plus_trans(self, true_n, pre_n, new_pre_n, true_2, pre_2):
        if self.__configured and self.multitask:
            self.__recording = True
            for i, (t, p, n_p, tr_t, tr_p) in enumerate(zip(true_n, pre_n, new_pre_n, true_2, pre_2)):
                if t.shape[0] != 0:
                    _p1 = np.equal(t, p).astype('int32')
                    _p2 = np.equal(t, n_p).astype('int32')
                    for e_t, e_p, e_n_p, e_tr_t, e_tr_p, e_p_, e_n_p_ in zip(t, p, n_p, tr_t, tr_p, _p1, _p2):
                        self.__n_matrix[e_p, e_t] += 1
                        self.__2_matrix[e_tr_p, e_tr_t] += 1
                        self.__n_for_transit[e_p_, e_tr_t] += 1
                        self.__multitask_n_matrix[e_n_p, e_t] += 1
                        self.__multitask_n_for_transit[e_n_p_, e_tr_t] += 1
            self.__recording = False
        else:
            raise Exception('not to configure ahead or not to support multitask')

    @property
    def recording(self):
        return self.__recording

    @property
    def configured(self):
        return self.__configured

    @property
    def matrix_n(self):
        return self.__n_matrix

    @property
    def matrix_2(self):
        return self.__2_matrix

    @property
    def matrix_multitask(self):
        return self.__multitask_n_matrix

    @property
    def transit(self):
        return self.__n_for_transit

    @property
    def transit_multitask(self):
        return self.__multitask_n_for_transit

    @property
    def data_frame_n(self):
        return pd.DataFrame(self.__n_matrix, index=self._n_dict['row'], columns=self._n_dict['col'])

    @property
    def data_frame_multitask_n(self):
        return pd.DataFrame(self.__multitask_n_matrix, index=self._n_dict['row'], columns=self._n_dict['col'])

    @property
    def data_frame_2(self):
        return pd.DataFrame(self.__2_matrix, index=self._2_dict['row'], columns=self._2_dict['col'])

    # matrix in ('n', '2', 'trans', 'multitask', 'multi_trans')
    def get_element(self, row, col, matrix='n'):
        if matrix == 'n':
            return self.__n_matrix[row, col]
        elif matrix == '2':
            return self.__2_matrix[row, col]
        elif matrix == 'trans':
            return self.__n_for_transit[row, col]
        elif matrix == 'multitask':
            return self.__multitask_n_matrix[row, col]
        elif matrix == 'multi_trans':
            return self.__multitask_n_for_transit[row, col]
        else:
            raise KeyError('the value of param \'matrix\' is wrong')

    # matrix in ('n', '2', 'trans', 'multitask', 'multi_trans')
    def compute_acc(self, matrix='n'):
        if matrix == 'n':
            return self.__n_matrix.trace() / np.sum(self.__n_matrix)
        elif matrix == '2':
            return self.__2_matrix.trace() / np.sum(self.__2_matrix)
        elif matrix == 'trans':
            return self.__n_for_transit[1] / np.sum(self.__n_for_transit, axis=0)
        elif matrix == 'multitask':
            return self.__multitask_n_matrix.trace() / np.sum(self.__multitask_n_matrix)
        elif matrix == 'multi_trans':
            return self.__multitask_n_for_transit[1] / np.sum(self.__multitask_n_for_transit, axis=0)
        else:
            raise KeyError('the value of param \'matrix\' is wrong')

    def reset(self):
        if self.__configured:
            self.__n_matrix = np.zeros(self.__n_matrix.shape, dtype='int32')
            self.__n_for_transit = np.zeros((2, 2), dtype='int32')
            if self.multitask:
                self.__2_matrix = np.zeros((2, 2), dtype='int32')
                self.__multitask_n_matrix = np.zeros(self.__n_matrix.shape, dtype='int32')
                self.__multitask_n_for_transit = np.zeros((2, 2), dtype='int32')
        else:
            raise Exception('please configure first')


# fetch the information of datasets which have been loaded
def get_info(dataset, path=None):
    if path:
        path = os.sep.join([path, ''])
    try:
        if type(dataset) == int and dataset2name.get(dataset, None) is not None:
            temp = dataset2name[dataset] + '_info.npz'
            filename = path + temp if path else temp
        elif type(dataset) == str and name2dataset.get(dataset, None) is not None:
            temp = dataset + '_info.npz'
            filename = path + temp if path else temp
        else:
            raise Exception('Dataset is nonexistent')
        if not os.path.exists(filename):
            raise Exception('Dataset has not been loaded, thus there is no information about it')
        with np.load(filename) as f:
            d = {
                'data_dir': str(f['data_dir']),
                'output_dir': str(f['output_dir']),
                'fs': float(f['sampling_rate']),
                'sc': list(f['selected_channels']),
                'data_number': int(f['data_number']),
                'dataset_name': str(f['dataset_name']),
                'inventory': str(f['inventory']),
                'before_stat': f['before_stat'],
                'after_stat': f['after_stat'],
                'trans_stat': f['trans_stat']
            }
        return d
    except Exception as e:
        raise e


#print(get_info(0))


def get_subject_data(file, *args, id=0, gain=0):
    '''
    get one subject information needed
    :param file:
    :param args:  'x', 'y', 'trans_y', 'fs', 'selected_channels' or list('W',...), 'before_stat', 'after_stat',
                  'trans_stat'
    :param id: file id
    :param gain:  is valid if 'id' needed, to indicate the file that the correspond epoch is from
    :return: list involves information
    '''
    try:
        if len(args) == 0:
            raise Exception('no parameters to read! Please input them')
        capture = []
        with np.load(file) as f:
            for k in args:
                if k == 'x':
                    capture.append(f['x'])
                elif k == 'y':
                    capture.append(f['y'])
                elif k == 'trans_y':
                    capture.append(f['trans_y'])
                elif k == 'fs':
                    capture.append(float(f['fs']))
                elif k == 'selected_channels':
                    capture.append(list(f['selected_channel']))
                elif k == 'before_stat':
                    capture.append(f['before_stat'])
                elif k == 'after_stat':
                    capture.append(f['after_stat'])
                elif k == 'trans_stat':
                    capture.append(f['trans_stat'])
                elif isinstance(k, (list, np.ndarray, tuple)):
                    stats = []
                    #print(k)
                    r = np.zeros((len(k),), dtype=np.int)
                    for i, label in enumerate(k):
                        item = np.where(f['y'] == stage_dict[label])[0] + int(id * pow(10, gain))
                        r[i] = len(item)
                        stats.append(item)
                    stats.append(r)
                    capture.append(stats)
        return capture
    except Exception as e:
        raise e


# 'types' must be iterative  --- used in config.SeqDataLoader
def get_part_experiment_data(file, start, stop, types, N=False):
    capture = []
    with np.load(file, allow_pickle=True) as f:
        for t in types:
            capture.append(f[t][start: stop])
        capture.append(f['y'][start: stop])
        n = f['y'].shape[0]
        capture.append(f['trans_y'][start: stop])
    if N:
        capture.append(n)
    return capture


# used in config.SeqDataLoader
def get_part_subject_data(path, start, stop, *args):
    with np.load(path) as f:
        x = f['x'][start:stop]
        if len(x) == 0:
            raise IndexError('index(\'start\' parameter) has exceed the boundary!')
        y = f['y'][start:stop]
        tr_y = f['trans_y'][start:stop]
        if len(args) != 0:
            other = []
            for k in args:
                if k == 'fs':
                    other.append(float(f['fs']))
                elif k == 'selected_channels':
                    other.append(list(f['selected_channel']))
                elif k == 'before_stat':
                    other.append(f['before_stat'])
                elif k == 'after_stat':
                    other.append(f['after_stat'])
                elif k == 'trans_stat':
                    other.append(f['trans_stat'])
                else:
                    raise KeyError('the intended information is not in!')
    return (x, y, tr_y) + tuple(other)


# groups --- ((0, 12), (), (5))  the second item is on behalf of var_list's remainder
# maybe used in train
def group_by_optimizers(var_list, indices):
    opt_num = len(indices)
    groups = []
    temp = np.arange(len(var_list))
    aux = []
    for i in range(opt_num):
        if len(indices[i]) == 0:
            flag = i
            continue
        else:
            groups.append([])
            aux += indices[i]
            for v in indices[i]:
                groups[-1].append(var_list[v])
    aux.sort()
    temp = np.setdiff1d(temp, np.asarray(aux).astype('int32'))
    aux = []
    for v in temp:
        aux.append(var_list[v])
    groups.insert(flag, aux)
    return groups


def get_epochs_data(path, *args, gain=5, mode=0, index=None):
    '''
    get several subjects' corresponding information, is a generator with one subject by one
    :param path:
    :param args: 'x', 'y', 'trans_y', 'fs', 'selected_channels' or list('W',...) 'before_stat', 'after_stat',
                 'trans_y'
    :param gain: used to form convenient file index if list('W',...) gives, the base of exponent is 10
    :param mode: mode represents path's type, 0-----list(str)/1D array(str), 1------npz_dir, 'path' is invalid
    :param index: when 'path' is valid, the 'index' is the corresponding indices
    :return: list
    '''
    try:
        if len(args) == 0:
            raise Exception('no parameters to read! Please input them')
        if mode == 1:
            if not os.path.exists(path):
                raise Exception('npz dir is nonexistent')
            else:
                files = glob.glob(os.sep.join([path, '*.npz']))
                files.sort()
                if len(files) == 0:
                    raise Exception('the dir has no \'.npz\' file')
        else:
            files = path
        if index is None:
            index = np.arange(len(files))
        for i, file in zip(index, files):
            capture = []
            with np.load(file) as f:
                for k in args:
                    if k == 'x':
                        capture.append(f['x'])
                    elif k == 'y':
                        capture.append(f['y'])
                    elif k == 'trans_y':
                        capture.append(f['trans_y'])
                    elif k == 'fs':
                        capture.append(float(f['fs']))
                    elif k == 'selected_channels':
                        capture.append(list(f['selected_channel']))
                    elif k == 'before_stat':
                        capture.append(f['before_stat'])
                    elif k == 'after_stat':
                        capture.append(f['after_stat'])
                    elif k == 'trans_stat':
                        capture.append(f['trans_stat'])
                    elif isinstance(k, (list, np.ndarray, tuple)):
                        stats = []
                        for label in k:
                            stats.append(np.where(f['y'] == stage_dict[label])[0] + int(i * pow(10, gain)))
                        capture.append(stats)
            yield capture
    except Exception as e:
        raise e


# 1) the whole dir; 2) save file union, index is in term of this union; 3) index is in term of dataset
def resolve_index_gain(indices, gain, labels=None):
    '''
    fetch epochs in term of subject from a epochs union which mixed subjects, note that returned subject index list
    matches with the file union which is used ever, so corresponding information should be known beforehand.
    :param indices: 1D ndarray-like, is the indices of sleep epochs from mixed subjects
    :param gain: this parameter used to form the file index
    :param labels: 1D ndarray-like, corresponding of indices
    :return: a list involves unique subject index , a corresponding list whose element is certain subject's epochs list
    '''
    indices = np.asarray(indices)
    epoch_idx = (indices % pow(10, gain)).astype(np.int)
    raw_subject_idx = (indices // pow(10, gain)).astype(np.int)
    subject_idx = set(raw_subject_idx)
    epochs = []
    if labels is not None:
        labels = np.asarray(labels)
        new_labels = []
        for s in subject_idx:
            idx = np.where(raw_subject_idx == s)[0]
            aux = np.vstack((epoch_idx[idx], idx)).transpose().tolist()
            aux = sorted(aux, key=lambda x: x[0])
            aux = np.asarray(aux)
            epochs.append(epoch_idx[aux[:, 0]].tolist())
            new_labels.append(labels[aux[:, 1]].tolist())
        return list(subject_idx), epochs, new_labels
    else:
        for s in subject_idx:
            epochs.append(np.sort(epoch_idx[np.where(raw_subject_idx == s)[0]]).tolist())
        return list(subject_idx), epochs


def from_subjects_fetch_epochs(subject_idx, epochs, data_dir=None, dataset_id=None):
    if data_dir:
        inv = pd.read_csv(os.sep.join([data_dir, 'inventory.csv']))['file_name']
    elif dataset_id:
        d = get_info(dataset_id)
        inv = pd.read_csv(d['inventory'])['file_name']
        data_dir = d['data_dir']
    else:
        raise ValueError('specify one of file dir and dataset id at least')
    x = []
    y = []
    tr_y = []
    for idx, indices in zip(subject_idx, epochs):
        print(idx, indices)
        file = os.sep.join([data_dir, inv.iloc[idx]])
        with np.load(file) as f:
            x.append(f['x'][indices])
            y.append(f['y'][indices])
            tr_y.append(f['trans_y'][indices])
    return np.concatenate(x), np.concatenate(y), np.concatenate(tr_y)


# SC dataset's numbers of people of all ages
class AgeStructure(Structure):
    _fields_ = [
       ('young', c_int),
       ('mid_aged', c_int),
       ('old', c_int),
       ('super_old', c_int)
    ]


# choice is integer, which is needed to train in balance
# not to use for sequence training
# return form: [np.ndarray,...] number is equivalent to labels'
def choose_epochs(dataset_id, choice, gain=5, labels=['W', 'N1'], indices=None):
    d = get_info(dataset_id)
    inventory = pd.read_csv(d['inventory'])
    #stat = d['after_stat']
    dir = d['data_dir']
    if indices is None:
        file_id = inventory.index.values
    else:
        file_id = indices
    details = np.zeros((len(labels),), dtype=np.int)
    epochs = []
    for i in file_id:
        path = os.sep.join([dir, inventory.iloc[i, 0]])
        x, y, epoch_id = get_subject_data(path, 'x', 'y', labels, id=i, gain=gain)
        details += epoch_id[-1]     # ()
        epochs.extend(epoch_id[:-1])
    print('details:\n', details)
    step = len(labels)
    stop = len(epochs) - step + 1
    lowest = np.min(details)
    if choice > lowest:
        choice = lowest
        print('warning: the number of choice is bigger than the number of sleep epochs, '
              'it has been corrected to {}'.format(choice))
    for i in range(step):
        s = slice(i, stop+i, step)
        epochs[i] = np.random.permutation(np.hstack(epochs[s]))[:choice]
    epochs = epochs[:step]
    epochs = np.asarray(epochs)
    return epochs


def choose_files(dataset_id, choice=40, age_mode='whole', inventory_path=None):
    '''
    return chosen file id
    :param dataset_id:
    :param choice:
    :param age_mode: only valid in term of dataset_id == 1, in ('whole','young',None)
    :param inventory_path: will skip fetch inventory's path if have given
    :return:
    '''
    if inventory_path is None:
        d = get_info(dataset_id)
        inventory = pd.read_csv(d['inventory'])
    else:
        inventory = pd.read_csv(inventory_path)
    #inventory = pd.read_csv(os.sep.join(['MyFiles', 'auto_upload_20201007034912', 'inventory.csv']))
    file_id = inventory.index.values
    if choice > len(file_id):
        raise Exception('the number of choice[{}] is bigger than the number of '
                        'dataset files[{}]'.format(choice, len(file_id)))
    else:
        if dataset_id == 1:
            try:
                if age_mode == 'young':
                    file_id = inventory.sort_values(by=['age'], axis=0, ascending=True).head(choice).index.values
                    return file_id
                elif age_mode is None:
                    return np.random.permutation(file_id)[:choice]
                elif age_mode == 'whole':  # age-num(file):(24, 35]---39，(35, 55]---27, (55, 75]---55, (75, 101]---32
                    young = inventory[inventory['age'] <= 35].index
                    mid_aged = inventory[(inventory['age'] > 35) & (inventory['age'] <= 55)].index
                    old = inventory[(inventory['age'] > 55) & (inventory['age'] <= 75)].index
                    super_old = inventory[inventory['age'] > 75].index
                    ageS = AgeStructure()
                    ageS.mid_aged = int(round(choice * len(mid_aged) / 153))
                    ageS.old = int(round(choice * len(old) / 153))
                    ageS.super_old = int(round(choice * len(super_old) / 153))
                    ageS.young = choice - ageS.mid_aged - ageS.old - ageS.super_old
                    young = np.random.permutation(young)[:ageS.young]
                    mid_aged = np.random.permutation(mid_aged)[:ageS.mid_aged]
                    old = np.random.permutation(old)[:ageS.old]
                    super_old = np.random.permutation(super_old)[:ageS.super_old]
                    return np.hstack((young, mid_aged, old, super_old))
                else:
                    raise Exception('parameter: the value of \'age_mode\' is invalid')
            except Exception as e:
                raise e
        elif dataset_id == 2 or dataset_id == 0:
            return np.random.permutation(file_id)[:choice]


def prepare_subjects(dataset_id, inventory_path=None):
    '''
    as for SC&ST dataset, return the first night indices of subjects which has coupled nights, as for personal dataset,
    means quite small
    :param dataset_id:
    :param inventory_path: will skip fetch inventory's path if have given
    :return:
    '''
    if inventory_path is None:
        d = get_info(dataset_id)
        inventory = pd.read_csv(d['inventory'])
    else:
        inventory = pd.read_csv(inventory_path)
    #inventory = pd.read_csv(os.sep.join(['MyFiles', 'auto_upload_20201007034912', 'inventory.csv']))
    if dataset_id == 1:  # because of missed files, need to get coupled data's first file id
        single_file_id = []
        i = 0
        first_night_id = []
        while i < len(inventory):
            if inventory.iloc[i, 1] == 13 or inventory.iloc[i, 1] == 36:
                single_file_id.append(i)
                i += 1
            elif inventory.iloc[i, 1] == 52:
                single_file_id.append(i)
                i += 1
                break
            else:
                first_night_id.append(i)
                i += 2
        while i < len(inventory):
            first_night_id.append(i)
            i += 2
        inventory = inventory.iloc[first_night_id]
    elif dataset_id == 2:
        first_night_id = (np.arange(int(len(inventory) / 2)) * 2).astype(np.int)
        inventory = inventory.iloc[first_night_id]
    #print('{} has {} valid subjects'.format(d['dataset_name'], len(ss_id)))
    return inventory.index.values   # DF has two Index----RangeIndex(has to_numpy()), Int64Index, here is Int64Index


# SC&ST data have obvious subject-basis nature
def choose_subjects(dataset_id, indices, choice=20, age_mode='young', inventory_path=None):
    '''
    return chosen subjects' file indices.
    as for SC&ST dataset, return the first night indices of chosen subjects
    :param dataset_id:
    :param indices: function prepare_subjects returned or such as it
    :param choice:
    :param age_mode: valid in term of dataset_id == 1, in ('whole','young',None)
    :param inventory_path: will skip fetch inventory's path if have given
    :return:
    '''
    if choice > len(indices):
        raise Exception('the number of choice[{}] is bigger than the number of subjects[{}]'.
                        format(choice, len(indices)))
    else:
        if inventory_path is None:
            d = get_info(dataset_id)
            inventory = pd.read_csv(d['inventory'])
        else:
            inventory = pd.read_csv(inventory_path)
        # inventory = pd.read_csv(os.sep.join(['MyFiles', 'auto_upload_20201007034912', 'inventory.csv']))
        if dataset_id == 1:
            inventory = inventory.iloc[indices]
            try:
                if age_mode == 'young':
                    first_night_id = inventory.sort_values(by=['age'], axis=0, ascending=True).head(choice).index.values
                    return first_night_id
                elif age_mode is None:
                    return np.random.permutation(indices)[:choice]
                elif age_mode == 'whole':   # age-num(subject):(24, 35]---19，(35, 55]---13, (55, 75]---27, (75, 101]---16
                    young = inventory[inventory['age'] <= 35].index
                    mid_aged = inventory[(inventory['age'] > 35) & (inventory['age'] <= 55)].index
                    old = inventory[(inventory['age'] > 55) & (inventory['age'] <= 75)].index
                    super_old = inventory[inventory['age'] > 75].index
                    ageS = AgeStructure()
                    ageS.mid_aged = int(round(choice * len(mid_aged) / 75))
                    ageS.old = int(round(choice * len(old) / 75))
                    ageS.super_old = int(round(choice * len(super_old) / 75))
                    ageS.young = choice - ageS.mid_aged - ageS.old - ageS.super_old
                    young = np.random.permutation(young)[:ageS.young]
                    mid_aged = np.random.permutation(mid_aged)[:ageS.mid_aged]
                    old = np.random.permutation(old)[:ageS.old]
                    super_old = np.random.permutation(super_old)[:ageS.super_old]
                    return np.hstack((young, mid_aged, old, super_old))
                else:
                    raise Exception('parameter: the value of \'age_mode\' is invalid')
            except Exception as e:
                raise e
        elif dataset_id == 2 or dataset_id == 0:
            return np.random.permutation(indices)[:choice]


def gen_experiment_set(dataset_id, archive_dir, split, choice_mode, repeats, nfolds, age_mode=None,
                       epoch_f_choice=None, labels=None, gain=0):
    '''
    generate random selection of the whole experiment at once
    :param dataset_id:
    :param archive_dir:
    :param split: tuple-like. (train_number, val_number[, test_number]). the number's counting unit is
                   based of 'train_mode'. note that 'subject' mode and 'file' mode are record of file,
                   'epoch' mode are record of epoch
    :param choice_mode: in ('subject', 'file', 'epoch'),
    :param repeats:
    :param nfolds:
    :param age_mode: valid in term of 'dataset_id'==1. in ('young','whole', None)
    :param epoch_f_choice: valid in term of 'choice_mode' == 'epoch', is a int.
                            choose corresponding files first before choosing sleep epochs
    :param labels: valid in term of 'choice_mode' == 'epoch', choose corresponding epoch data as for labels
    :param gain: valid in term of 'choice_mode' == 'epoch', is convenient to indicate original file
    :return:
    '''

    def _save_split_indices(p_split, ind, repeat_id):
        m_shape = list(ind.shape)
        m_shape[-1] = p_split[0]
        train = np.zeros(([nfolds] if nfolds > 1 else []) + m_shape, dtype=np.int)
        m_shape.pop(-1)
        m_shape.append(p_split[1])
        val = np.zeros(([nfolds] if nfolds > 1 else []) + m_shape, dtype=np.int)
        test = None
        if len(p_split) == 3:
            aux_id = np.arange(ind.shape[-1])
            np.random.shuffle(aux_id)
            s1_te = aux_id[:p_split[2]]
            s2_te = aux_id[p_split[2]:]
            test = ind.T[s1_te].T      # because 'epoch' mode has 2D-indices
            ind = ind.T[s2_te].T
            if choice_mode == 'subject':
                test = np.vstack((test, test + 1)).transpose().ravel()
        if nfolds > 1:
            aux_id = np.arange(ind.shape[-1])
            for i in range(nfolds):
                s1 = np.arange(i*p_split[1], (i+1)*p_split[1])
                s2 = np.setdiff1d(aux_id, s1)
                val[i] = ind.T[s1].T
                train[i] = ind.T[s2].T
        else:
            val = ind.T[:p_split[1]].T
            train = ind.T[p_split[1]:].T
        if choice_mode == 'subject':
            ori_shape = (val.shape[:-1] + (val.shape[-1] * 2,), train.shape[:-1] + (train.shape[-1] * 2,))
            val = val.reshape(val.shape + (1,))
            train = train.reshape(train.shape + (1,))
            val = np.concatenate((val, val + 1), axis=-1).reshape(ori_shape[0])
            train = np.concatenate((train, train + 1), axis=-1).reshape(ori_shape[1])
        d = {
            'train': train,
            'val': val
        }
        if test is not None:
            d['test'] = test
        if labels is not None:
            d['labels'] = labels
        if gain:
            d['10_exponent'] = gain
        file_name = os.sep.join([archive_dir, 'No.{}_repeat.npz'.format(repeat_id + 1)])
        np.savez(file_name, **d)

    def _rectify_split(p_split, new_sum):
        old_sum = np.sum(p_split)
        v = int(p_split[1] * new_sum / old_sum)
        if nfolds > 1:
            tr = (nfolds - 1) * v
        else:
            tr = int(p_split[0] * new_sum / old_sum)
        if len(p_split) == 3:
            te = new_sum - v - tr
            return tr, v, te
        else:
            return tr, v

    print('====================================== split data begin =======================================')
    split = np.array(split, dtype=np.int)
    choice = np.sum(split)
    if choice_mode == 'subject':
        inventory = prepare_subjects(dataset_id)
        for i in range(repeats):
            indices = choose_subjects(dataset_id, inventory, choice, age_mode)
            indices = np.random.permutation(indices)
            _save_split_indices(split, indices, i)
    elif choice_mode == 'file':
        for i in range(repeats):
            indices = choose_files(dataset_id, choice, age_mode)
            indices = np.random.permutation(indices)
            _save_split_indices(split, indices, i)
    elif choice_mode == 'epoch':
        for i in range(repeats):
            if epoch_f_choice is not None:
                file_id = choose_files(dataset_id, epoch_f_choice, age_mode)
            else:
                file_id = None
            indices = choose_epochs(dataset_id, choice, gain, labels, file_id)   # ndarray, shape(len(labels),np.sum(split))
            for j in range(len(indices)):
                indices[j] = np.random.permutation(indices[j])
            if choice == indices.shape[-1]:
                _save_split_indices(split, indices, i)
            else:
                split_new = _rectify_split(split, indices.shape[-1])
                print('new partition scheme is: {}'.format(split_new))
                _save_split_indices(split_new, indices, i)
    print('====================================== split data end =======================================')


# class weight only is as for train set
# sample weight is as for train set and validation set
def fetch_split_set(inv, d_dir, a_dir, repeat_id, fold_id=None):
    inventory = pd.read_csv(inv)
    files = glob.glob(os.sep.join([a_dir, '*']))
    files.sort()
    file = files[repeat_id]
    info = {}
    with np.load(file) as f:
        info['train'] = f['train']
        info['val'] = f['val']
        if f.get('test', None) is not None:
            info['test'] = f['test']
    if fold_id is not None:
        train = inventory['file_name'].loc[info['train'][fold_id]].values.astype(str).tolist()
        class_details = inventory['5_details'].loc[info['train'][fold_id]].values.tolist()
        trans_details = inventory['2_details'].loc[info['train'][fold_id]].values.tolist()
        validation = inventory['file_name'].loc[info['val'][fold_id]].values.astype(str).tolist()
    else:
        train = inventory['file_name'].loc[info['train']].values.astype(str).tolist()
        class_details = inventory['5_details'].loc[info['train']].values.tolist()
        trans_details = inventory['2_details'].loc[info['train']].values.tolist()
        validation = inventory['file_name'].loc[info['val']].values.astype(str).tolist()
    class_details = np.array(list(map(lambda sli: sli.strip(r'[|]').split()[:-1], class_details))).astype(np.int)
    class_details = np.sum(class_details, axis=0)
    trans_details = np.array(list(map(lambda sli: sli.strip(r'[|]').split(), trans_details))).astype(np.int)
    trans_details = np.sum(trans_details, axis=0)
    test = None
    if info.get('test', None) is not None:
        test = inventory['file_name'].loc[info['test']].values.astype(str).tolist()
        for i, te in enumerate(test):
            test[i] = os.sep.join([d_dir, te])
    for i, tr in enumerate(train):
        train[i] = os.sep.join([d_dir, tr])
    for i, v in enumerate(validation):
        validation[i] = os.sep.join([d_dir, v])
    return train, validation, test, class_details, trans_details    # the former three are list, the last two is ndarray


# split is tuple/list
# mode in ('epoch', 'subject', 'file')
# get file paths
def gen_temprory_set(dataset_id, split, mode, age_mode=None, epoch_f_choice=None, labels=None, gain=5, path=None):

    def _split_indices(l_split, ind):
        sets = []
        re_ind = ind
        for s in l_split:
            aux_id = np.arange(re_ind.shape[-1])
            aux_id = np.random.permutation(aux_id)
            choose = re_ind.T[aux_id[:s]].T   #'epoch' mode has 2D-indices
            re_ind = re_ind.T[aux_id[s:]].T
            if mode == 'subject':
                choose = np.vstack((choose, choose + 1)).transpose().ravel()
            sets.append(choose)
            if path is not None:
                np.savez(path + '.npz', *sets)     # when loading --- dict(f)
        return sets

    def _rectify_split(l_split, new_sum):
        old_sum = sum(l_split)
        new = []
        for s in l_split:
            new.append(int(round(s * new_sum / old_sum)))
        if sum(new) != new_sum:
            new[-1] -= sum(new) - new_sum
        return new

    num = int(sum(split))
    split = list(split)
    if mode == 'subject':
        inventory = prepare_subjects(dataset_id)
        indices = choose_subjects(dataset_id, inventory, num, age_mode)
        indices = np.random.permutation(indices)
        sets = _split_indices(split, indices)
    elif mode == 'file':
        indices = choose_files(dataset_id, num, age_mode)
        indices = np.random.permutation(indices)
        sets = _split_indices(split, indices)
    elif mode == 'epoch':
        if epoch_f_choice is not None:
            file_id = choose_files(dataset_id, epoch_f_choice, age_mode)
        else:
            file_id = None
        indices = choose_epochs(dataset_id, num, gain, labels, file_id)  # ndarray, shape(len(labels),np.sum(split))
        for j in range(len(indices)):
            indices[j] = np.random.permutation(indices[j])
        if num == indices.shape[-1]:
            sets = _split_indices(split, indices)
        else:
            split_new = _rectify_split(split, indices.shape[-1])
            print('new partition scheme is: {}'.format(split_new))
            sets = _split_indices(split_new, indices)
    if mode == 'epoch':
        info = get_info(dataset_id)
        inv = pd.read_csv(info['inventory'])
        d_dir = info['data_dir']
        d = {'set': [], 'epoch': [], 'label': []}
        for s in sets:
            l = np.tile(np.array(labels), (s.shape[-1], 1)).transpose().ravel()
            e = s.ravel()
            f, e, l = resolve_index_gain(e, gain, l)
            f = [os.sep.join([d_dir, n]) for n in inv['file_name'].loc[f].values.astype(str).tolist()]
            d['set'].append(f)
            d['epoch'].append(e)
            d['label'].append(l)
    else:
        info = get_info(dataset_id)
        inv = pd.read_csv(info['inventory'])
        d_dir = info['data_dir']
        d = {'set': [], 'stat': [], 'trans_stat': []}
        for s in sets:
            d['set'].append([os.sep.join([d_dir, n]) for n in inv['file_name'].loc[s].values.astype(str).tolist()])
            class_details = inv['5_details'].loc[s].values.tolist()
            trans_details = inv['2_details'].loc[s].values.tolist()
            class_details = np.array(list(map(lambda sli: sli.strip(r'[|]').split()[:-1], class_details))).astype(
                np.int)
            class_details = np.sum(class_details, axis=0)
            trans_details = np.array(list(map(lambda sli: sli.strip(r'[|]').split(), trans_details))).astype(np.int)
            trans_details = np.sum(trans_details, axis=0)
            d['stat'].append(class_details)
            d['trans_stat'].append(trans_details)
    return d


def save_trainer(model, path, lr=None, opt_ws=None):
    if model.built:
        names = []
        weights = []
        w_names = []
        for l in model.layers:
            names.append(l.name)
            weights.append([])
            w_names.append([])
            for w in l.weights:
                weights[-1].append(w.numpy())
                w_names[-1].append(w.name)
        d = {'model_layer_names': names, 'model_layer_weights': weights, 'model_layer_weight_names': w_names}
        if lr is not None:
            d.update({'lr': lr})
        if opt_ws is not None:
            d.update({'opt_weights': opt_ws})
        np.savez(path + '.npz', **d)
    else:
        raise Exception('{} model has not been built'.format(model.name))


# when load initial weights, the variances like moving_mean and moving_variance of BN layer are except
def load_initial_weights(model, path):
    if model.built:
        with np.load(path, allow_pickle=True) as f:
            names = list(f['names'])
            weights = list(f['weights'])
        bn = {'moving_mean': [], 'moving_variance': [], 'moving_stddev': [], 'beta': [], 'gamma': [], 'renorm_mean': [],
              'renorm_stddev': []}
        ln = {'beta': [], 'gamma': []}
        for i, n in enumerate(names):
            if re.match(r'.*bn.*moving_mean.*', n):
                bn['moving_mean'].append(i)
            elif re.match(r'.*bn.*moving_variance.*', n):
                bn['moving_variance'].append(i)
            elif re.match(r'.*bn.*moving_stddev.*', n):
                bn['moving_stddev'].append(i)
            elif re.match(r'.*bn.*beta.*', n):
                bn['beta'].append(i)
            elif re.match(r'.*bn.*gamma.*', n):
                bn['gamma'].append(i)
            elif re.match(r'.*bn.*renorm_mean.*', n):
                bn['renorm_mean'].append(i)
            elif re.match(r'.*bn.*renorm_stddev.*', n):
                bn['renorm_stddev'].append(i)
            elif re.match(r'.*ln.*beta.*', n):
                ln['beta'].append(i)
            elif re.match(r'.*ln.*gamma.*', n):
                ln['gamma'].append(i)
            else:
                pass
        ones_ini = bn['moving_variance'] + bn['moving_stddev'] + bn['renorm_stddev']
        zeros_ini = bn['moving_mean'] + bn['renorm_mean']
        ones_ini.sort()
        zeros_ini.sort()
        # to assign values which should be restored
        for idx in ones_ini:
            w = weights[idx]
            weights[idx] = np.ones_like(w)
        for idx in zeros_ini:
            w = weights[idx]
            weights[idx] = np.zeros_like(w)
        model.set_weights(weights)
    else:
        raise Exception('{} model has not been built, please build it first'.format(model.name))


def load_same_model_weights(model, path):
    if model.built:
        with np.load(path, allow_pickle=True) as f:
            d = dict(f)
            ws = d.get('model_layer_weights', None)
            if ws is not None:
                weights = []
                for w in ws:
                    weights += w
            else:
                weights = []
            opt_weights = d.get('opt_weights', None)
            if opt_weights is not None:
                opt_weights = list(opt_weights)
                if opt_weights and hasattr(opt_weights[0], '__iter__'):
                    for i in range(len(opt_weights)):
                        opt_weights[i] = list(opt_weights[i])
            else:
                opt_weights = []
        model.set_weights(weights)
        return opt_weights
    else:
        raise Exception('{} model has not been built, please build it first'.format(model.name))


def from_file_set_model_weights(model, path):
    with np.load(path, allow_pickle=True) as f:
        weights = f['model_layer_weights']
    if model.built:
        for i, l in enumerate(model.layers):
            if weights[i]:
                l.set_weights(weights[i])
    else:
        raise Exception('please infer model forward at first')


def from_value_set_model_weights(model, weights):
    if model.built:
        for i, l in enumerate(model.layers):
            if weights[i]:
                l.set_weights(weights[i])
    else:
        raise Exception('please infer model forward at first')


def select_layer_set_model_weights(model, layer_names, weights):
    names = []
    for l in model.layers:
        names.append(l.name)
    for n in layer_names:
        if n not in names:
            raise ValueError('{} is not one of {} model\'s layers'.format(n, model.name))
        idx = names.index(n)
        layer = model.layers[idx]
        if layer.built:
            if weights[idx]:
                layer.set_weights(weights[idx])
        else:
            raise Exception('please infer model forward at first')


# ==================================================================================================================
dataset_path = {
    0: ('C:\\EEG_data\\*.edf', 'C:\\EEG_data\\EEG_label\\*.csv'),
    1: ('D:\\sleep-expanded\\data_2018\\*PSG.edf', 'D:\\sleep-expanded\\data_2018\\*Hypnogram.edf'),
    2: ('D:\\sleep-expanded\\sleep-telemetry\\*PSG.edf', 'D:\\sleep-expanded\\sleep-telemetry\\*Hypnogram.edf')
}


def _info_archive(dataset, data_dir, output_dir, sampling_rate, selected_channels, x_paths):
    import pandas as pd
    filename = dataset + '_info.npz'
    inventory = os.sep.join([data_dir, 'inventory.csv'])
    df = pd.read_csv(inventory)
    stat1 = np.zeros((6,), dtype=np.int32)
    stat2 = np.zeros((6,), dtype=np.int32)
    stat3 = np.zeros((2,), dtype=np.int32)
    for i in df.index:
        path = os.sep.join([data_dir, df.iloc[i, 0]])
        stat_1, stat_2, stat_3 = get_subject_data(path, 'before_stat', 'after_stat', 'trans_stat')
        stat1 += stat_1
        stat2 += stat_2
        stat3 += stat_3
    d = {
        'data_dir': data_dir,
        'output_dir': output_dir,
        'sampling_rate': sampling_rate,
        'selected_channels': selected_channels,
        'data_number': len(x_paths),
        'dataset_name': dataset,
        'inventory': inventory,
        'before_stat': stat1,
        'after_stat': stat2,
        'trans_stat': stat3
    }
    np.savez(filename, **d)


#d = get_info(2)
#for k in d.keys():
#    print('{}: {}'.format(k, d[k]))

def demo(i):
    x_path = glob.glob(dataset_path[i][0])
    x_path.sort()
    print(len(x_path))
    dataset = dataset2name[i]
    data_dir = os.sep.join(['data', dataset])
    path = os.sep.join([data_dir, 'ST7011J0.npz'])
    print(get_subject_data(path, 'selected_channels'))
    output_dir = os.sep.join(['output', dataset])
    selected_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']
    _info_archive(dataset,data_dir,output_dir,100.,selected_channels, x_path)

#demo(2)


#path = 'data\\personal\\01.npz'
#x_, y_, W_stat = get_subject_data(path, 'x', 'y', ['W'], 1, 4)
#print(W_stat)


def demo2(i):
    import pandas as pd
    import ntpath
    dataset = dataset2name[i]
    data_dir = os.sep.join(['data', dataset])
    x_paths = glob.glob(dataset_path[i][0])
    x_paths.sort()
    inventory = os.sep.join([data_dir, 'inventory.csv'])
    if i == 1:
        info = pd.read_excel(SC_subjects_path)
        filenames = [ntpath.basename(path).replace('-PSG.edf', '.npz') for path in x_paths]
        df = pd.DataFrame({'file_name': filenames, 'subject': info['subject'], 'night': info['night'],
                           'age': info['age'], 'lights_off_time': info['LightsOff']})
        df.to_csv(inventory, index=False)
    elif i == 2:
        info = pd.read_excel(ST_subjects_path)
        info.drop([0], axis=0, inplace=True)
        info.reset_index(drop=True, inplace=True)
        #print(info)
        df = pd.DataFrame(columns=('file_name', 'subject', 'medicine', 'lights_off_time'))
        filenames = [ntpath.basename(path).replace('-PSG.edf', '.npz') for path in x_paths]
        df['file_name'] = filenames
        df['subject'] = np.tile(info['Subject - age - sex'].to_numpy(), [2, 1]).transpose().ravel()
        for j in info.index:
            if info.iloc[j, 3] == 1:
                df.iloc[j*2, 2] = 'Placebo'
                df.iloc[j*2, 3] = info.iloc[j, 4]
                df.iloc[j*2+1, 2] = 'Temazepam'
                df.iloc[j*2+1, 3] = info.iloc[j, 6]
            else:
                df.iloc[j*2, 2] = 'Temazepam'
                df.iloc[j*2, 3] = info.iloc[j, 6]
                df.iloc[j*2+1, 2] = 'Placebo'
                df.iloc[j*2+1, 3] = info.iloc[j, 4]
        df.to_csv(inventory, index=False)
    elif i == 0:
        df = pd.DataFrame({'file_name': [ntpath.basename(path).replace('.edf', '.npz') for path in x_paths]})
        df.to_csv(inventory, index=False)


#demo2(1)


def demo3(i):
    import pandas as pd
    dataset = dataset2name[i]
    data_dir = os.sep.join(['data', dataset])
    info = os.sep.join([data_dir, 'inventory.csv'])
    inventory = pd.read_csv(info)
    file_id = inventory.index.to_numpy()
    single_file_id = []
    i = 0

    first_night_id = []
    while i < len(inventory):
        if inventory.iloc[i, 1] == 13 or inventory.iloc[i, 1] == 36:
            single_file_id.append(i)
            i += 1
        elif inventory.iloc[i, 1] == 52:
            single_file_id.append(i)
            i += 1
            break
        else:
            first_night_id.append(i)
            i += 2
    while i < len(inventory):
        first_night_id.append(i)
        i += 2
    file_id = np.setdiff1d(file_id, single_file_id)
    inventory = inventory.iloc[first_night_id]
    ss_id = inventory['subject'].to_numpy()

    young = inventory[inventory['age'] <= 35].index
    mid_aged = inventory[(inventory['age'] > 35) & (inventory['age'] <= 55)].index
    old = inventory[(inventory['age'] > 55) & (inventory['age'] <= 75)].index
    super_old = inventory[inventory['age'] > 75].index
    print(len(young), len(mid_aged), len(old), len(super_old))


#demo3(1)


def demo4(i):
    import pandas as pd
    dataset = dataset2name[i]
    data_dir = os.sep.join(['data', dataset])
    inventory = os.sep.join([data_dir, 'inventory.csv'])
    df = pd.read_csv(inventory)
    epochs_list = np.zeros((len(df),), dtype=np.int)
    details = {'n_class': [], 'trans': []}
    for i in df.index:
        path = os.sep.join([data_dir, df.iloc[i, 0]])
        with np.load(path) as f:
            x = f['x']
            epochs_list[i] = x.shape[0]
            details['n_class'].append(f['after_stat'])
            details['trans'].append(f['trans_stat'])

    df['epochs'] = epochs_list
    df = pd.concat([df, pd.DataFrame(details)], ignore_index=True, axis=1)
    df.to_csv(inventory, index=False)


#demo4(2)


def save_split_indices(split, nfolds, indices, repeat_id, dir, mode=None, labels=None, gain=0):
    m_shape = list(indices.shape)
    m_shape[-1] = split[0]
    train = np.zeros([nfolds] + m_shape, dtype=np.int)
    m_shape.pop(-1)
    m_shape.append(split[1])
    val = np.zeros([nfolds] + m_shape, dtype=np.int)
    test = None
    if len(split) == 3:
        aux_id = np.arange(indices.shape[-1])
        s1_te = np.random.permutation(aux_id)[:split[2]]
        s2_te = np.setdiff1d(aux_id, s1_te)
        test = indices.T[s1_te].T
        indices = indices.T[s2_te].T
        if mode == 'subject':
            test = np.vstack((test, test + 1)).transpose().ravel()
    aux_id = np.arange(indices.shape[-1])
    for i in range(nfolds):
        s1 = np.arange(i*split[1], (i+1)*split[1])
        s2 = np.setdiff1d(aux_id, s1)
        val[i] = indices.T[s1].T
        train[i] = indices.T[s2].T
    if mode == 'subject':
        ori_shape = (val.shape[:-1] + (val.shape[-1] * 2,), train.shape[:-1] + (train.shape[-1] * 2,))
        val = val.reshape(val.shape + (1,))
        train = train.reshape(train.shape + (1,))
        val = np.concatenate((val, val + 1), axis=-1).reshape(ori_shape[0])
        train = np.concatenate((train, train + 1), axis=-1).reshape(ori_shape[1])
    d = {
        'train': train,
        'val': val
    }
    if test is not None:
        d['test'] = test
    if labels is not None:
        d['labels'] = labels
    if gain:
        d['10_exponent'] = gain
    file_name = os.sep.join([dir, 'No.{}_repeat.npz'.format(repeat_id + 1)])
    np.savez(file_name, **d)


def rectify_split(split, nfolds, new_sum):
    old_sum = np.sum(split)
    v = int(split[1] * new_sum / old_sum)
    tr = (nfolds - 1) * v
    if len(split) == 3:
        te = new_sum - v - tr
        return tr, v, te
    else:
        return tr, v


def gen_data(dataset_id, split, train_mode, repeats, nfolds, inventory=None, age_mode=None, epoch_f_choice=None,
                labels=['W', 'N1'], gain=5):
    assert (split[0] + split[1]) // nfolds == split[1], '{} folds cross validation\'s the division ' \
                                                             'of dataset is wrong!'.format(nfolds)
    split = np.array(split, dtype=np.int)
    if dataset_id == 1:
        if age_mode is not None:
            archive_dir = os.sep.join(['{}repeats_{}folds'.format(repeats, nfolds), train_mode, age_mode])
        else:
            archive_dir = os.sep.join(['{}repeats_{}folds'.format(repeats, nfolds), train_mode, 'None'])
    else:
        archive_dir = os.sep.join(['{}repeats_{}folds'.format(repeats, nfolds), train_mode])
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
    choice = np.sum(split)
    if train_mode == 'subject':
        if inventory is None:
            inventory = prepare_subjects(dataset_id)
        for i in range(repeats):
            indices = choose_subjects(dataset_id, inventory, choice, age_mode)
            indices = np.random.permutation(indices)
            save_split_indices(split, nfolds, indices, i, archive_dir, train_mode)
    elif train_mode == 'file':
        for i in range(repeats):
            indices = choose_files(dataset_id, choice, age_mode)
            indices = np.random.permutation(indices)
            save_split_indices(split, nfolds, indices, i, archive_dir)
    elif train_mode == 'epoch':
        for i in range(repeats):
            if epoch_f_choice is not None:
                file_id = choose_files(dataset_id, epoch_f_choice, age_mode)
            else:
                file_id = None
            indices = choose_epochs(dataset_id, choice, gain, labels, file_id)
            for j in range(len(indices)):
                indices[j] = np.random.permutation(indices[j])
            if choice != indices.shape[-1]:
                split_new = rectify_split(split, nfolds, indices.shape[-1])
                print('new partition scheme is: {}'.format(split_new))
                save_split_indices(split_new, nfolds, indices, i, archive_dir, labels, gain)
            else:
                save_split_indices(split, nfolds, indices, i, archive_dir, labels, gain)


#gen_data(1, (12, 2, 3), 'subject', 5, 7, age_mode='young', epoch_f_choice=12)


def read_dataset(mode, methed):
    path = os.sep.join(['5repeats_7folds', mode, methed, 'No.1_repeat.npz'])
    with np.load(path) as file:
        if file.get('test', None) is not None:
            d = dict(file)
    return d


#d = read_dataset('subject', 'young')
#x, y = from_subjects_fetch_epochs(*resolve_index_gain(d['train'][0, 0, :], int(d['10_exponent'])), dataset_id=1)
#print(d)





