# -*- coding: utf-8 -*-
import argparse
import os
import glob
import shutil
import pyedflib
import ntpath
import gc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from math import ceil
from data_reader import EDFReader


dataset_name = {
    0: 'personal',
    1: 'old_sleep-EDF-data_2018',
    2: 'sleep-EDF-sleep-telemetry'
}


origin_dir = os.sep.join(['E:', 'EEG_dataset', 'original'])


dataset_path = {
    0: (os.sep.join([origin_dir, 'EEG_data', '*.edf']), os.sep.join([origin_dir, 'EEG_data', 'EEG_label', '*.csv'])),
    1: (os.sep.join([origin_dir, 'sleep-expanded', 'data_2018', '*PSG.edf']), os.sep.join([origin_dir, 'sleep-expanded',
                                                                                           'data_2018',
                                                                                           '*Hypnogram.edf'])),
    2: (os.sep.join([origin_dir, 'sleep-expanded', 'sleep-telemetry', '*PSG.edf']), os.sep.join(
        [origin_dir, 'sleep-expanded', 'sleep-telemetry', '*Hypnogram.edf']))
}


SC_subjects_path = os.sep.join([origin_dir, 'sleep-expanded', 'SC-subjects.xls'])
ST_subjects_path = os.sep.join([origin_dir, 'sleep-expanded', 'ST-subjects.xls'])


# Label values
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
    5: 'UNKNOWN',
}


sleep_EDF_label = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4,
    'Sleep stage ?': 5,
    'Movement time': 5
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
RESERVED_MIN = 30


class DataPre(object):
    def __init__(self, data_dir, output_dir, dataset, x_paths, y_paths):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.dataset = dataset
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.selected_channels = None
        self.sampling_rate = 0.
        self.samples_per_epoch = 0
        self.inventory = os.sep.join([data_dir, 'inventory.csv'])    # prepared beforehand
        self.stat1 = np.zeros((len(class_dict),), dtype=np.int32)  # the primitive stat of pairs of data and tag
        self.stat2 = np.zeros((len(class_dict),), dtype=np.int32)  # the experiment stat
        self.stat3 = np.zeros((2,), dtype=np.int32)   # the trans stat

    def _union(self, seq):
        head = []
        tail = []
        j = 0
        i = len(seq) - 1
        while j <= i:
            f = seq[j]
            if seq[i] - i + j == f:
                head.append(f)
                tail.append(seq[i])
                j = i + 1
                i = len(seq) - 1
                continue
            else:
                i -= 1
        sum = 0
        for m, n in zip(head, tail):
            sum += n - m + 1
        assert sum == len(seq)
        return head, tail

    def _info_archive(self):
        filename = self.dataset + '_info.npz'
        d = {
           'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'sampling_rate': self.sampling_rate,
            'selected_channels': self.selected_channels,
            'data_number': len(self.x_paths),
            'dataset_name': self.dataset,
            'inventory': self.inventory,
            'before_stat': self.stat1,
            'after_stat': self.stat2,
            'trans_stat': self.stat3
        }
        np.savez(filename, **d)

    def _read_save(self, have_shown, x_file, y_file, selected_channels, file_id):
        h_reader_x = EDFReader(x_file)
        head_x = h_reader_x.read_head()
        d_reader = pyedflib.EdfReader(x_file)
        if not have_shown:
            print('the data file contain signals as follow:\n{}\nand their sampling rates are:\n{}'.format(
                head_x['labels'], head_x['sampling_rates']))
            num_chs = int(input('please select channels, note that these channels should have the same sampling'
                                ' rates. Now please input the number of channels:\n'))
            for n in range(num_chs):
                selected_channels.append(int(input('No.{} channel\'s number (start from 0) is:'.format(n + 1))))
            have_shown = True
            self.selected_channels = np.asarray(d_reader.getSignalLabels())[selected_channels]
            print('selected channels are: \n{}'.format(self.selected_channels))
            self.sampling_rate = head_x['sampling_rates'][selected_channels[0]]
            self.samples_per_epoch = int(self.sampling_rate * EPOCH_SEC_SIZE)
            print('======================================loading data========================================')

        x = [d_reader.readSignal(k) for k in selected_channels]  # [1D_array,...]  dtype=float64

        del d_reader
        gc.collect()

        x = np.asarray(x)
        y = []
        x_date_time = datetime.strptime(head_x['date'] + ' ' + head_x['time'], '%Y-%m-%d %H:%M:%S')
        y_abandon = 0  # the unit of a epoch, record y's abandoned epochs before processing

        if self.dataset != dataset_name[0]:
            lights_off_time = pd.read_csv(self.inventory).iloc[file_id, -1]   # str
            h_reader_y = EDFReader(y_file)
            head_y = h_reader_y.read_head()
            time_bias, anns, _ = tuple(zip(*h_reader_y.read_data()))  # time_bias---(xxx,...)  anns---([(,,[]),...],...)
            save_df = pd.DataFrame(np.asarray(anns[0]), columns=['start_time', 'duration', 'sleep_seg'])
            filedir = os.sep.join([self.data_dir, 'label'])
            if not os.path.exists(filedir):
                os.mkdir(filedir)
            name = ntpath.basename(x_file).rstrip('-PSG.edf')
            filename = os.sep.join([filedir, name + '.csv'])
            save_df.to_csv(filename, index=False)
            for j in range(len(time_bias)):  # in fact, annotation file only has one record
                # the first tal in per record is a time keeping tal containing no annotations after view
                onset = (np.asarray(anns[j][1:])[:, 0] - time_bias[j]).astype(np.int32)
                assert (onset % EPOCH_SEC_SIZE == 0).all(), 'annotations in form of EDF+ have some durations not' \
                                                            ' {} divisive except file tail'.format(EPOCH_SEC_SIZE)
                duration = (np.asarray(anns[j][1:])[:, 1]).astype(np.int32) // EPOCH_SEC_SIZE
                y_tail_truncated = (anns[j][-1][1] % EPOCH_SEC_SIZE != 0.)
                for z, k in enumerate(anns[j][1:]):
                    # the first one in annotations per tal is sleep stage after view
                    y.extend([sleep_EDF_label[k[2][0]]] * duration[z])
                if y_tail_truncated:
                    y_abandon += 1
            filename = os.sep.join([self.data_dir, name + '.npz'])
            y_date_time = datetime.strptime(head_y['date'] + ' ' + head_y['time'], '%Y-%m-%d %H:%M:%S')
            lights_off_time = datetime.strptime(head_y['date'] + ' ' + lights_off_time,
                                                '%Y-%m-%d %H:%M:%S')  # datetime.datetime
            assert y_date_time == x_date_time, \
                '{} file\'s time is different between data({}) and label({})'.format(y_file,
                                                                                     str(x_date_time),
                                                                                     str(y_date_time))
            if lights_off_time < y_date_time:   # 1----lighting off before recording; 2----lighting off time is the next day
                lights_off_time += timedelta(days=1)
            if (lights_off_time - y_date_time).seconds < 43200:  # scene 2
                align = ceil((lights_off_time - y_date_time).seconds / EPOCH_SEC_SIZE)
                y_abandon += align
                y = y[align:]
                x = x[:, align * self.samples_per_epoch:]
                date_time = str(y_date_time + timedelta(seconds=align * EPOCH_SEC_SIZE))
            else:     # scene 1
                date_time = str(y_date_time)
        else:
            head_y = None
            y_df = pd.read_csv(y_file)
            timer = y_df['start_time'].values
            name = ntpath.basename(x_file).replace('.edf', '.npz')
            filename = os.sep.join([self.data_dir, name])
            y_date_time = datetime.strptime(head_x['date'] + ' ' + timer[0],
                                            '%Y-%m-%d %H:%M:%S')
            if int((datetime.strptime(timer[-1], '%H:%M:%S') -
                    datetime.strptime(timer[-2], '%H:%M:%S')).seconds) % EPOCH_SEC_SIZE != 0:
                y_abandon += 1
                y = [personal_label[k] for k in y_df['sleep_seg'].tolist()[:-1]]
            else:
                y = [personal_label[k] for k in y_df['sleep_seg'].tolist()]
            date_time = str(x_date_time)

            if x_date_time > y_date_time:
                # most EDF/EDF+ files' duration at the unit of a second
                align = int((x_date_time - y_date_time).seconds)
                remainder = align % EPOCH_SEC_SIZE
                if remainder != 0:
                    M = int(ceil(align / EPOCH_SEC_SIZE))
                    x_front_bias = EPOCH_SEC_SIZE - remainder
                    date_time = str(x_date_time + timedelta(seconds=x_front_bias))
                    x_front_bias = int(x_front_bias * self.sampling_rate)
                    x = x[:, x_front_bias:]
                else:
                    M = align // EPOCH_SEC_SIZE
                    date_time = str(x_date_time)
                y = y[M:]
                y_abandon += M
            elif x_date_time < y_date_time:
                align = int((y_date_time - x_date_time).seconds)
                x_front_bias = align * self.sampling_rate
                x = x[x_front_bias:]
                date_time = str(y_date_time)

        # the data edf files in sleep-EDF may contain no unknown epochs' data after view
        # so, abandon unknown epochs and consider data's last incomplete epochs
        # because of data files' continuity, truncate directly
        N = x.shape[-1] // self.samples_per_epoch
        x = x[:, :N * self.samples_per_epoch]
        x = np.asarray(np.split(x, N, axis=-1))
        print('{}: {}'.format(x_file, x.shape))
        y = np.asarray(y, dtype=np.int32)
        y_idx = np.arange(len(y))
        x_idx = np.arange(len(x))

        # start to process
        selected_idx = np.intersect1d(x_idx, y_idx)    # intersect1d drop repeated
        # x and y are coupled
        before_stat = np.zeros((len(class_dict),), dtype=np.int32)
        for c in class_dict.keys():
            before_stat[c] = len(np.where(y[selected_idx] == c)[0])   # the indices from where method is as for selected_idx
            self.stat1[c] += before_stat[c]
        abandon_y_idx = list(np.setdiff1d(y_idx, selected_idx))
        abandon = [[], []]
        if len(abandon_y_idx) > 0:
            abandon = [[abandon_y_idx[0]], [abandon_y_idx[-1]]]
        # abandon unknown epochs in the middle of file, mid_unknown is index of selected_idx
        mid_unknown = list(np.where(y[selected_idx] == stage_dict['UNKNOWN'])[0])
        abandon_y_idx.extend(selected_idx[mid_unknown])
        abandon_y_idx.sort()
        h, t = self._union(selected_idx[mid_unknown])
        abandon[0].extend(h)
        abandon[1].extend(t)
        selected_idx = np.setdiff1d(selected_idx, selected_idx[mid_unknown])
        # abandon redundant 'W' epochs
        # may be discontinuous because of deleting 'UNKNOWN', in such case that discontinuous 'W' have become continuous
        temp = np.where(y[selected_idx] != stage_dict['W'])[0]
        start = 0
        end = len(selected_idx) - 1
        if self.dataset == dataset_name[0]:     # because public data has 'light-off-time'
            if temp[0] - RESERVED_MIN * 2 > -1:
                start = temp[0] - RESERVED_MIN * 2
                h, t = self._union(selected_idx[:start])
                abandon[0].extend(h)
                abandon[1].extend(t)
        if temp[-1] + RESERVED_MIN * 2 < len(selected_idx):
            end = temp[-1] + RESERVED_MIN * 2
            h, t = self._union(selected_idx[end + 1:])
            abandon[0].extend(h)
            abandon[1].extend(t)
        abandon_y_idx.extend(list(np.setdiff1d(selected_idx, selected_idx[start: end + 1])))
        selected_idx = selected_idx[start: end + 1]

        # may include part of another sleep
        if len(selected_idx) > 1250 and self.dataset == dataset_name[1]:
            # 'SC4761E0' file have 363 'W' epochs in the head
            aux1 = selected_idx.tolist()[:332:-1]     # promise 333 epochs' sleep record
            idx = 0
            temp_sum = 0
            while idx < 333:
                if len(aux1) != 0:
                    temp_sum += 1
                    if y[aux1.pop(0)] == stage_dict['W']:
                        idx += 1
                    else:
                        idx = 0
                else:
                    break
            if idx >= 333:
                aux2 = selected_idx[-(temp_sum + 1)::-1]
                idx = temp_sum
                for s in aux2:
                    if y[s] != stage_dict['W']:
                        break
                    else:
                        idx += 1
                temp_sum -= 333
                if idx - RESERVED_MIN * 2 < temp_sum:
                    start = -temp_sum
                else:
                    start = -(idx - RESERVED_MIN * 2)
                h, t = self._union(selected_idx[start:])
                abandon[0].extend(h)
                abandon[1].extend(t)
                abandon_y_idx.extend(list(selected_idx[start:]))
                selected_idx = selected_idx[:start]

        assert len(selected_idx) + len(abandon_y_idx) == len(y), '{} fail to select required epochs'.format(
            y_file)
        y = y[selected_idx]
        x = x[selected_idx]
        after_stat = np.zeros((len(class_dict),), dtype=np.int32)
        # 0 --- non-trans    1 --- trans
        trans_y = np.not_equal(np.concatenate(([-1], y[:-1]), axis=0), y).astype('int32')
        trans_stat = np.zeros((2,), dtype=np.int32)
        trans_stat[0] = len(np.where(trans_y == 0)[0])
        trans_stat[1] = len(np.where(trans_y == 1)[0])
        self.stat3[0] += trans_stat[0]
        self.stat3[1] += trans_stat[1]
        for c in class_dict.keys():
            after_stat[c] = len(np.where(y == c)[0])
            self.stat2[c] += after_stat[c]
        print('after selection: {}\nabandon label idx is:\n{}\nabandon total number is: {} + {}'.format(
            x.shape, abandon, y_abandon, len(abandon_y_idx)))
        print('start time is :{}'.format(date_time))

        save_dict = {
            'x': x,
            'y': y,
            'trans_y': trans_y,
            'fs': self.sampling_rate,
            'selected_channel': self.selected_channels,    # str ndarray
            # 'data_head': head_x,
            # 'annotation_head': head_y,
            'start_time': date_time,
            'before_stat': before_stat,
            'after_stat': after_stat,
            'trans_stat': trans_stat,
            'aligned_y_abandon': len(abandon_y_idx),    # aligned_y means y's start time is the same as x's
        }
        np.savez(filename, **save_dict)
        return have_shown, selected_channels, x.shape[0], after_stat, trans_stat

    def gen_inventory(self):
        if self.dataset == dataset_name[1]:
            info = pd.read_excel(SC_subjects_path)
            filenames = [ntpath.basename(path).replace('-PSG.edf', '.npz') for path in self.x_paths]
            df = pd.DataFrame({'file_name': filenames, 'subject': info['subject'], 'night': info['night'],
                               'age': info['age'], 'lights_off_time': info['LightsOff']})
            df.to_csv(self.inventory, index=False)
        elif self.dataset == dataset_name[2]:
            info = pd.read_excel(ST_subjects_path)
            info.drop([0], axis=0, inplace=True)
            info.reset_index(drop=True, inplace=True)
            # print(info)
            df = pd.DataFrame(columns=('file_name', 'subject', 'medicine', 'lights_off_time'))
            filenames = [ntpath.basename(path).replace('-PSG.edf', '.npz') for path in self.x_paths]
            df['file_name'] = filenames
            df['subject'] = np.tile(info['Subject - age - sex'].to_numpy(), [2, 1]).transpose().ravel()
            for j in info.index:
                if info.iloc[j, 3] == 1:
                    df.iloc[j * 2, 2] = 'Placebo'
                    df.iloc[j * 2, 3] = info.iloc[j, 4]
                    df.iloc[j * 2 + 1, 2] = 'Temazepam'
                    df.iloc[j * 2 + 1, 3] = info.iloc[j, 6]
                else:
                    df.iloc[j * 2, 2] = 'Temazepam'
                    df.iloc[j * 2, 3] = info.iloc[j, 6]
                    df.iloc[j * 2 + 1, 2] = 'Placebo'
                    df.iloc[j * 2 + 1, 3] = info.iloc[j, 4]
            df.to_csv(self.inventory, index=False)
        elif self.dataset == dataset_name[0]:
            df = pd.DataFrame({'file_name': [ntpath.basename(path).replace('.edf', '.npz') for path in self.x_paths]})
            df.to_csv(self.inventory, index=False)

    def read_save(self):
        self.gen_inventory()
        num = len(self.x_paths)
        have_shown = False
        selected_channels = []
        epochs_list = np.zeros((num,), dtype=np.int)
        details_5 = []
        details_2 = []
        for i in range(num):
            have_shown, selected_channels, epochs, stat1, stat2 = self._read_save(have_shown, self.x_paths[i],
                                                                                  self.y_paths[i], selected_channels, i)
            epochs_list[i] = epochs
            details_5.append(stat1)
            details_2.append(stat2)
        df = pd.read_csv(self.inventory)
        df['epochs'] = epochs_list
        df['5_details'] = details_5
        df['2_details'] = details_2
        df.to_csv(self.inventory, index=False)
        self._info_archive()
        '''
        have_shown = False
        selected_channels = []
        for i in range(len(self.x_paths)):
            if self.x_paths[i].find('SC4761E0') > -1 or self.x_paths[i].find('SC4762E0') > -1:
                have_shown, selected_channels, epochs, stat = self._read_save(have_shown, self.x_paths[i],
                                                                              self.y_paths[i], selected_channels, i)
        '''


def data_pre():
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_dir', type=str, default='data',
                       help="This is a dir including processed data from original dataset")
    parse.add_argument('--output_dir', type=str, default='output',
                       help="This is a dir to store trained model")
    parse.add_argument('--dataset', type=int, required=True,
                       help="0 is personal dataset;\n"
                            "1 is the expanded cassette of sleep-EDF in 2018;\n"
                            "2 is the sleep-telemetry of sleep-EDF;\n"
                            "please input the integer")
    paras = parse.parse_args()
    # os.getcwd()
    data_dir = os.sep.join([paras.data_dir, dataset_name[paras.dataset]])
    output_dir = os.sep.join([paras.output_dir, dataset_name[paras.dataset]])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    x_paths = glob.glob(dataset_path[paras.dataset][0])
    y_paths = glob.glob(dataset_path[paras.dataset][1])
    assert len(x_paths) == len(y_paths), 'the number of data files is not matched with the number of label files!'
    x_paths.sort()
    y_paths.sort()
    x_paths = np.asarray(x_paths)
    y_paths = np.asarray(y_paths)

    preparing = DataPre(data_dir, output_dir, dataset_name[paras.dataset], x_paths, y_paths)
    preparing.read_save()
    print('==========================================data has loaded!============================================')


if __name__ == '__main__':
    data_pre()

