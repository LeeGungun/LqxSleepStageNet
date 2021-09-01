# -*- coding: utf-8 -*-
import numpy as np
import utils
import preprocessing as prepare
import os
import shutil
import time
import pandas as pd
from collections import OrderedDict


class RunConfig(object):
    def __init__(self):
        super(RunConfig, self).__init__()
        self.dataset = 1
        self.repeat_id = 0
        self.data_format = 'NCHW'     # not strict format, only to define the order between channel and size
        self.inputs = ('t',)     # must at least one of 't' and 'f', perhaps have 'df'
        self.file_num = 1
        self.batch_size = 60   # seqsleepnet使用的是32
        # number of epochs per sequence
        self.seq_len = 20  # 20, 30, 10 --- 10min, 15min, 5min
        self.seq_batch_size = 1
        # None indicates don't use, the unit of 'batch' mode is one file channel, otherwise is one sample
        self.heap = 10
        #self.batch_files = 1
        self.classes = 5   # W, N1, N2, N3, REM
        self.chs = 3     # from data                9
        self.wins = 59    # from preprocessing         59
        self.window = 100   # from preprocessing       200
        self.fft = 65   # from preprocessing     129
        self.fs = 100  # from data      200
        self.epoch_second = 30   # according to AASM standard
        self.multitask = False

        self.initial_epoch = 0
        #self.epochs_per_fold = 15  # for successive n_folds cross validation
        self.epochs = 50    # seqsleepnet = 10 , 我先提前终止找到合适的超参数，比如：epoch，然后换数据集开始交叉验证 --- 要不要保持训练好的参数？
        # 2个方案 --- （1）再次初始化参数，使用扩大的数据集和提前终止确定的epoch --- drawback: 到底应该保持相同的epoch还是global_iter
        #             （2）保持提前终止得到的参数，使用扩大的数据集继续训练，监控验证集的平均loss，直至其低于提前终止时的loss ---- 我选这个
        self.epochs_per_fold = 20
        self.train_unit = 'sample'    # 'sample' or 'step'
        self.patience = 10    # used for early stopping --- unit: epoch --- usually more than LR's hold
        self.metric_threshold = 0.88
        self.callback_monitor = 'loss'    # ('acc', 'loss')
        self.optimizer = 'nadam'   # ('adagrad', 'adam', 'nadam)
        # in term of several optimizers about different weight groups
        self.opt_clip_flag = ('',)    # ('', 'global_norm')
        self.opt_clip_norm = (0.,)
        self.learning_rate = (0.0001,)    # seqsleepnet = 0.0001    weights分组训练 0.0003
        self.use_initial_warmup = False
        self.ini_warm_lr = (0.0003,)
        self.base_lr = (0.0005,)   # max_lr for warm_up
        self.min_lr = (0.00005,)
        # 最优先是初始warm_up，主要依靠use_initial_warmup控制且只在warm_step内有效，然后就是持续有效的step调节，最后是epoch调节
        self.train_strategy = ('reduce_max', 'step')   #('step', 'reduce_max', 'reduce_min', 'cosine_decay')  multi-classification use 'reduce_min'--- ce_loss
        # 'step'
        self.step_min_lr = 0.0001
        self.lr_decay = 0.85
        self.lr_decay_steps = 1e5   # 1e6
        # 'reduce_min'/'reduce_max'  min or max depends on metric monitored
        # note that the subtract between base_lr(max_lr) and min_lr should not too large
        self.reduce_monitor = 'acc'   # ('acc', 'loss')
        self.train_metric_hold = 1   # epoch  --- 当相等时，表示至少已经有train_metric_hold + 1次epoch了
        self.val_metric_hold = 1
        self.reduce_lr_factor = 0.8
        self.cold_hold = 2    # 若为1则会在本次就调整   cold_hold - 1
        self.warm_rate = 0.1
        # 'cosine_decay' has no special params
        # 'warm_up'
        self.warm_steps = 250   # 800
        #self.warm_hold_steps = 10

        self.opt_groups = ((),)

        # for successive nfolds cross validation
        self.nfolds_lr = ((1e-4,), (1e-4,), (1e-4,), (1e-4,), (1e-4,), (1e-4,), (1e-4,), (1e-4,), (1e-4,), (1e-4,),
                          (1e-4,), (1e-4,), (1e-4,), (1e-4,), (1e-4,), (1e-4,), (1e-4,), (1e-4,), (1e-4,), (1e-4,))
        self.nfolds_tr_strategy = (('reduce_max', 'step'), ('reduce_max', 'step'), ('reduce_max', 'step'),
                                   ('reduce_max', 'step'), ('reduce_max', 'step'), ('reduce_max', 'step'),
                                   ('reduce_max', 'step'), ('reduce_max', 'step'), ('reduce_max', 'step'),
                                   ('reduce_max', 'step'), ('reduce_max', 'step'), ('reduce_max', 'step'),
                                   ('reduce_max', 'step'), ('reduce_max', 'step'), ('reduce_max', 'step'),
                                   ('reduce_max', 'step'), ('reduce_max', 'step'), ('reduce_max', 'step'),
                                   ('reduce_max', 'step'), ('reduce_max', 'step'))
        self.nfolds_ini_warmup = (False, False, False, False, False, False, False, False, False, False,
                                  False, False, False, False, False, False, False, False, False, False)
        self.nfolds_epochs = (50, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                              20, 20, 20, 20, 20, 20, 20, 20, 20, 20)
        self.nfolds_ini_epoch = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


        self.wd = 0.001  # wd
        self.seed = np.random.seed(4869 + self.repeat_id)
        self.dtype = 'float32'
        self.relu_leaky = 0.01    # -1 --- absolute value rectification
        self.relu_threshold = 0.
        self.he_relu_scale = 2. / (1. + self.relu_leaky**2)

        self.dropout_t = 0.5
        self.dropout_f = 0.5
        self.rnn_input_dropout = 0.5
        self.rnn_hidden_dropout = 0.

        # for SelectF layer
        self.f_low_num = 16
        self.f_mid_num = 22
        # high, mid, low
        self.select_f = ((13, 11, 8), (4, 7, 5))  # 收集16个特征：((13, 11, 8), (4, 7, 5)) 收集32/24个特征：((9/30, 20, 15), (4/8, 8/12, 12))
        self.f_wd = self.wd
        self.f_he_scale = 1  # no activation

        # for PatternConv  public: (9_kernel, 5_stride)     personal: (100, 20)
        # inner conv is depth conv
        self.pattern_layers = 2
        self.pattern_filters = ((50, 32, 4), (6, 32, 4))    # (kernel, dfilter_depth, pfilter_depth)  ((18, 16, 2), (6, 16, 4/3))
        self.pattern_stride = (10, 1)
        self.pattern_padding = ('SAME', 'SAME')
        self.pattern_sup = ((0.01, 0.01),) * 2
        self.pattern_conv = ('in_chs',) * 2    # ('in_chs', 'normal', 'separate')
        self.pattern_wd = (self.wd,) * 2
        self.pattern_he_scale = (self.he_relu_scale,) * 2    # activation --- relu / leaky_relu

        self.nn_layers = 1
        self.nn_filters = ((8, 128),)    # (kernel, filter_depth) / (kernel, dfilter_depth, pfilter_depth)
        self.nn_stride = (1,)
        self.nn_padding = ('SAME',)
        self.nn_act = ('relu',)     # (False, 'relu', 'tanh', 'sigmoid')
        self.nn_conv = ('normal',)   # ('in_chs', 'normal', 'separate')
        self.nn_wd = (self.wd,)
        self.nn_he_scale = (self.he_relu_scale,)    # alter with respect to act

        self.max_pool_layers = 2
        self.max_pool_filters = (8, 4)
        self.max_pool_stride = (8, 4)
        self.max_pool_padding = ('SAME',) * 2

        self.avg_pool_layers = 1
        self.avg_pool_filters = (4,)
        self.avg_pool_stride = (4,)
        self.avg_pool_padding = ('SAME',)

        self.cnn_dropout_layers = 1
        self.cnn_dropout_rate = (0.5,)

        # VarianceSE/SVDSE
        self.se_trans_chs = 3
        self.se_he_scale = self.he_relu_scale
        self.se_bias = True     # SVD中相当于数据中心化
        self.se_wd = self.wd

        self.padding_mode = 'batch'  # ('seq', 'batch')   'batch' --- 仅保证序列的batch_size即可，序列长小于等于seq_len

        # 根据特征数更改
        self.rnn_layers = 1                                                      # 2
        self.rnn_h_units = (64,)   # corresponding with num_layers
        self.rnn_h_wd = (self.wd,)
        self.focus_units = 64   # wins_focus, chs_focus   stack-bi --- the first layer, bi-stack --- the top layer
        self.focus_bias = True
        self.focus_wd = self.wd
        self.rnn_mmode = ('sum',)   # corresponding with num_layers
        self.bi_wrapper_mmode = 'sum'

        self.classifier_bias = True
        self.classifier_dr_rate = 0.5

        self.print_steps = 80     # used to indicate how many steps should print training information


class ExperControl(object):
    def __init__(self, dataset_id, id_sets, repeat_id, info_dir, prepro_dir=None, use_store=True, reuse=False):
        super(ExperControl, self).__init__()
        self.dataset_id = dataset_id
        #self.experiment_mode = 'n_folds_cross_validation'  # ('early_stopping', 'n_folds_cross_validation')
        self.id_sets = id_sets
        self.repeat_id = repeat_id    # n折交叉验证的n就和普通的repeats一样，所以该参数也适用于n折交叉验证
        self.repeats = len(id_sets['train'])

        self.dataset_info = utils.get_info(dataset_id, info_dir)

        if use_store:
            if prepro_dir is not None:
                self.data_dir = os.sep.join([prepro_dir, utils.dataset2name[self.dataset_id]])
                self.data_type = 'processed'
            else:
                raise Exception('if use stored preprocessed data, must give \'prepro_dir\'')
        else:
            self.data_dir = self.dataset_info['data_dir']
            self.data_type = 'ori'

        # output_dir = os.sep.join(['MyFiles', 'output']) ?
        self.record_dir = os.sep.join([self.dataset_info['output_dir'], 'archive'])
        if os.path.exists(self.record_dir):
            if not reuse:    # 不继续训练则删除原来的
                shutil.rmtree(self.record_dir)
                os.makedirs(self.record_dir)
        else:
            os.makedirs(self.record_dir)
        self.reuse = reuse
        self.train_stat = []
        self.sets = self.get_sets()    # dict 并完成训练集的统计

    def set_repeat_id(self, repeat):
        self.repeat_id = repeat

    def set_new_sets(self):
        self.sets = self.get_sets()

    def get_sets(self):
        if self.repeat_id < self.repeats:
            info = pd.read_csv(self.dataset_info['inventory'])
            details = (info['5_details'], info['2_details'])

            def _id2name(s):   # 原地修改s集
                for i in range(len(s)):
                    s[i] = os.sep.join([self.data_dir, info.iloc[s[i], 0]])

            def _stat(tr):
                if not self.train_stat:
                    self.train_stat.extend([[], []])
                else:
                    self.train_stat = [[], []]
                for idx in tr:
                    class_details = details[0].loc[idx].strip(r'[|]').split()[:-1]
                    self.train_stat[0].append(np.array(list(map(lambda e: int(e), class_details))))
                    trans_details = details[1].loc[idx].strip(r'[|]').split()
                    self.train_stat[1].append(np.array(list(map(lambda e: int(e), trans_details))))
                self.train_stat[0] = np.sum(np.vstack(self.train_stat[0]), axis=0)
                self.train_stat[1] = np.sum(np.vstack(self.train_stat[1]), axis=0)

            sets = {}
            for k, v in self.id_sets.items():
                if k == 'train':
                    train = v[self.repeat_id].tolist()
                    _stat(train)
                    _id2name(train)
                    sets['train'] = train
                elif k == 'val':
                    val = v[self.repeat_id].tolist()
                    _id2name(val)
                    sets['val'] = val
                elif k == 'test':
                    if len(v) != self.repeats:
                        test = v[0].tolist()
                    else:
                        test = v[self.repeat_id].tolist()
                    _id2name(test)
                    sets['test'] = test

            return sets
        else:
            self.train_stat = []
            return {}

    def get_predict_set(self):
        if self.sets.get('test', None) is not None:
            return self.sets['test']
        else:
            raise Exception('this experiment manager has no test set')

    # phase --- ('training', 'evaluating', 'fully_evaluating', 'predicting', 'cm_result'---训练中用于保留预测结果)
    # logs would better be a dict and per value must be a list, but can be 2D array also
    # 'weights' indicate to save weights and their gradients' L2
    def archive(self, phase, logs, *args, **kwargs):
        if 'attentions' in args:   # 文件形式为NPZ
            directory = os.sep.join([self.record_dir, 'No.%d_repeat' % (self.repeat_id + 1), phase, 'attentions'])
            if not os.path.exists(directory):
                os.makedirs(directory)
            file = os.sep.join([directory, logs['file_name'].replace('.npz', '_attention.npz')])
            np.savez(file, **logs['attentions'])   # {'ch': ..., 'win': ...}
        elif phase == 'cm_result':    # 文件形式为CSV，追加保存
            if 'fully_evaluating' in args:
                directory = os.sep.join([self.record_dir, 'No.%d_repeat' % (self.repeat_id + 1), 'fully_evaluating'])
            elif 'predicting' in args:
                directory = os.sep.join([self.record_dir, 'No.%d_repeat' % (self.repeat_id + 1), 'predicting'])
            else:
                directory = os.sep.join([self.record_dir, 'No.%d_repeat' % (self.repeat_id + 1)])
            if not os.path.exists(directory):
                os.makedirs(directory)
            name = os.sep.join([directory, 'cm_result.csv'])
            if not os.path.exists(name):
                df = pd.DataFrame(logs)
            else:
                df = pd.read_csv(name)
                df = pd.concat([df, pd.DataFrame(logs)], ignore_index=True)
            df.to_csv(name, index=False)
        elif 'info' in args:
            directory = os.sep.join([self.record_dir, 'No.%d_repeat' % (self.repeat_id + 1)])
            if not os.path.exists(directory):
                os.makedirs(directory)
            name = os.sep.join([directory, phase + '_info.npz'])
            np.savez(name, **logs)    # need allow_pickle when loading
        else:      # save to a CSV
            directory = os.sep.join([self.record_dir, 'No.%d_repeat' % (self.repeat_id + 1), phase, *args])
            if not os.path.exists(directory):
                os.makedirs(directory)
            # %a indicates week abbreviations
            #name = time.strftime('%m-%d_%H-%M-%S') + '.csv'   # ':' can't be included in file name
            name = kwargs.get('epoch', time.strftime('%m-%d_%H-%M-%S')) + '.csv'
            if kwargs.get('columns', None) is not None:
                df = pd.DataFrame(logs, columns=kwargs.get('columns'))
            else:
                df = pd.DataFrame(logs)
            #print('archive {}\'s DataFrame handle: {}'.format(args, df))
            df.to_csv(os.sep.join([directory, name]), index=False)


# especially for predicting, id_set为list类型
class PredictControl(object):
    def __init__(self, dataset_id, output_dir, id_set, info_dir, prepro_dir=None, use_store=True):
        super(PredictControl, self).__init__()
        self.dataset_id = dataset_id
        self.record_dir = output_dir   # 保存预测结果的目录
        if os.path.exists(self.record_dir):
            shutil.rmtree(self.record_dir)
        os.makedirs(self.record_dir)

        self.dataset_info = utils.get_info(dataset_id, info_dir)

        if use_store:
            if prepro_dir is not None:
                self.data_dir = os.sep.join([prepro_dir, utils.dataset2name[self.dataset_id]])
                self.data_type = 'processed'
            else:
                raise Exception('if use stored preprocessed data, must give \'prepro_dir\'')
        else:
            self.data_dir = self.dataset_info['data_dir']
            self.data_type = 'ori'

        if not hasattr(id_set, '__iter__'):
            raise Exception('parameter \'id_set\' should be iterative')
        self.set = self.get_set(id_set)    # list

    def get_set(self, id_set):
        info = pd.read_csv(self.dataset_info['inventory'])
        s = []
        for item in id_set:
            s.append(os.sep.join([self.data_dir, info.iloc[item, 0]]))
        return s

    def get_predict_set(self):
        return self.set

    # phase --- ('predicting', 'cm_result')
    # logs would better be a dict and per value must be a list
    # 'weights' indicate to save weights and their gradients' L2
    def archive(self, phase, logs, *args):
        if 'attentions' in args:
            directory = os.sep.join([self.record_dir, 'attentions'])
            if not os.path.exists(directory):
                os.makedirs(directory)
            file = os.sep.join([directory, logs['file_name'].replace('.npz', '_attention.npz')])
            np.savez(file, **logs['attentions'])  # {'ch': ..., 'win': ...}
        elif phase == 'cm_result':
            directory = self.record_dir
            name = os.sep.join([directory, phase + '.csv'])
            if not os.path.exists(name):
                df = pd.DataFrame(logs)
            else:
                df = pd.read_csv(name)
                df = pd.concat([df, pd.DataFrame(logs)], ignore_index=True)
            df.to_csv(name, index=False)
        elif phase == 'top_feature':
            directory = os.sep.join([self.record_dir, phase])
            if not os.path.exists(directory):
                os.makedirs(directory)
            file = os.sep.join([directory, logs['file_name'].replace('.npz', '_features.npy')])
            np.save(file, logs['top_feature'])
        else:  # save to a CSV
            directory = os.sep.join([self.record_dir, *args])
            if not os.path.exists(directory):
                os.makedirs(directory)
            # %a indicates week abbreviations
            name = time.strftime('%m-%d_%H-%M-%S') + '.csv'
            df = pd.DataFrame(logs)
            df.to_csv(os.sep.join([directory, name]), index=False)


# the unit size is about batch_size in scene of 'seq' mode  and seq_len in scene of 'batch' mode
class SeqDataLoader(object):
    def __init__(self, data_type, load_mode, input_types, unit_size=None, file_num=1):
        super(SeqDataLoader, self).__init__()
        if load_mode not in ('seq', 'batch'):
            raise ValueError('is invalid \'load_mode\'')
        self.mode = load_mode    # ('seq', 'batch')
        if hasattr(input_types, '__iter__'):
            if len(input_types) > 3:
                raise Exception('\'input_types\' must be iterative object and its length can not more than 3')
            for i in input_types:
                if i not in ('t', 'f', 'df'):
                    raise ValueError('each element in \'input_types\' only is among \'t\', \'f\' and \'df\'')
        else:
            raise TypeError('\'input_types\' must be iterative object and its length can not more than 3')
        self.signal_types = input_types   # see RunConfig variable 'inputs': ('t', 'f', 'df')
        if 'df' in self.signal_types and 'f' not in self.signal_types:
            raise ValueError('when \'df\' in \'input_types\', \'f\' must be existent')
        if (self.mode == 'seq' and (unit_size is None or file_num != 1)) or (self.mode == 'batch' and
                                                                             (unit_size is None or file_num == 1)):
            raise ValueError('the information used for starting data loader is not enough')
        if self.mode == 'seq':
            self.file_num = 1
            self.unit_size = unit_size
        else:
            self.file_num = file_num
            self.unit_size = unit_size
        if data_type not in ('ori', 'processed'):
            raise ValueError('is invalid \'data_type\'')
        if data_type == 'ori':
            self.get_one_batch_data = self.from_ori_get_one_batch_data
            self.corresponding = None
        else:
            self.get_one_batch_data = self.from_store_get_one_batch_data
            self.corresponding = np.array([-1, -1, -1]).astype('int32')
            for i, it in enumerate(self.signal_types):
                if it == 't':
                    self.corresponding[0] = i
                elif it == 'f':
                    self.corresponding[1] = i
                else:
                    self.corresponding[2] = i
        self.__start = False

    def start(self, data):
        self.ks = [chr(65 + i) + '_house' for i in range(self.file_num)]
        self.warehouse = OrderedDict()
        for k in self.ks:
            self.warehouse[k] = {'id': -1, 'file_path': '', 'stat': (), 'need_epochs': 0, 'start': 0}
        self.__start = True
        self.__batch = 0
        self.data = data
        self.data_info = None
        self.acc_id = 0

    def stop(self):
        self.__start = False

    @property
    def is_exhausted(self):
        if self.__start:
            if self.acc_id < len(self.data):
                return False
            for k in self.ks:
                if self.warehouse[k]['need_epochs'] > 0:
                    return False
            return True
        else:
            raise Exception('This SeqDataLoader doesn\'t start to load data')

    def exhausted_info(self):
        if self.__start:
            idx = []
            for k in self.ks:
                # T or F indicate the latter new data and previous data will be sequential or not
                if self.warehouse[k]['need_epochs'] <= 0:
                    idx.append(False)     # this channel needs new file
                else:
                    idx.append(True)      # this channel has the same file to load
            return idx
        else:
            raise Exception('This SeqDataLoader doesn\'t start to load data')

    # please judge whether exhaust data first
    def from_ori_get_one_batch_data(self):
        if self.__start:
            self.__batch += 1
            batch_size = 0
            no_begin = []
            fetch = tuple([] for _ in range(3))
            fetch_y = []
            fetch_tr_y = []
            mask = np.ones((self.file_num, self.unit_size), dtype='bool')
            for i, k in enumerate(self.ks):
                d = self.warehouse[k]
                if d['need_epochs'] > 0:  # still need to load
                    no_begin.append(False)
                    data = utils.get_part_subject_data(d['file_path'], d['start'], d['start'] + self.unit_size, 'fs')
                    d['start'] += data[0].shape[0]
                    batch_size += data[0].shape[0]
                    fetch_y.append(data[1])
                    fetch_tr_y.append(data[2])
                    data, _ = prepare.preprocessing(data[-1], data[0], self.signal_types, stat=d['stat'])
                    d['need_epochs'] -= self.unit_size
                    if d['need_epochs'] < 0 and self.file_num > 1:  # need padding
                        mask[i][d['need_epochs']:] = np.zeros((-d['need_epochs'],), dtype='bool')
                        for f, da in zip(fetch, data):
                            f.append(np.concatenate((da, np.zeros((-d['need_epochs'],) + da.shape[1:], da.dtype)),
                                                    axis=0) if da is not None else None)
                    else:
                        for f, da in zip(fetch, data):
                            f.append(da if da is not None else None)
                else:  # need switch new file to load
                    no_begin.append(True)
                    if self.acc_id >= len(self.data):  # only perhaps for 'batch' scene
                        for j, info in enumerate(self.data_info[:-1]):
                            fetch[j].append(np.zeros(info[0], info[1]) if info is not None else None)
                        mask[i] = np.zeros((self.unit_size,), dtype='bool')
                        fetch_y.append(np.zeros((0,) + self.data_info[-1][0][1:], self.data_info[-1][1]))
                        fetch_tr_y.append(np.zeros((0,) + self.data_info[-1][0][1:], self.data_info[-1][1]))
                        d['id'] = -1
                        d['file_path'] = ''
                    else:
                        d['id'] = self.acc_id
                        d['file_path'] = self.data[self.acc_id]
                        data = utils.get_subject_data(d['file_path'], 'x', 'y', 'trans_y', 'fs')
                        d['need_epochs'] = data[0].shape[0] - self.unit_size
                        batch_size += self.unit_size if d['need_epochs'] > 0 else data[0].shape[0]
                        d['start'] = self.unit_size if d['need_epochs'] > 0 else data[0].shape[0]
                        fetch_y.append(data[1][: self.unit_size])
                        fetch_tr_y.append(data[2][: self.unit_size])
                        data, stat = prepare.preprocessing(data[-1], data[0], self.signal_types,
                                                           trunc_reserved=self.unit_size)
                        if self.data_info is None:
                            self.data_info = tuple(map(lambda d: ((self.unit_size,) + d.shape[1:], d.dtype)
                                                   if d is not None else None, list(data) + fetch_y))
                        d['stat'] = stat   # have no 'placeholder' as for the order of 't', 'f, ' df'
                        if d['need_epochs'] < 0 and self.file_num > 1:  # need padding
                            mask[i][d['need_epochs']:] = np.zeros((-d['need_epochs'],), dtype='bool')
                            for f, da in zip(fetch, data):
                                f.append(np.concatenate((da, np.zeros((-d['need_epochs'],) + da.shape[1:], da.dtype)),
                                                        axis=0) if da is not None else None)
                        else:
                            for f, da in zip(fetch, data):
                                f.append(da if da is not None else None)
                        self.acc_id += 1
            if self.mode == 'seq':
                return (fetch[0][0], fetch[1][0], fetch[2][0]), fetch_y, fetch_tr_y, None, batch_size, \
                       np.asarray(no_begin, dtype='bool')
            else:
                returned_x = tuple(np.stack(f) if f[0] is not None else None for f in fetch)
                return returned_x, fetch_y, fetch_tr_y, mask, batch_size, np.asarray(no_begin, dtype='bool')
        else:
            raise Exception('This SeqDataLoader doesn\'t start to load data')

    # please judge whether exhaust data first
    def from_store_get_one_batch_data(self):
        if self.__start:
            self.__batch += 1
            batch_size = 0
            no_begin = []
            fetch = tuple([] for _ in range(len(self.signal_types)))
            fetch_y = []
            fetch_tr_y = []
            mask = np.ones((self.file_num, self.unit_size), dtype='bool')
            for i, k in enumerate(self.ks):
                d = self.warehouse[k]
                if d['need_epochs'] > 0:  # still need to load
                    no_begin.append(False)
                    data = utils.get_part_experiment_data(d['file_path'], d['start'], d['start'] + self.unit_size,
                                                          self.signal_types)
                    d['start'] += data[0].shape[0]
                    batch_size += data[0].shape[0]
                    d['need_epochs'] -= self.unit_size
                    fetch_y.append(data[-2])
                    fetch_tr_y.append(data[-1])
                    if d['need_epochs'] < 0 and self.file_num > 1:  # need padding
                        mask[i][d['need_epochs']:] = np.zeros((-d['need_epochs'],), dtype='bool')
                        for f, da in zip(fetch, data[:-2]):
                            f.append(np.concatenate((da, np.zeros((-d['need_epochs'],) + da.shape[1:], da.dtype)),
                                                    axis=0))
                    else:
                        for f, da in zip(fetch, data[:-2]):
                            f.append(da)
                else:  # need switch new file to load
                    no_begin.append(True)
                    if self.acc_id >= len(self.data):  # only perhaps for 'batch' scene
                        for j, info in enumerate(self.data_info[:-1]):
                            fetch[j].append(np.zeros(info[0], info[1]))
                        mask[i] = np.zeros((self.unit_size,), dtype='bool')
                        fetch_y.append(np.zeros((0,) + self.data_info[-1][0][1:], self.data_info[-1][1]))
                        fetch_tr_y.append(np.zeros((0,) + self.data_info[-1][0][1:], self.data_info[-1][1]))
                        d['id'] = -1
                        d['file_path'] = ''
                    else:
                        d['id'] = self.acc_id
                        d['file_path'] = self.data[self.acc_id]
                        data = utils.get_part_experiment_data(d['file_path'], 0, self.unit_size, self.signal_types,
                                                              True)
                        d['need_epochs'] = data[-1] - self.unit_size
                        batch_size += self.unit_size if d['need_epochs'] > 0 else data[0].shape[0]
                        d['start'] = self.unit_size if d['need_epochs'] > 0 else batch_size
                        fetch_y.append(data[-3])
                        fetch_tr_y.append(data[-2])
                        if self.data_info is None:
                            self.data_info = tuple(map(lambda d: ((self.unit_size,) + d.shape[1:], d.dtype), data[:-2]))
                        if d['need_epochs'] < 0 and self.file_num > 1:  # need padding
                            mask[i][d['need_epochs']:] = np.zeros((-d['need_epochs'],), dtype='bool')
                            for f, da in zip(fetch, data[:-3]):
                                f.append(np.concatenate((da, np.zeros((-d['need_epochs'],) + da.shape[1:], da.dtype)),
                                    axis=0))
                        else:
                            for f, da in zip(fetch, data[:-3]):
                                f.append(da)
                        self.acc_id += 1
            if self.mode == 'seq':
                returned_x = tuple(fetch[c][0] if c >= 0 else None for c in self.corresponding)
                # (batch, ...(单帧特征构成)) --- returned_x元素
                return returned_x, fetch_y, fetch_tr_y, None, batch_size, np.asarray(no_begin, dtype='bool')
            else:
                returned_x = tuple(np.stack(fetch[c]) if c >= 0 else None for c in self.corresponding)
                # (files, seq_len, ...(单帧特征构成)) --- returned_x元素
                return returned_x, fetch_y, fetch_tr_y, mask, batch_size, np.asarray(no_begin, dtype='bool')
        else:
            raise Exception('This SeqDataLoader doesn\'t start to load data')





