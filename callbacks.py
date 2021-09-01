# -*- coding: utf-8 -*-
from tensorflow import keras
import numpy as np
import time
import sys
import os
import shutil


# 'monitor' passed in is one of ('ori_loss', 'acc')
class EarlyStoppingCallback(keras.callbacks.Callback):
    def __init__(self, mode='sample', patience=None, monitor='loss'):
        super(EarlyStoppingCallback, self).__init__()     # have self.model, call self.set_model(model) to alter
        self.patience = patience   # 保持记录阈值
        self.mode = mode     # 'sample' or 'step' --- stable unit --- only realize 'sample' scene
        self.stateful_metrics = None
        #self.save_dir = os.sep.join(['MyFiles', 'temp'])
        self.save_dir = 'temp'    # 保留一些显示记录
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(os.sep.join([self.save_dir, 'train']), mode=0o777)
        self.train_hold = {'best_total_loss': np.Inf, 'best_total_loss_wait': 0, 'best_total_loss_weights': [],
                           'best_total_loss_lr': 0., 'best_total_loss_epoch': 0, 'best_ori_loss': np.Inf,
                           'best_ori_loss_wait': 0, 'best_ori_loss_weights': [], 'best_ori_loss_lr': 0.,
                           'best_ori_loss_epoch': 0, 'best_acc': -np.Inf, 'best_acc_weights': [],
                           'best_acc_epoch': 0, 'best_acc_lr': 0., 'best_acc_wait': 0}
        self.val_hold = {'best_total_loss': np.Inf, 'best_total_loss_wait': 0, 'best_total_loss_weights': [],
                         'best_total_loss_lr': 0., 'best_total_loss_epoch': 0, 'best_ori_loss': np.Inf,
                         'best_ori_loss_wait': 0, 'best_ori_loss_weights': [], 'best_ori_loss_lr': 0.,
                         'best_ori_loss_epoch': 0, 'best_acc': -np.Inf, 'best_acc_weights': [], 'best_acc_epoch': 0,
                         'best_acc_lr': 0., 'best_acc_wait': 0}   # 之后可能会有 'end_epoch'
        if monitor == 'loss':
            self.monitor = 'best_ori_loss_wait'
        elif monitor == 'acc':
            self.monitor = 'best_acc_wait'
        else:
            raise ValueError('the \'monitor\' passed in doesn\'t meet the specification')

    def get_lr_monitor(self, monitor):
        if monitor == 'acc':
            return self.val_hold['best_acc_wait']
        elif monitor == 'loss':
            return self.val_hold['best_ori_loss_wait']

    # stateful_metrics --- iterable --- ele: metric_name
    def register_metrics(self, stateful_metrics):     # 在训练开始前，set_model后
        self.stateful_metrics = ['total_loss', 'ori_loss'] + list(stateful_metrics)
        if self.model is None:
            raise Exception('please register model to callback at first')
        self.model.history = [[], []] + list([] for _ in stateful_metrics)
        # because model.metrics_names is read-only attribute
        self.model.metrics_na = self.stateful_metrics

    # used to trainer phase convention
    def __change_phase_reset_history(self):
        if self.model is not None:
            if hasattr(self.model, 'history'):
                for idx in range(len(self.model.history)):
                    self.model.history[idx].clear()
        self.last_batch_ori_loss = 0.
        self.seen = 0

    # after epoch_begin
    # values --- list of tuples (name, value)
    def __print_stateful_metrics(self, values, print_time):
        pre_console_width = self.__console_width
        info = ''
        sys.stdout.write('\n')

        if print_time:
            now = time.time()
            use_time = now - self.begin_time
            info += ' - %.0fs' % use_time
            if self.target is not None:
                num_digits = int(np.log10(self.target)) + 1    # n-digit number
                bar = ('%' + str(num_digits) + 'd/%d [') % (self.seen, self.target)
                prog = float(self.seen) / self.target
                prog_width = int(prog * 30)     # 进度条总长30
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                if self.seen < self.target:
                    bar += '>'
                else:
                    bar += '='
                bar += ('.' * (30 - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % self.seen
            self.__console_width = len(bar)
            sys.stdout.write(bar)
            if self.seen:
                time_per_unit = use_time / self.seen
            else:
                time_per_unit = 0
            if self.target is not None and self.seen < self.target:
                eta = time_per_unit * (self.target - self.seen)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                # when self.target is not None, use expected time, otherwise, use unit_time
                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += ' %.0fs/%s' % (time_per_unit, self.mode)
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/%s' % (time_per_unit * 1e3, self.mode)
                else:
                    info += ' %.0fus/%s' % (time_per_unit * 1e6, self.mode)
        else:
            self.__console_width = 0

        for n, v in values.items():
            info += ' - %s:' % n
            if v.size == 1:
                v = v.item()
                if abs(v) > 1e-3:
                    info += ' %.4f' % v
                else:
                    info += ' %.4e' % v
            else:
                info += ' %s' % v.tolist()

        self.__console_width += len(info)
        if pre_console_width > self.__console_width:
            info += (' ' * (pre_console_width - self.__console_width))

        if self.target is not None and self.seen >= self.target:
            info += '\n'

        sys.stdout.write(info)
        sys.stdout.flush()
        return info

    def on_train_begin(self, logs=None):
        self.model.stop_training = False
        self.print_steps = logs['print_steps']
        self.epochs = logs['epochs']
        self.i_epoch = logs['start_epoch']
        self.begin_time = time.time()
        print('========================================= start training =============================================')

    def on_epoch_begin(self, epoch, logs=None):
        self.__change_phase_reset_history()   # add that clear last batch ori_loss and seen
        self.lr_history = []
        self.epoch = epoch
        # target --- as for train set
        if self.mode == 'sample':
            self.target = logs.get('total_samples', None)
        elif self.mode == 'step':
            self.target = logs.get('total_steps', None)
        print('Epoch {}/{}'.format(epoch + 1, self.epochs))
        # its stateful_metrics is a numeric value, best is a scalar, otherwise, to use mean
        self.__console_width = 0

    def on_train_batch_end(self, batch, logs=None):
        metrics_values = logs['metrics_log'].values()
        for i, v in enumerate(metrics_values):
            self.model.history[i + 2].append(v)
        other_losses = logs['batch_log']['other_losses']
        ori_loss = logs['batch_log']['ori_loss']
        batch_size = logs['batch_log']['batch_size']
        ori_loss = self.last_batch_ori_loss * self.seen + ori_loss * batch_size
        ori_loss = ori_loss / (self.seen + batch_size)
        total_loss = other_losses + ori_loss
        self.last_batch_ori_loss = ori_loss
        self.seen += batch_size
        self.model.history[0].append(total_loss)
        self.model.history[1].append(ori_loss)
        acc = logs['metrics_log']['overall_ACC']
        self.lr_history.append(logs['batch_log']['lr'])
        if self.seen == self.target:     # 该epoch的最后一批
            if total_loss < self.train_hold['best_total_loss']:
                self.train_hold['best_total_loss'] = total_loss
                self.train_hold['best_total_loss_wait'] = 0
                self.train_hold['best_total_loss_weights'] = self.model.get_weights()
                self.train_hold['best_total_loss_lr'] = self.lr_history[-1]     # list/tuple
                self.train_hold['best_total_loss_epoch'] = self.epoch
            else:
                self.train_hold['best_total_loss_wait'] += 1
            if ori_loss < self.train_hold['best_ori_loss']:
                self.train_hold['best_ori_loss'] = ori_loss
                self.train_hold['best_ori_loss_wait'] = 0
                self.train_hold['best_ori_loss_weights'] = self.model.get_weights()
                self.train_hold['best_ori_loss_lr'] = self.lr_history[-1]
                self.train_hold['best_ori_loss_epoch'] = self.epoch
            else:
                self.train_hold['best_ori_loss_wait'] += 1
            if acc > self.train_hold['best_acc']:
                self.train_hold['best_acc'] = acc
                self.train_hold['best_acc_wait'] = 0
                self.train_hold['best_acc_weights'] = self.model.get_weights()
                self.train_hold['best_acc_lr'] = self.lr_history[-1]
                self.train_hold['best_acc_epoch'] = self.epoch
        if self.seen == self.target or batch % self.print_steps == 0:
            handle = 'current total loss is {}, original loss is {}'.format(total_loss, ori_loss)
            file = os.sep.join([self.save_dir, 'train', str(self.epoch) + '.txt'])
            sys.stdout.write('batch: %5d' % batch + '    ' + handle)
            #self.progbar.update(self.seen, None)      # [(key, value),...] --- can be OrderedDict
            app = self.__print_stateful_metrics(logs['metrics_log'], True)
            with open(file, 'a+') as f:
                f.write('%5d' % batch + ' ' + handle + '\n')
                f.write('      current lr is: {}\n'.format(self.lr_history[-1]))
                f.write('      ' + app + '\n')

    def on_test_begin(self, logs=None):  # 在epoch_begin里初始化的有可能需要在这里置位
        self.__change_phase_reset_history()
        self.last_batch_total_loss = 0.
        self.__console_width = 0    # epoch_begin
        self.target = None  # epoch_begin
        self.val_acc = 0

    def on_test_batch_end(self, batch, logs=None):
        metrics_values = logs['metrics_log'].values()
        for i, v in enumerate(metrics_values):
            self.model.history[i + 2].append(v)
        other_losses = logs['batch_log']['other_losses']
        ori_loss = logs['batch_log']['ori_loss']
        batch_size = logs['batch_log']['batch_size']
        ori_loss = self.last_batch_ori_loss * self.seen + ori_loss * batch_size
        ori_loss = ori_loss / (self.seen + batch_size)
        total_loss = other_losses + ori_loss
        self.last_batch_ori_loss = ori_loss
        self.last_batch_total_loss = total_loss
        self.seen += batch_size
        self.model.history[0].append(total_loss)
        self.model.history[1].append(ori_loss)
        self.val_acc = logs['metrics_log']['overall_ACC']

    def on_test_end(self, logs=None):
        if self.last_batch_ori_loss < self.val_hold['best_ori_loss']:
            self.val_hold['best_ori_loss'] = self.last_batch_ori_loss
            self.val_hold['best_ori_loss_epoch'] = self.epoch
            self.val_hold['best_ori_loss_wait'] = 0
            self.val_hold['best_ori_loss_weights'] = self.model.get_weights()
            self.val_hold['best_ori_loss_lr'] = self.lr_history[-1]
        else:
            self.val_hold['best_ori_loss_wait'] += 1
        if self.last_batch_total_loss < self.val_hold['best_total_loss']:
            self.val_hold['best_total_loss'] = self.last_batch_total_loss
            self.val_hold['best_total_loss_epoch_id'] = self.epoch
            self.val_hold['best_total_loss_wait'] = 0
            self.val_hold['best_total_loss_weights'] = self.model.get_weights()
            self.val_hold['best_total_loss_lr'] = self.lr_history[-1]
        else:
            self.val_hold['best_total_loss_wait'] += 1
        if self.val_acc > self.val_hold['best_acc']:
            self.val_hold['best_acc'] = self.val_acc
            self.val_hold['best_acc_epoch_id'] = self.epoch
            self.val_hold['best_acc_wait'] = 0
            self.val_hold['best_acc_weights'] = self.model.get_weights()
            self.val_hold['best_acc_lr'] = self.lr_history[-1]
        else:
            self.val_hold['best_acc_wait'] += 1
        print('\n---------- validation result:')
        handle = 'total loss is {}, original loss is {}'.format(self.last_batch_total_loss, self.last_batch_ori_loss)
        sys.stdout.write(handle)
        app = self.__print_stateful_metrics(logs, False)
        sys.stdout.write('\n')
        file = os.sep.join([self.save_dir, 'test.txt'])
        with open(file, 'a+') as f:
            f.write('%4d' % self.epoch + ' ' + handle + '\n')
            f.write('     ' + app + '\n')

    def on_epoch_end(self, epoch, logs=None):
        if self.patience is not None:
            if self.val_hold[self.monitor] >= self.patience:
                self.model.stop_training = True
                self.val_hold['end_epoch'] = epoch

    def on_train_end(self, logs=None):
        during = time.time() - self.begin_time
        min = during // 60
        sec = int(during - min * 60)
        hour = int(min // 60)
        min = int(min - hour * 60)
        print('training has spent : {} hours, {} minutes, {} seconds'.format(hour, min, sec))
        print('========================================= stop training =============================================')

    def on_fully_test_begin(self):
        self.begin_time = time.time()
        self.__change_phase_reset_history()
        self.target = None
        self.__console_width = 0
        self.acc = 0.
        self.last_batch_total_loss = 0.
        print('--------------------------------------- start full evaluation ----------------------------------------')

    def on_fully_test_batch_end(self, batch, logs=None):
        metrics_values = logs['metrics_log'].values()
        for i, v in enumerate(metrics_values):
            self.model.history[i + 2].append(v)
        other_losses = logs['batch_log']['other_losses']
        ori_loss = logs['batch_log']['ori_loss']
        batch_size = logs['batch_log']['batch_size']
        ori_loss = self.last_batch_ori_loss * self.seen + ori_loss * batch_size
        ori_loss = ori_loss / (self.seen + batch_size)
        total_loss = other_losses + ori_loss
        self.last_batch_ori_loss = ori_loss
        self.last_batch_total_loss = total_loss
        self.seen += batch_size
        self.model.history[0].append(total_loss)
        self.model.history[1].append(ori_loss)
        self.acc = logs['metrics_log']['overall_ACC']
        if batch % self.print_steps == 0:
            loss = [total_loss, ori_loss]
            for i, l in enumerate(loss):
                loss[i] = ('%.4f' % l) if abs(l) > 1e-3 else ('%.4e' % l)
            handle = 'total loss is {}, original loss is {}'.format(loss[0], loss[1])
            sys.stdout.write('batch: %5d' % batch + '    ' + handle)
            file = os.sep.join([self.save_dir, 'full_evaluation.txt'])
            app = self.__print_stateful_metrics(logs['metrics_log'], False)
            with open(file, 'a+') as f:
                f.write('%5d' % batch + ' ' + handle + '\n')
                f.write('      ' + app + '\n')

    def on_fully_test_end(self, logs=None):
        loss = [self.last_batch_total_loss, self.last_batch_ori_loss]
        print('\n---------- on full training set, the final metrics:')
        for i, l in enumerate(loss):
            loss[i] = ('%.4f' % l) if abs(l) > 1e-3 else ('%.4e' % l)
        handle = 'total loss is {}, original loss is {}'.format(loss[0], loss[1])
        sys.stdout.write(handle)
        self.__print_stateful_metrics(logs, False)
        during = time.time() - self.begin_time
        min = during // 60
        sec = int(during - min * 60)
        hour = int(min // 60)
        min = int(min - hour * 60)
        print('inference has spent : {} hours, {} minutes, {} seconds\ntotal sample number is {}'.format(
            hour, min, sec, self.seen))
        print('------------------------------------- stop full evaluation ----------------------------------------')


# for 'n_fold_cross_validation'
# the threshold is in corresponding of monitor
# 'monitor' passed in is one of ('ori_loss', 'acc')
class NFoldsTrainCallback(keras.callbacks.Callback):
    def __init__(self, patience=None, threshold=None, unit='sample', monitor='loss'):
        super(NFoldsTrainCallback, self).__init__()
        self._unit = unit    # in ('sample', 'step') --- only realize 'sample' scene
        self.patience = patience
        self.stateful_metrics = None
        #self.save_dir = os.sep.join(['MyFiles', 'temp'])
        self.save_dir = 'temp'
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(os.sep.join([self.save_dir, 'train']), mode=0o777)
        self.train_hold = {'best_total_loss': np.Inf, 'best_total_loss_wait': 0, 'best_total_loss_weights': [],
                           'best_total_loss_epoch': 0, 'best_ori_loss': np.Inf, 'best_ori_loss_wait': 0,
                           'best_oti_loss_weights': [], 'best_ori_loss_epoch': 0, 'best_acc': -np.Inf,
                           'best_acc_weights': 0, 'best_acc_epoch': 0, 'best_acc_wait': 0}
        self.val_hold = {'best_total_loss': np.Inf, 'best_total_loss_wait': 0, 'best_total_loss_weights': [],
                         'best_total_loss_epoch': 0, 'best_ori_loss': np.Inf, 'best_ori_loss_wait': 0,
                         'best_ori_loss_weights': [], 'best_ori_loss_epoch': 0, 'best_acc': -np.Inf,
                         'best_acc_weights': [], 'best_acc_epoch': 0, 'best_acc_wait': 0}   # 之后可能会有 'end_epoch'
        if monitor == 'loss':
            self.monitor = 'best_ori_loss_wait'
            self.monitor_value = 'best_ori_loss'
            self.sign = 1
            self.threshold = threshold or -np.Inf
        elif monitor == 'acc':
            self.monitor = 'best_acc_wait'
            self.monitor_value = 'best_acc'
            self.sign = -1
            self.threshold = threshold or np.Inf
        else:
            raise ValueError('the \'monitor\' passed in doesn\'t meet the specification')

    # 转换折数之前应该确保hold信息已保留
    def reset_hold_info(self, threshold=None):
        self.train_hold = {'best_total_loss': np.Inf, 'best_total_loss_wait': 0, 'best_total_loss_weights': [],
                           'best_total_loss_epoch': 0, 'best_ori_loss': np.Inf, 'best_ori_loss_wait': 0,
                           'best_oti_loss_weights': [], 'best_ori_loss_epoch': 0, 'best_acc': -np.Inf,
                           'best_acc_weights': 0, 'best_acc_epoch': 0, 'best_acc_wait': 0}
        self.val_hold = {'best_total_loss': np.Inf, 'best_total_loss_wait': 0, 'best_total_loss_weights': [],
                         'best_total_loss_epoch': 0, 'best_ori_loss': np.Inf, 'best_ori_loss_wait': 0,
                         'best_ori_loss_weights': [], 'best_ori_loss_epoch': 0, 'best_acc': -np.Inf,
                         'best_acc_weights': [], 'best_acc_epoch': 0, 'best_acc_wait': 0}    # 之后可能会有 'end_epoch'
        if self.sign == 1:
            self.threshold = threshold or -np.Inf
        else:
            self.threshold = threshold or np.Inf

    def get_lr_monitor(self, monitor):
        if monitor == 'acc':
            return self.val_hold['best_acc_wait']
        elif monitor == 'loss':
            return self.val_hold['best_ori_loss_wait']

    # stateful_metrics --- iterable --- ele: metric_name
    def register_metrics(self, stateful_metrics):  # 在训练开始前，set_model后
        self.stateful_metrics = ['total_loss', 'ori_loss'] + list(stateful_metrics)
        if self.model is None:
            raise Exception('please register model to callback at first')
        self.model.history = [[], []] + list([] for _ in stateful_metrics)
        self.model.metrics_na = self.stateful_metrics

    # also used to trainer phase convention
    def __change_phase_reset_history(self):
        if self.model is not None:
            if hasattr(self.model, 'history'):     # because save history with basis of training epoch
                for idx in range(len(self.model.history)):
                    self.model.history[idx].clear()
        self.seen = 0     # used for progress bar
        self.acc = 0.
        self.total_loss = 0.
        self.ori_loss = 0.

    # after epoch_begin
    # values --- list of tuples (name, value)
    def __print_stateful_metrics(self, values, print_time):
        pre_console_width = self.__console_width
        sys.stdout.write('\n')
        info = ''

        if print_time:
            now = time.time()
            use_time = now - self.begin_time
            info = ' - %.0fs' % use_time
            if self.target is not None:
                num_digits = int(np.log10(self.target)) + 1  # n-digit number
                bar = ('%' + str(num_digits) + 'd/%d [') % (self.seen, self.target)
                prog = float(self.seen) / self.target
                prog_width = int(prog * 30)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                if self.seen < self.target:
                    bar += '>'
                else:
                    bar += '='
                bar += ('.' * (30 - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % self.seen
            self.__console_width = len(bar)
            sys.stdout.write(bar)
            if self.seen:
                time_per_unit = use_time / self.seen
            else:
                time_per_unit = 0
            if self.target is not None and self.seen < self.target:
                # compute estimated time(ET) --- not print total time used so far
                eta = time_per_unit * (self.target - self.seen)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                # when self.target is not None, use expected time, otherwise, use unit_time
                info = ' - ETA: %s' % eta_format
            else:
                # append to info
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += ' %.0fs/%s' % (time_per_unit, self._unit)
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/%s' % (time_per_unit * 1e3, self._unit)
                else:
                    info += ' %.0fus/%s' % (time_per_unit * 1e6, self._unit)
        else:
            self.__console_width = 0

        for n, v in values.items():
            info += ' - %s:' % n
            if v.size == 1:
                v = v.item()
                if abs(v) > 1e-3:
                    info += ' %.4f' % v
                else:
                    info += ' %.4e' % v
            else:
                info += ' %s' % v.tolist()

        self.__console_width += len(info)
        if pre_console_width > self.__console_width:
            info += (' ' * (pre_console_width - self.__console_width))

        if self.target is not None and self.seen >= self.target:
            info += '\n'

        sys.stdout.write(info)
        sys.stdout.flush()
        return info

    def on_train_begin(self, logs=None, start=True):
        self.model.stop_training = False
        self.print_steps = logs['print_steps']
        self.epochs = logs['epochs']
        self.i_epoch = logs['start_epoch']
        self.begin_time = time.time()
        if start:
            print('======================================== start training ===========================================')

    def on_epoch_begin(self, epoch, logs=None):
        self.__change_phase_reset_history()
        self.lr_history = []
        self.epoch = epoch
        # target --- as for train set
        if self._unit == 'sample':
            self.target = logs.get('total_samples', None)
        elif self._unit == 'step':    # no use
            self.target = logs.get('total_steps', None)
        print('Epoch {}/{}'.format(epoch + 1, self.epochs))
        self.__console_width = 0

    def on_fully_test_begin(self):
        self.begin_time = time.time()
        self.__change_phase_reset_history()
        self.target = None
        self.__console_width = 0
        print('--------------------------------------- start full evaluation ----------------------------------------')

    def on_train_batch_end(self, batch, logs=None):
        metrics_values = logs['metrics_log'].values()
        for i, v in enumerate(metrics_values):
            self.model.history[i + 2].append(v)
        other_losses = logs['batch_log']['other_losses']
        ori_loss = logs['batch_log']['ori_loss']
        batch_size = logs['batch_log']['batch_size']
        ori_loss = self.ori_loss * self.seen + ori_loss * batch_size
        ori_loss = ori_loss / (self.seen + batch_size)
        total_loss = other_losses + ori_loss
        self.total_loss = total_loss
        self.ori_loss = ori_loss
        self.seen += batch_size
        self.model.history[0].append(total_loss)
        self.model.history[1].append(ori_loss)
        self.acc = logs['metrics_log']['overall_ACC']
        self.lr_history.append(logs['batch_log']['lr'])   # tuple/list
        if self.seen == self.target:
            if total_loss < self.train_hold['best_total_loss']:
                self.train_hold['best_total_loss'] = total_loss
                self.train_hold['best_total_loss_wait'] = 0
                self.train_hold['best_total_loss_weights'] = self.model.get_weights()
                self.train_hold['best_total_loss_epoch'] = self.epoch
            else:
                self.train_hold['best_total_loss_wait'] += 1
            if ori_loss < self.train_hold['best_ori_loss']:
                self.train_hold['best_ori_loss'] = ori_loss
                self.train_hold['best_ori_loss_wait'] = 0
                self.train_hold['best_ori_loss_weights'] = self.model.get_weights()
                self.train_hold['best_ori_loss_epoch'] = self.epoch
            else:
                self.train_hold['best_ori_loss_wait'] += 1
            if self.acc > self.train_hold['best_acc']:
                self.train_hold['best_acc'] = self.acc
                self.train_hold['best_acc_wait'] = 0
                self.train_hold['best_acc_weights'] = self.model.get_weights()
                self.train_hold['best_acc_epoch'] = self.epoch
            else:
                self.train_hold['best_acc_wait'] += 1
        if self.seen == self.target or batch % self.print_steps == 0:
            loss = [total_loss, ori_loss]
            for i, l in enumerate(loss):
                loss[i] = ('%.4f' % l) if abs(l) > 1e-3 else ('%.4e' % l)
            handle = 'total loss is {}, original loss is {}'.format(loss[0], loss[1])
            file = os.sep.join([self.save_dir, 'train', str(self.epoch) + '.txt'])
            sys.stdout.write('batch: %5d' % batch + '    ' + handle)
            app = self.__print_stateful_metrics(logs['metrics_log'], True)
            with open(file, 'a+') as f:
                f.write('%5d' % batch + ' ' + handle + '\n')
                f.write('      current lr is: {}\n'.format(self.lr_history[-1]))
                f.write('      ' + app + '\n')

    def on_test_begin(self, logs=None):
        self.__change_phase_reset_history()
        self.__console_width = 0
        self.target = None

    def on_test_batch_end(self, batch, logs=None):
        metrics_values = logs['metrics_log'].values()
        for i, v in enumerate(metrics_values):
            self.model.history[i + 2].append(v)
        other_losses = logs['batch_log']['other_losses']
        ori_loss = logs['batch_log']['ori_loss']
        batch_size = logs['batch_log']['batch_size']
        ori_loss = self.ori_loss * self.seen + ori_loss * batch_size
        ori_loss = ori_loss / (self.seen + batch_size)
        total_loss = other_losses + ori_loss
        self.ori_loss = ori_loss
        self.total_loss = total_loss
        self.seen += batch_size
        self.model.history[0].append(total_loss)
        self.model.history[1].append(ori_loss)
        self.acc = logs['metrics_log']['overall_ACC']

    def on_test_end(self, logs=None):
        if self.ori_loss < self.val_hold['best_ori_loss']:
            self.val_hold['best_ori_loss'] = self.ori_loss
            self.val_hold['best_ori_loss_epoch'] = self.epoch
            self.val_hold['best_ori_loss_wait'] = 0
            self.val_hold['best_ori_loss_weights'] = self.model.get_weights()
        else:
            self.val_hold['best_ori_loss_wait'] += 1
        if self.total_loss < self.val_hold['best_total_loss']:
            self.val_hold['best_total_loss'] = self.total_loss
            self.val_hold['best_total_loss_epoch'] = self.epoch
            self.val_hold['best_total_loss_wait'] = 0
            self.val_hold['best_total_loss_weights'] = self.model.get_weights()
        else:
            self.val_hold['best_total_loss_wait'] += 1
        if self.acc > self.val_hold['best_acc']:
            self.val_hold['best_acc'] = self.acc
            self.val_hold['best_acc_epoch'] = self.epoch
            self.val_hold['best_acc_wait'] = 0
            self.val_hold['best_acc_weights'] = self.model.get_weights()
        else:
            self.val_hold['best_acc_wait'] += 1
        print('\n---------- validation result:')
        loss = [self.total_loss, self.ori_loss]
        for i, l in enumerate(loss):
            loss[i] = ('%.4f' % l) if abs(l) > 1e-3 else ('%.4e' % l)
        handle = 'total sample number is {}, total loss is {}, original loss is {}'.format(self.seen, loss[0], loss[1])
        sys.stdout.write(handle)
        app = self.__print_stateful_metrics(logs, True)
        sys.stdout.write('\n')
        file = os.sep.join([self.save_dir, 'test.txt'])
        with open(file, 'a+') as f:
            f.write('%4d' % self.epoch + ' ' + handle + '\n')
            f.write('     ' + app)

    def on_epoch_end(self, epoch, logs=None):
        if self.threshold is not None:
            if (self.val_hold[self.monitor_value] - self.threshold) * self.sign < 0:  # self.val_hold[self.monitor] >= self.patience
                self.model.stop_training = True
                self.val_hold['end_epoch'] = epoch
        elif self.patience is not None:
            if self.val_hold[self.monitor] >= self.patience:
                self.model.stop_training = True
                self.val_hold['end_epoch'] = epoch

    def on_train_end(self, logs=None, stop=True):
        during = time.time() - self.begin_time
        min = during // 60
        sec = int(during - min * 60)
        hour = int(min // 60)
        min = int(min - hour * 60)
        print('training has spent : {} hours, {} minutes, {} seconds'.format(hour, min, sec))
        if stop:
            print('======================================== stop training ===========================================')

    def on_fully_test_batch_end(self, batch, logs=None):
        metrics_values = logs['metrics_log'].values()
        for i, v in enumerate(metrics_values):
            self.model.history[i + 2].append(v)
        other_losses = logs['batch_log']['other_losses']
        ori_loss = logs['batch_log']['ori_loss']
        batch_size = logs['batch_log']['batch_size']
        ori_loss = self.ori_loss * self.seen + ori_loss * batch_size
        ori_loss = ori_loss / (self.seen + batch_size)
        total_loss = other_losses + ori_loss
        self.ori_loss = ori_loss
        self.total_loss = total_loss
        self.seen += batch_size
        self.model.history[0].append(total_loss)
        self.model.history[1].append(ori_loss)
        self.acc = logs['metrics_log']['overall_ACC']
        if batch % self.print_steps == 0:
            loss = [self.total_loss, self.ori_loss]
            for i, l in enumerate(loss):
                loss[i] = ('%.4f' % l) if abs(l) > 1e-3 else ('%.4e' % l)
            handle = 'total loss is {}, original loss is {}'.format(loss[0], loss[1])
            sys.stdout.write('batch: %5d' % batch + '    ' + handle)
            file = os.sep.join([self.save_dir, 'full_evaluation.txt'])
            app = self.__print_stateful_metrics(logs['metrics_log'], False)
            with open(file, 'a+') as f:
                f.write('%5d' % batch + ' ' + handle + '\n')
                f.write('      ' + app + '\n')

    def on_fully_test_end(self, logs=None):
        loss = [self.total_loss, self.ori_loss]
        print('\n---------- on full training set, the final metrics:')
        for i, l in enumerate(loss):
            loss[i] = ('%.4f' % l) if abs(l) > 1e-3 else ('%.4e' % l)
        handle = 'total loss is {}, original loss is {}'.format(loss[0], loss[1])
        sys.stdout.write(handle)
        self.__print_stateful_metrics(logs, False)
        during = time.time() - self.begin_time
        min = during // 60
        sec = int(during - min * 60)
        hour = int(min // 60)
        min = int(min - hour * 60)
        print('inference has spent : {} hours, {} minutes, {} seconds\ntotal sample number is {}'.format(
            hour, min, sec, self.seen))
        print('------------------------------------- stop full evaluation ----------------------------------------')


class PredictCallback(keras.callbacks.Callback):
    def __init__(self, mode='sample'):
        super(PredictCallback, self).__init__()
        self.mode = mode  # in ('sample', 'step') --- only realize 'sample' scene
        self.stateful_metrics = None

    # stateful_metrics --- iterable --- ele: metric_name
    def register_metrics(self, stateful_metrics):
        self.stateful_metrics = list(stateful_metrics)
        if self.model is None:
            raise Exception('please register model to callback at first')
        self.model.history = list([] for _ in stateful_metrics)
        self.model.metrics_na = self.stateful_metrics

    # values --- list of tuples (name, value)
    def __print_stateful_metrics(self, values, print_time):
        pre_console_width = self.__console_width
        info = ''
        sys.stdout.write('\n')

        if print_time:
            now = time.time()
            use_time = now - self.begin_time
            info += ' - %.0fs' % use_time
            if self.target is not None:
                num_digits = int(np.log10(self.target)) + 1  # n-digit number
                bar = ('%' + str(num_digits) + 'd/%d [') % (self.seen, self.target)
                prog = float(self.seen) / self.target
                prog_width = int(prog * 30)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                if self.seen < self.target:
                    bar += '>'
                else:
                    bar += '='
                bar += ('.' * (30 - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % self.seen
            self.__console_width = len(bar)
            sys.stdout.write(bar)
            if self.seen:
                time_per_unit = use_time / self.seen
            else:
                time_per_unit = 0
            if self.target is not None and self.seen < self.target:
                eta = time_per_unit * (self.target - self.seen)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                # when self.target is not None, use expected time, otherwise, use unit_time
                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += ' %.0fs/%s' % (time_per_unit, self.mode)
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/%s' % (time_per_unit * 1e3, self.mode)
                else:
                    info += ' %.0fus/%s' % (time_per_unit * 1e6, self.mode)
        else:
            self.__console_width = 0

        for n, v in values.items():
            info += ' - %s:' % n
            if v.size == 1:
                v = v.item()
                if abs(v) > 1e-3:
                    info += ' %.4f' % v
                else:
                    info += ' %.4e' % v
            else:
                info += ' %s' % v.tolist()

        self.__console_width += len(info)
        if pre_console_width > self.__console_width:
            info += (' ' * (pre_console_width - self.__console_width))

        if self.target is not None and self.seen >= self.target:
            info += '\n'

        sys.stdout.write(info)
        sys.stdout.flush()

    def on_predict_begin(self, logs=None):
        self.print_steps = logs['print_steps']
        self.begin_time = time.time()
        self.seen = 0
        self.target = None
        self.__console_width = 0
        print('========================================= start inference =============================================')

    def on_predict_batch_end(self, batch, logs=None):
        metrics_values = logs['metrics_log'].values()
        for i, v in enumerate(metrics_values):
            self.model.history[i].append(v)
        self.seen += logs['batch_size']
        if batch % self.print_steps == 0:
            #self.progbar.update(self.seen, None)  # [(key, value),...] --- can be OrderedDict
            self.__print_stateful_metrics(logs['metrics_log'], True)

    def on_predict_end(self, logs=None):
        print('\n---------- predict final metrics:')
        self.__print_stateful_metrics(logs, True)
        during = time.time() - self.begin_time
        min = during // 60
        sec = int(during - min * 60)
        hour = int(min // 60)
        min = int(min - hour * 60)
        print('\npredicting has spent : {} hours, {} minutes, {} seconds\ntotal sample number is {}'.format(
            hour, min, sec, self.seen))
        print('======================================= stop inference =========================================')
