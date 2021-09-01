# -*- coding: utf-8 -*-
from tensorflow import keras
import numpy as np
import utils
from config import SeqDataLoader
from metrics import *
from collections import OrderedDict
import ntpath
import tensorflow as tf
#import shutil
import os
# tf has imported in metrics.py


class FocalLoss(object):
    def __init__(self, index=2):
        super(FocalLoss, self).__init__()
        self.index = index

    # 模型产生的y_pre应该是(batch, class)的
    def compute(self, y_true, y_pre, sample_weight=None):
        if not tf.is_tensor(y_true):
            y_true = tf.constant(y_true)
        if y_pre.shape.ndims > y_true.shape.ndims:
            y_true = tf.one_hot(y_true, y_pre.shape[-1])
        y_true = tf.cast(y_true, y_pre.dtype)
        new_ = tf.multiply(tf.pow(tf.cast(1, y_pre.dtype) - y_pre, self.index), -tf.math.log(y_pre))
        sample_loss = tf.reduce_sum(tf.multiply(y_true, new_), axis=-1)
        if sample_weight is not None:
            if not tf.is_tensor(sample_weight):
                sample_weight = tf.constant(sample_weight, dtype=y_pre.dtype)
            else:
                sample_weight = tf.cast(sample_weight, y_pre.dtype)
            loss = tf.reduce_mean(tf.multiply(sample_weight, sample_loss))
        else:
            loss = tf.reduce_mean(sample_loss)
        return loss


class Trainer(object):
    def __init__(self, config, manager, callback, data_load_mode):
        super(Trainer, self).__init__()
        self.config = config
        self.manager = manager    # 归档结果信息，持有转换成路径名的数据集
        self.optimizers = []
        self.losses = []
        self.train_metrics = None
        self.val_metrics = None
        self.class_weights = {'n_class': 0, 'trans': 0}   # only used in 'training'
        self.callback = callback     # 控制终端显示信息
        self.compiled = False
        self.experiment_set = None    # only file_names
        self.__phase = None      # ('training', 'evaluating'，'fully_evaluating', 'result')
        if data_load_mode not in ('seq', 'batch'):
            raise Exception('unqualified parameter \'data_loader_mode\'')
        self.data_load_mode = data_load_mode      # old - ('dynamic', 'normal')   new - ('seq', 'batch')
        self.data_input_type = config.inputs
        self.unit_size = config.batch_size if data_load_mode == 'seq' else config.seq_len
        self.file_num = config.file_num
        self.model = None
        self.trainable_weights_names = None
        self.trainable_weights_l2 = None
        self.dtype = config.dtype
        self.global_iters = 0

    def set_phase(self, phase):
        self.__phase = phase

    @property
    def phase(self):
        return self.__phase

    def compile(self):
        # optimizers
        if self.config.optimizer == 'adagrad':
            for i, lr in enumerate(self.config.learning_rate):
                self.optimizers.append(keras.optimizers.Adagrad(learning_rate=lr, decay=0., epison=1e-8,
                                                                name='Adagrad{}'.format(i + 1)))
        elif self.config.optimizer == 'adam':
            for i, lr in enumerate(self.config.learning_rate):
                self.optimizers.append(keras.optimizers.Adam(learning_rate=lr, decay=0., beta_1=0.9, beta_2=0.999,
                                                             epsilon=1e-8, amsgrad=True, name='Adam{}'.format(i + 1)))
        elif self.config.optimizer == 'nadam':
            for i, lr in enumerate(self.config.learning_rate):
                self.optimizers.append(keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-7,
                                                              name='Nadam{}'.format(i + 1)))
        # losses， used in training and evaluating
        # keras.losses.SparseCategoricalCrossentropy()   call(y_true, y_predict, sample_weight=None)
        self.losses.append(FocalLoss())      # .compute(y_true, y_predict, sample_weight=None)
        # metrics --- hold states --- can't use in common
        # for 5-classification (就算使用了2分类也是为5分类服务的)
        self.train_metrics = [keras.metrics.SparseCategoricalAccuracy(name='train_overall_ACC', dtype=self.dtype),
                              MacroF1('train_macro_f1', self.dtype),
                              F1('train_multi_f1', n_class=self.config.classes, dtype=self.dtype),
                              Kappa('train_kappa', self.dtype),
                              Rate('train_overall_recall', 'recall', mode='overall', dtype=self.dtype),
                              Rate('train_multi_recall', 'recall', n_class=self.config.classes, dtype=self.dtype),
                              Rate('train_overall_ppv', 'precision', mode='overall', dtype=self.dtype),
                              Rate('train_multi_ppv', 'precision', n_class=self.config.classes, dtype=self.dtype)]
        # for 5-classification
        self.val_metrics = [keras.metrics.SparseCategoricalAccuracy(name='val_overall_ACC', dtype=self.dtype),
                            MacroF1('val_macro_f1', self.dtype),
                            F1('val_multi_f1', n_class=self.config.classes, dtype=self.dtype),
                            Kappa('val_kappa', self.dtype),
                            Rate('val_overall_recall', 'recall', mode='overall', dtype=self.dtype),
                            Rate('val_multi_recall', 'recall', n_class=self.config.classes, dtype=self.dtype),
                            Rate('val_overall_ppv', 'precision', mode='overall', dtype=self.dtype),
                            Rate('val_multi_ppv', 'precision', n_class=self.config.classes, dtype=self.dtype)]
        self.stateful_metrics = ['overall_ACC', 'macro_f1', 'multi_f1', 'kappa', 'overall_recall',
                                 'multi_recall', 'overall_ppv', 'multi_ppv']
        self.val_cm = utils.ConfusionMatrix(utils.LABELS)
        self.train_cm = utils.ConfusionMatrix(utils.LABELS)
        self.compiled = True

    def _register_model(self, model):
        self.model = model
        self.callback.set_model(model)

    def reset_metrics(self, which):
        select = getattr(self, '{}_metrics'.format(which))
        for metric in select:
            metric.reset_states()

    # after compilation  参数1优先级高于参数2
    def _set_LR(self):
        self.global_iters = 0  # 表示重新开始训练
        if 'step' in self.config.train_strategy:     # 'step'可以和 'reduce'混用
            self.lr_fun = self.stepLR
            self.lr_epoch_control = False
        elif 'cosine_decay' in self.config.train_strategy:
            self.ini_accu = 0
            self.lr_fun = self.cosine_decay
        else:
            self.lr_fun = self.warm_up
            self.lr_epoch_control = False      # 没有特别的step调节机制
        if 'reduce_min' in self.config.train_strategy:
            self.reduce_monitor = self.config.reduce_monitor
            self.cold_counter = 0
            self.warm_start = False
            self.rate = 0
            self.key = 'best_ori_loss_wait'  # the corresponding of callback
            self.epoch_regulate_lr = self.reduceLR_warmup
        elif 'reduce_max' in self.config.train_strategy:
            self.reduce_monitor = self.config.reduce_monitor
            self.cold_counter = 0
            self.warm_start = False
            self.rate = 0
            self.key = 'best_acc_wait'  # the corresponding of callback
            self.epoch_regulate_lr = self.reduceLR_warmup
        else:
            self.epoch_regulate_lr = self.no_do

    def no_do(self, *args):
        pass

    def stepLR(self, base_lr, warm_lr, min_lr, opt):
        if self.config.use_initial_warmup and self.global_iters <= self.config.warm_steps:
            slope = (base_lr - warm_lr) / self.config.warm_steps
            new_lr = slope * self.global_iters + warm_lr  # 产生下一步所用lr，此时self.global_iters应该已经+1
            opt.learning_rate.assign(new_lr)
        elif not self.lr_epoch_control:
            if opt.learning_rate > self.config.step_min_lr:
                new_lr = opt.learning_rate * tf.pow(
                    self.config.lr_decay,
                    (self.global_iters - self.config.warm_steps * self.config.use_initial_warmup) /
                    self.config.lr_decay_steps)
                if new_lr > self.config.step_min_lr:
                    opt.learning_rate.assign(new_lr)
                else:
                    opt.learning_rate.assign(self.config.step_min_lr)
                    self.lr_epoch_control = True

    def warm_up(self, base_lr, warm_lr, min_lr, opt):
        if self.config.use_initial_warmup and self.global_iters <= self.config.warm_steps:
            slope = (base_lr - warm_lr) / self.config.warm_steps
            new_lr = slope * self.global_iters + warm_lr
            opt.learning_rate.assign(new_lr)
        else:
            self.lr_epoch_control = True

    #  the regulation unit is epoch not step
    def reduceLR_warmup(self, base_lr, min_lr, opt):
        if self.lr_epoch_control:    # 与warm_up相配合，在步调节中不发挥作用，步调节通常应用于开始
            monitor = self.callback.get_lr_monitor(self.reduce_monitor)
            if monitor >= self.config.val_metric_hold:
                if opt.learning_rate > min_lr and not self.warm_start:  # 防止在回暖中下降，以保证回暖完成
                    new_lr = opt.learning_rate * self.config.reduce_lr_factor
                    if new_lr > min_lr:
                        opt.learning_rate.assign(new_lr)
                    else:
                        opt.learning_rate.assign(min_lr)
                elif not self.warm_start:  # lr已经是最低但性能不好，可能陷入局部最小值点，需要跳出来，因此要增加lr
                    self.cold_counter += 1
                    if self.cold_counter >= self.config.cold_hold:
                        self.warm_start = True
                        self.rate = (base_lr - min_lr) * self.config.warm_rate
                        new_lr = self.rate + opt.learning_rate
                        opt.learning_rate.assign(new_lr)
                else:
                    # warm up
                    new_lr = self.rate + opt.learning_rate
                    if new_lr >= base_lr:
                        self.warm_start = False
                        self.cold_counter = 0
                        self.rate = 0
                        new_lr = base_lr
                    opt.learning_rate.assign(new_lr)
            else:
                if self.warm_start:
                    opt.learning_rate.assign(min_lr)
                # lr remain unchanged
                self.warm_start = False
                self.cold_counter = 0
                self.rate = 0

    # kwargs key: 'accu': on basis of sample or step; 'total': the corresponding of 'accu'
    def cosine_decay(self, base_lr, warm_lr, min_lr, opt):
        # self.epoch_id 是指当前训练真实进行的，不论历史积累（之前的训练）
        accu = self.epoch_id * self.total_samples + self.callback.seen + self.current_ba_frames
        if self.config.use_initial_warmup and self.global_iters <= self.config.warm_steps:
            slope = (base_lr - warm_lr) / self.config.warm_steps
            new_lr = slope * self.global_iters + warm_lr
            opt.learning_rate.assign(new_lr)
            self.ini_accu = accu
        else:
            new_lr = min_lr + 0.5 * (base_lr - min_lr) * \
                     (1. + np.cos((accu - self.ini_accu) / (self.total_samples * self.config.epochs) * np.pi))
            opt.learning_rate.assign(new_lr)

    # 在调用apply_gradients之前调用，在得到变量之后调用
    def _load_optimizer_weights(self, opt, var_list, weights):
        if self.global_iters == 0 and self.manager.reuse and weights:
            with tf.name_scope(opt._name):
                with tf.init_scope():   # tf2.0 没有将下面操作封装成_create_all_weights(var_list)
                    _ = opt.iterations
                    opt._create_hypers()
                    opt._create_slots(var_list)
            opt.set_weights(list(weights))

    @staticmethod
    def compute_class_weight(stat):
        stat = np.asarray(stat)
        stat = np.sum(stat) / stat
        stat = len(stat) * stat / np.sum(stat)
        return stat

    def _fully_test(self, collect):
        pass

    def _train_on_epoch(self, epoch_id, repeat):
        pass

    def reset_cm(self, archive, *args, arch_phase='cm_result', arch_obj='all'):
        pass

    def train(self, model, multitask=False, opt_weights=None):    # 该训练属于self.manager.repeat_id之内
        from random import shuffle
        if not self.compiled:  # optimizer, losses, train/val metrics, confusion matrix
            self.compile()

        self.total_samples = sum(self.manager.train_stat[0])  # sum() needs iterable object
        self.experiment_set = (self.manager.sets['train'].copy(), self.manager.sets['val'])
        if self.class_weights:
            self.class_weights['n_class'] = tf.convert_to_tensor(self.compute_class_weight(self.manager.train_stat[0]),
                                                                 self.dtype)
            self.class_weights['trans'] = tf.convert_to_tensor(self.compute_class_weight(self.manager.train_stat[1]),
                                                               self.dtype)
        self.train_cm.record_ini(multitask)
        self.val_cm.record_ini(multitask)

        self._register_model(model)
        self.callback.register_metrics(self.stateful_metrics)  # must after register model and before train/test/predict
        self.callback.on_train_begin({'print_steps': self.config.print_steps, 'epochs': self.config.epochs,
                                      'start_epoch': self.config.initial_epoch})
        self._set_LR()
        self.opt_weights = opt_weights if opt_weights else list([] for _ in range(len(self.optimizers)))   # 无opt_weights: [] or None
        for e in range(self.config.initial_epoch, self.config.epochs):
            self.epoch_id = e - self.config.initial_epoch
            if self.model.stop_training:
                break
            else:
                self._train_on_epoch(e, self.manager.repeat_id)
                if e == self.config.initial_epoch:
                    self.model.summary()
                shuffle(self.experiment_set[0])
        self.callback.on_train_end()

    def train_nfolds(self, model, multitask=False, opt_weights=None):
        from random import shuffle
        if not self.compiled:    # optimizer, losses, train/val metrics, confusion matrix
            self.compile()

        self.total_samples = sum(self.manager.train_stat[0])      # sum() needs iterable object
        self.experiment_set = (self.manager.sets['train'].copy(), self.manager.sets['val'])
        self.class_weights['n_class'] = tf.convert_to_tensor(self.compute_class_weight(self.manager.train_stat[0]),
                                                             self.dtype)
        self.class_weights['trans'] = tf.convert_to_tensor(self.compute_class_weight(self.manager.train_stat[1]),
                                                           self.dtype)
        self.train_cm.record_ini(multitask)
        self.val_cm.record_ini(multitask)

        self._register_model(model)
        self.callback.register_metrics(self.stateful_metrics)      # must after register model and before train/test/predict
        self.callback.on_train_begin({'print_steps': self.config.print_steps, 'epochs': self.config.epochs,
                                      'start_epoch': self.config.initial_epoch})
        self._set_LR()
        self.opt_weights = opt_weights if opt_weights else list([] for _ in range(len(self.optimizers)))
        for e in range(self.config.initial_epoch, self.config.epochs):
            self.epoch_id = e - self.config.initial_epoch
            if self.model.stop_training:
                break
            else:
                self._train_on_epoch(e, self.manager.repeat_id)
                if e == self.config.initial_epoch:
                    self.model.summary()
                shuffle(self.experiment_set[0])
        self.callback.on_train_end()
        collect = self.experiment_set[0] + self.experiment_set[1]
        self._fully_test(collect)

    def nfolds_reset(self, fold):
        self.config.learning_rate = self.config.nfolds_lr[fold]
        self.config.train_strategy = self.config.nfolds_tr_strategy[fold]
        self.config.use_initial_warmup = self.config.nfolds_ini_warmup[fold]
        self.config.epochs = self.config.nfolds_epochs[fold]
        self.config.initial_epoch = self.config.nfolds_ini_epoch[fold]

    def _reset_opt(self):
        for opt in self.optimizers:
            get_ws = opt.get_weights()
            renew_ws = []
            for w in get_ws:
                renew_ws.append(np.zeros_like(w, dtype=w.dtype))
            opt.set_weights(renew_ws)

    # !!! only for NFoldsTrainCallback !!!
    def train_successive_nfolds(self, fold, model, multitask=False, opt_weights=None, use_threshold=False,
                                threshold=None, fully_test_end=True):
        from random import shuffle
        self.nfolds_reset(fold)
        if not self.compiled:  # optimizer, losses, train/val metrics, confusion matrix
            self.compile()
            self.train_cm.record_ini(multitask)
            self.val_cm.record_ini(multitask)
        else:
            self._reset_opt()
            self.reset_metrics('train')
            self.reset_metrics('val')
            self.reset_cm(False)

        self.total_samples = sum(self.manager.train_stat[0])  # sum() needs iterable object
        self.experiment_set = (self.manager.sets['train'].copy(), self.manager.sets['val'])
        self.class_weights['n_class'] = tf.convert_to_tensor(self.compute_class_weight(self.manager.train_stat[0]),
                                                             self.dtype)
        self.class_weights['trans'] = tf.convert_to_tensor(self.compute_class_weight(self.manager.train_stat[1]),
                                                           self.dtype)
        self._register_model(model)
        self.callback.register_metrics(self.stateful_metrics)  # must after register model and before train/test/predict
        self.callback.on_train_begin({'print_steps': self.config.print_steps, 'epochs': self.config.epochs,
                                      'start_epoch': self.config.initial_epoch}, False)
        self._set_LR()
        self.opt_weights = opt_weights if opt_weights else list([] for _ in range(len(self.optimizers)))
        for e in range(self.config.initial_epoch, self.config.epochs):
            self.epoch_id = e - self.config.initial_epoch
            if self.model.stop_training:
                break
            else:
                self._train_on_epoch(e, fold)
                if e == self.config.initial_epoch:
                    self.model.summary()
                shuffle(self.experiment_set[0])
        self.callback.on_train_end()
        if fully_test_end:
            collect = self.experiment_set[0] + self.experiment_set[1]
            self._fully_test(collect)
        _threshold = self.callback.val_hold[self.callback.monitor_value]
        self.callback.on_train_end(stop=False)
        if use_threshold and threshold is None:
            threshold = _threshold
        self.callback.reset_hold_info(threshold)


class CustomTrainer(Trainer):
    def __init__(self, config, manager, callback, data_load_mode):
        super(CustomTrainer, self).__init__(config, manager, callback, data_load_mode)

    # inp must be tuple/list, gt and sample_weights must be tuple/list
    def _train_step(self, inp, gt, tr_gt, mask, ini, last_o, no_begin, sw, logs):
        with tf.GradientTape() as tape:
            # chs_attention and wins_attention use to visualize, thus, valid in predicting/evaluating
            if mask is not None:  # 第二种上数据上数据方式的模型 --- 'batch' 该模型还未修改、完善
                predict, next_ini, next_o, _, _ = self.model(inp, mask, ini, last_o, no_begin, True, True)
            else:
                predict, next_ini, next_o, _, _ = self.model(inp, ini, last_o, no_begin, True, True)
            predict = list(tf.clip_by_value(p, 1e-8, 1.) for p in predict)
            ce_loss = self.losses[0].compute(tf.concat(gt, axis=0), tf.concat(predict, axis=0),
                                             sample_weight=tf.concat(sw, axis=0))  # average
            other_losses = self.model.regu()  # regularizer      _callable_losses
            total_loss = [ce_loss] + other_losses
        logs['other_losses'] = sum(other_losses).numpy()
        # print('other loss: {}'.format(logs['other_losses']))
        logs['ori_loss'] = ce_loss.numpy()
        w = self.model.trainable_weights  # non-nested list
        _w = tape.watched_variables()
        if self.trainable_weights_names is None:
            self.trainable_weights_names = []
            self.trainable_weights_l2 = []  # used for experiment
            for w_i in w:
                self.trainable_weights_names.append(w_i.name)
                self.trainable_weights_l2.append([tf.clip_by_value(tf.linalg.norm(w_i, ord=2), 1e-8, np.Inf).numpy()])
        else:
            for i, w_i in enumerate(w):
                self.trainable_weights_l2[i].append(tf.clip_by_value(tf.linalg.norm(w_i, ord=2), 1e-8, np.Inf).numpy())
        group_watched_w = utils.group_by_optimizers(_w, self.config.opt_groups)  # only one group temporarily
        g = tape.gradient(total_loss, group_watched_w)
        # for idx, f in enumerate(self.config.opt_clip_flag):
        #    if f:
        #        g[idx] = tf.clip_by_global_norm(g[idx], self.config.opt_clip_norm[idx])[0]
        logs['lr'] = []
        for idx, opt in enumerate(self.optimizers):
            self.lr_fun(self.config.base_lr[idx], self.config.ini_warm_lr[idx], self.config.min_lr[idx], opt)
            logs['lr'].append(opt.learning_rate.numpy())
            self._load_optimizer_weights(opt, group_watched_w[idx], self.opt_weights[idx])
            opt.apply_gradients(list(zip(g[idx], group_watched_w[idx])))
            self.opt_weights[idx] = opt.get_weights()
        # update confusion matrix
        pre = tuple(map(lambda e: tf.cast(tf.argmax(e, axis=-1), 'int32').numpy(), predict))
        gt_ = tuple(e.numpy() for e in gt)
        tr_gt_ = tuple(e.numpy() for e in tr_gt)
        self.train_cm.record(gt_, pre, tr_gt_)
        # update train metrics
        self.train_metrics[0].update_state(tf.concat(gt, axis=0), tf.concat(predict, axis=0))
        cm = tf.convert_to_tensor(self.train_cm.matrix_n)
        for metric in self.train_metrics[1:]:
            metric.update_state(cm)
        # tackle next_init, next_o
        return next_ini, next_o, logs

    def _test_step(self, inp, gt, tr_gt, mask, ini, last_o, no_begin, logs):
        # chs_attention and wins_attention use to visualize, thus, valid in predicting/evaluating
        if mask is not None:  # 第二种上数据上数据方式的模型 --- 'batch' 该模型还未修改、完善
            predict, next_ini, next_o, ch_focus, win_focus = \
                self.model(inp, mask, ini, last_o, no_begin, False, True)
        else:
            predict, next_ini, next_o, ch_focus, win_focus = self.model(inp, ini, last_o, no_begin, False, True)
        predict = list(tf.clip_by_value(p, 1e-8, 1.) for p in predict)
        ce_loss = self.losses[0].compute(tf.concat(gt, axis=0), tf.concat(predict, axis=0))  # averag
        other_losses = self.model.regu()  # regularizer
        logs['other_losses'] = sum(other_losses).numpy()
        logs['ori_loss'] = ce_loss.numpy()
        # update validation confusion matrix
        pre = tuple(map(lambda e: tf.cast(tf.argmax(e, axis=-1), 'int32').numpy(), predict))
        gt_ = tuple(e.numpy() for e in gt)
        tr_gt_ = tuple(e.numpy() for e in tr_gt)
        self.val_cm.record(gt_, pre, tr_gt_)
        # update validation metrics
        self.val_metrics[0].update_state(tf.concat(gt, axis=0), tf.concat(predict, axis=0))
        cm = tf.convert_to_tensor(self.val_cm.matrix_n)
        for metric in self.val_metrics[1:]:
            metric.update_state(cm)
        # reserve right or not
        right = []
        for p, g in zip(pre, gt_):
            right.append((p == g).astype('int32'))
        return next_ini, next_o, ch_focus, win_focus, logs, right

    def reset_cm(self, archive, *args, arch_phase='cm_result', arch_obj='all'):
        if archive:
            if arch_obj == 'all':
                self.manager.archive(arch_phase, {'train_multi_m': [self.train_cm.matrix_n],
                                                  'train_multi_m_acc': [self.train_cm.compute_acc('n')],
                                                  'train_transit_info': [self.train_cm.transit],
                                                  'train_on_transit_acc': [self.train_cm.compute_acc('trans')],
                                                  'val_multi_m': [self.val_cm.matrix_n],
                                                  'val_multi_m_acc': [self.val_cm.compute_acc('n')],
                                                  'val_transit_info': [self.val_cm.transit],
                                                  'val_on_transit_acc': [self.val_cm.compute_acc('trans')]}, *args)
                self.train_cm.reset()
                self.val_cm.reset()
            elif arch_obj == 'val' or arch_obj == 'train':
                obj = getattr(self, arch_obj + '_cm')
                self.manager.archive(arch_phase, {arch_obj + 'multi_m': [obj.matrix_n],
                                                  arch_obj + 'multi_m_acc': [obj.compute_acc('n')],
                                                  arch_obj + 'transit_info': [obj.transit],
                                                  arch_obj + 'on_transit_acc': [obj.compute_acc('trans')]}, *args)
                obj.reset()
        else:
            self.train_cm.reset()
            self.val_cm.reset()

    def _train_on_epoch(self, epoch_id, repeat):
        # prepare data --- take the amount of data into consideration --- get iteration by iteration ?
        # note that one file's data constitute a big sequence
        # train
        self.set_phase('training')
        epoch_log = {'total_samples': self.total_samples}
        self.callback.on_epoch_begin(epoch_id, epoch_log)
        loader = SeqDataLoader(self.manager.data_type, self.data_load_mode, self.data_input_type, self.unit_size,
                               self.file_num)
        # to start loader
        loader.start(self.experiment_set[0])
        batch_id = 0
        initial_states = None
        last_outputs = None
        while not loader.is_exhausted:
            # print('batch id: %4d' % batch_id)
            inp, y, tr_y, mask, self.current_ba_frames, no_begin = loader.get_one_batch_data()
            # convert to tensor
            inp = tuple(map(lambda e: tf.constant(e) if e is not None else None, inp))
            # inp = tuple(map(lambda e, n: tf.constant(e, name='input_seq_data_%s' % n), inp,
            #                ('time',) + (('fre',) if len(inp) == 2 else ('fre_des', 'fre_inc'))))
            y = tuple(tf.constant(y_i, dtype='int32') for y_i in y)
            tr_y = tuple(tf.constant(y_i, dtype='int32') for y_i in tr_y)
            sw = tuple(tf.gather(self.class_weights['n_class'], y_i) for y_i in y)
            if mask is not None:
                mask = tf.constant(mask, dtype='bool')
            no_begin = tf.constant(no_begin, dtype='bool')
            batch_log = {'batch_size': self.current_ba_frames}
            initial_states, last_outputs, batch_log = self._train_step(inp, y, tr_y, mask, initial_states,
                                                                       last_outputs, no_begin, sw, batch_log)

            # tackle next_ini, next_o[, next_refer]
            store_update = tf.constant(loader.exhausted_info(), dtype='bool')
            if not tf.reduce_all(store_update):
                initial_states, last_outputs, _ = self.model.reset_ele(store_update, initial_states, last_outputs)
            metrics_log = OrderedDict()
            for k, fn in zip(self.model.metrics_na[2:], self.train_metrics):
                metrics_log[k] = fn.result().numpy()  # ndarray
            self.callback.on_train_batch_end(batch_id, {'batch_log': batch_log, 'metrics_log': metrics_log})
            # imperative record used to save and restore model
            # =================================================================================================
            self.global_iters += 1
            batch_id += 1
        # archive metric history
        archive_log = {}
        for k, accu_v in zip(self.model.metrics_na, self.model.history):
            archive_log[k] = accu_v
        self.manager.archive(self.phase, archive_log, 'metrics', epoch=str(epoch_id))    # 归档的是含历史epoch在内的
        # archive lr history
        self.manager.archive(self.phase, np.array(self.callback.lr_history), 'lr', epoch=str(epoch_id),
                             columns=list('opt_%d' % i for i in range(len(self.optimizers))))
        # reset metrics
        self.reset_metrics('train')
        # archive weight l2
        weights_log = {'name': [], 'weight_l2_min': [], 'weight_l2_max': []}
        for n, l2 in zip(self.trainable_weights_names, self.trainable_weights_l2):
            weights_log['name'].append(n)
            weights_log['weight_l2_min'].append(np.min(np.asarray(l2)))
            weights_log['weight_l2_max'].append(np.max(np.asarray(l2)))
        self.manager.archive(self.phase, weights_log, 'weight_l2_refer', epoch=str(epoch_id))
        # because of long running time, to save model per epoch
        saved_lr = []
        for opt in self.optimizers:
            saved_lr.append(opt.learning_rate.numpy())
        save_dir = os.sep.join([self.manager.record_dir, 'No.%d_repeat' % (repeat + 1), 'trained_model'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # save trainer
        utils.save_trainer(self.model, os.sep.join([save_dir, str(epoch_id)]), saved_lr, self.opt_weights)
        # clear weight l2 refer
        for l in self.trainable_weights_l2:
            l.clear()
        # close data loader
        loader.stop()

        # validation
        self.set_phase('evaluating')
        self.callback.on_test_begin()
        # to test trained model, so keep equal file_num --- relevant to weights shape --- so use the same loader
        loader.start(self.experiment_set[1])
        batch_id = 0
        initial_states = None
        last_outputs = None
        attention_log = OrderedDict()
        while not loader.is_exhausted:
            # print('batch id: %4d' % batch_id)
            inp, y, tr_y, mask, self.current_ba_frames, no_begin = loader.get_one_batch_data()
            # tackle attention_log
            if len(attention_log) == 0:
                for k in loader.ks:
                    attention_log[k] = {'file_name': ntpath.basename(loader.warehouse[k]['file_path']),
                                        'attentions': {
                                            'ch': np.array((), dtype=self.model.dtype),
                                            'win': np.array((), dtype=self.model.dtype),
                                            'right': np.array((), dtype='int32')}}
            else:
                for i, flag in enumerate(no_begin):
                    k = loader.ks[i]
                    if flag and attention_log[k] is not None:
                        # archive this channel's file attention
                        self.manager.archive(self.phase, attention_log[k], 'attentions')
                        # update
                        if loader.warehouse[k]['id'] != -1:
                            attention_log[k] = {'file_name': ntpath.basename(loader.warehouse[k]['file_path']),
                                                'attentions': {'ch': np.array((), dtype=self.model.dtype),
                                                               'win': np.array((), dtype=self.model.dtype),
                                                               'right': np.array((), dtype='int32')}}
                        else:
                            attention_log[k] = None   # 已经没有文件可读了
            # convert to tensor
            inp = tuple(map(lambda e: tf.constant(e) if e is not None else None, inp))
            # inp = tuple(map(lambda e, n: tf.constant(e, name='input_seq_data_%s' % n), inp,
            #                ('time',) + (('fre',) if len(inp) == 2 else ('fre_des', 'fre_inc'))))
            y = tuple(tf.constant(y_i, dtype='int32') for y_i in y)
            tr_y = tuple(tf.constant(y_i, dtype='int32') for y_i in tr_y)
            # validation doesn't use sample_weights
            if mask is not None:
                mask = tf.constant(mask, dtype='bool')
            no_begin = tf.constant(no_begin, dtype='bool')
            batch_log = {'batch_size': self.current_ba_frames}
            initial_states, last_outputs, ch_attention, win_attention, batch_log, right = \
                self._test_step(inp, y, tr_y, mask, initial_states, last_outputs, no_begin, batch_log)

            # tackle next_ini, next_o[, next_refer]
            info = loader.exhausted_info()
            store_update = tf.constant(info, dtype='bool')
            if not tf.reduce_all(store_update):
                initial_states, last_outputs, _ = self.model.reset_ele(store_update, initial_states, last_outputs)
            # tackle attention, may be tuple/list type
            for i, (ca, wa, r) in enumerate(zip(ch_attention, win_attention, right)):
                k = loader.ks[i]
                if ca.shape[0] != 0:  # in other words, attention_log[k] is not None, because of judgement
                    # before test and after loaded
                    ca = ca.numpy()
                    wa = wa.numpy()
                    old = attention_log[k]['attentions']['ch']
                    attention_log[k]['attentions']['ch'] = np.concatenate(
                        (old.reshape((old.shape[0],) + ca.shape[1:]),
                         ca), axis=0)
                    old = attention_log[k]['attentions']['win']
                    attention_log[k]['attentions']['win'] = np.concatenate(
                        (old.reshape((old.shape[0],) + wa.shape[1:]),
                         wa), axis=0)
                    old = attention_log[k]['attentions']['right']
                    attention_log[k]['attentions']['right'] = np.concatenate((old, r), axis=0)
            metrics_log = OrderedDict()
            for k, fn in zip(self.model.metrics_na[2:], self.val_metrics):
                metrics_log[k] = fn.result().numpy()
            self.callback.on_test_batch_end(batch_id, {'batch_log': batch_log, 'metrics_log': metrics_log})
            batch_id += 1

        loader.stop()
        for k, v in attention_log.items():
            if v is not None:
                self.manager.archive(self.phase, v, 'attentions')
        # archive history
        archive_log = {}
        for k, accu_v in zip(self.model.metrics_na, self.model.history):
            archive_log[k] = accu_v
        self.manager.archive(self.phase, archive_log, 'metrics', epoch=str(epoch_id))
        self.reset_metrics('val')
        self.callback.on_test_end(metrics_log)

        self.callback.on_epoch_end(epoch_id)
        self.reset_cm(True)
        # archive weights   use 'allow_pickle=True' when loading, save train result
        self.manager.archive('training', self.callback.train_hold, 'info')
        self.manager.archive('evaluating', self.callback.val_hold, 'info')
        for idx, opt in enumerate(self.optimizers):
            self.epoch_regulate_lr(self.config.base_lr[idx], self.config.min_lr[idx], opt)

    def _fully_test(self, collect):
        self.set_phase('fully_evaluating')
        self.callback.on_fully_test_begin()
        loader = SeqDataLoader(self.manager.data_type, self.data_load_mode, self.data_input_type, self.unit_size,
                               self.file_num)
        # to start loader
        loader.start(collect)
        batch_id = 0
        initial_states = None
        last_outputs = None
        attention_log = OrderedDict()
        while not loader.is_exhausted:
            # print('batch id: %4d' % batch_id)
            inp, y, tr_y, mask, self.current_ba_frames, no_begin = loader.get_one_batch_data()
            # tackle attention_log
            if len(attention_log) == 0:
                for k in loader.ks:
                    attention_log[k] = {'file_name': ntpath.basename(loader.warehouse[k]['file_path']),
                                        'attentions': {
                                            'ch': np.array((), dtype=self.model.dtype),
                                            'win': np.array((), dtype=self.model.dtype),
                                            'right': np.array((), dtype='int32')}}
            else:
                for i, flag in enumerate(no_begin):
                    k = loader.ks[i]
                    if flag and attention_log[k] is not None:
                        # archive this channel's file attention
                        self.manager.archive(self.phase, attention_log[k], 'attentions')
                        # update
                        if loader.warehouse[k]['id'] != -1:
                            attention_log[k] = {'file_name': ntpath.basename(loader.warehouse[k]['file_path']),
                                                'attentions': {'ch': np.array((), dtype=self.model.dtype),
                                                               'win': np.array((), dtype=self.model.dtype),
                                                               'right': np.array((), dtype='int32')}}
                        else:
                            attention_log[k] = None
            # convert to tensor
            inp = tuple(map(lambda e: tf.constant(e) if e is not None else None, inp))
            # inp = tuple(map(lambda e, n: tf.constant(e, name='input_seq_data_%s' % n), inp,
            #                ('time',) + (('fre',) if len(inp) == 2 else ('fre_des', 'fre_inc'))))
            y = tuple(tf.constant(y_i, dtype='int32') for y_i in y)
            tr_y = tuple(tf.constant(y_i, dtype='int32') for y_i in tr_y)
            # validation doesn't use sample_weights
            if mask is not None:
                mask = tf.constant(mask, dtype='bool')
            no_begin = tf.constant(no_begin, dtype='bool')
            batch_log = {'batch_size': self.current_ba_frames}
            initial_states, last_outputs, ch_attention, win_attention, batch_log, right = \
                self._test_step(inp, y, tr_y, mask, initial_states, last_outputs, no_begin, batch_log)
            # tackle next_ini, next_o[, next_refer]
            info = loader.exhausted_info()
            store_update = tf.constant(info, dtype='bool')
            if not tf.reduce_all(store_update):
                initial_states, last_outputs, _ = self.model.reset_ele(store_update, initial_states, last_outputs)
            # tackle attention, may be tuple/list type
            for i, (ca, wa, r) in enumerate(zip(ch_attention, win_attention, right)):
                k = loader.ks[i]
                if ca.shape[0] != 0:  # in other words, attention_log[k] is not None
                    ca = ca.numpy()
                    wa = wa.numpy()
                    old = attention_log[k]['attentions']['ch']
                    attention_log[k]['attentions']['ch'] = np.concatenate(
                        (old.reshape((old.shape[0],) + ca.shape[1:]),
                         ca), axis=0)
                    old = attention_log[k]['attentions']['win']
                    attention_log[k]['attentions']['win'] = np.concatenate(
                        (old.reshape((old.shape[0],) + wa.shape[1:]),
                         wa), axis=0)
                    old = attention_log[k]['attentions']['right']
                    attention_log[k]['attentions']['right_old'] = np.concatenate((old, r), axis=0)
            metrics_log = OrderedDict()
            for k, fn in zip(self.model.metrics_na[2:], self.val_metrics):
                metrics_log[k] = fn.result().numpy()
            self.callback.on_fully_test_batch_end(batch_id, {'batch_log': batch_log, 'metrics_log': metrics_log})
            batch_id += 1

        loader.stop()
        for k, v in attention_log.items():
            if v is not None:
                self.manager.archive(self.phase, v, 'attentions')
        # archive history
        archive_log = {}
        for k, accu_v in zip(self.model.metrics_na, self.model.history):
            archive_log[k] = accu_v
        self.manager.archive(self.phase, archive_log, 'metrics', epoch='full_evaluation')
        self.reset_metrics('val')
        self.callback.on_fully_test_end(metrics_log)
        self.reset_cm(True, self.phase, arch_obj='val')


class MultitaskTrainer(Trainer):
    def __init__(self, config, manager, callback, data_load_mode):
        super(MultitaskTrainer, self).__init__(config, manager, callback, data_load_mode)

    # inp must be tuple/list, gt and sample_weights must be tuple/list
    def _train_step(self, inp, gt, tr_gt, mask, ini, last_o, no_begin, refer, sw_n, sw_2, logs):
        with tf.GradientTape() as tape:
            # chs_attention and wins_attention use to visualize, thus, valid in predicting/evaluating
            if mask is not None:  # 第二种上数据上数据方式的模型 --- 'batch' 该模型还未修改、完善
                predict_n, predict_2, next_ini, next_o, _, _, next_refer = self.model(inp, mask, ini, last_o,
                                                                                      no_begin,
                                                                                      True, refer, True)
            else:
                predict_n, predict_2, next_ini, next_o, _, _, next_refer = self.model(inp, ini, last_o, no_begin,
                                                                                      True,
                                                                                      refer, True)
            predict_n = list(tf.clip_by_value(p, 1e-8, 1.) for p in predict_n)
            predict_2 = list(tf.clip_by_value(p, 1e-8, 1.) for p in predict_2)
            ce_loss_n = self.losses[0].compute(tf.concat(gt, axis=0), tf.concat(predict_n, axis=0),
                                               sample_weight=tf.concat(sw_n, axis=0))  # average
            ce_loss_2 = self.losses[0].compute(tf.concat(tr_gt, axis=0), tf.concat(predict_2, axis=0),
                                               sample_weight=tf.concat(sw_2, axis=0))  # average
            if gt[0].shape[0] > 1:
                new_gt = list(g[1:] for g in gt)
                new_pre_n = list(map(lambda p1, p2: tf.reduce_sum(tf.multiply(
                    tf.expand_dims(p1[1:], axis=-2), tf.stack([p2[:-1], p2[1:]], axis=-1)), axis=-1),
                                     predict_2, predict_n))
                new_pre_n = list(tf.clip_by_value(p, 1e-8, 1.) for p in new_pre_n)
                new_sw_n = list(map(lambda w1, w2: tf.multiply(w1[1:], w2[1:]), sw_2, sw_n))
                batch_rectify_loss = self.losses[0].compute(tf.concat(new_gt, axis=0), tf.concat(new_pre_n, axis=0),
                                                            sample_weight=tf.concat(new_sw_n, axis=0))  # average
                # rectify batch_rectify_loss
                batch_rectify_loss *= (logs['batch_size'] - self.file_num) / logs['batch_size']
                new_pre_n = list(map(lambda p1, p2: tf.concat((p1[0: 1], p2), axis=0), predict_n, new_pre_n))
            else:
                batch_rectify_loss = tf.cast(0., ce_loss_n.dtype)
                new_pre_n = predict_n
            ori_loss = [ce_loss_n, ce_loss_2, batch_rectify_loss]
            other_losses = self.model.regu()  # regularizer      _callable_losses
            total_loss = ori_loss + other_losses
        logs['other_losses'] = sum(other_losses).numpy()
        # print('other loss: {}'.format(logs['other_losses']))
        logs['ori_loss'] = sum(ori_loss).numpy()
        w = self.model.trainable_weights  # non-nested list
        _w = tape.watched_variables()
        if self.trainable_weights_names is None:
            self.trainable_weights_names = []
            self.trainable_weights_l2 = []  # used for experiment
            for w_i in w:
                self.trainable_weights_names.append(w_i.name)
                self.trainable_weights_l2.append([tf.clip_by_value(tf.linalg.norm(w_i, ord=2), 1e-8, np.Inf).numpy()])
        else:
            for i, w_i in enumerate(w):
                self.trainable_weights_l2[i].append(tf.clip_by_value(tf.linalg.norm(w_i, ord=2), 1e-8, np.Inf).numpy())
        group_watched_w = utils.group_by_optimizers(_w, self.config.opt_groups)  # only one group temporarily
        g = tape.gradient(total_loss, group_watched_w)
        # for idx, f in enumerate(self.config.opt_clip_flag):
        #    if f:
        #        g[idx] = tf.clip_by_global_norm(g[idx], self.config.opt_clip_norm[idx])[0]
        logs['lr'] = []
        for idx, opt in enumerate(self.optimizers):
            self.lr_fun(self.config.base_lr[idx], self.config.ini_warm_lr[idx], self.config.min_lr[idx], opt)
            logs['lr'].append(opt.learning_rate.numpy())
            self._load_optimizer_weights(opt, group_watched_w[idx], self.opt_weights[idx])
            opt.apply_gradients(list(zip(g[idx], group_watched_w[idx])))
            self.opt_weights[idx] = opt.get_weights()
        # update confusion matrix
        pre_n = tuple(map(lambda e: tf.cast(tf.argmax(e, axis=-1), 'int32').numpy(), predict_n))
        pre_2 = tuple(map(lambda e: tf.cast(tf.argmax(e, axis=-1), 'int32').numpy(), predict_2))
        n_pre = tuple(map(lambda e: tf.cast(tf.argmax(e, axis=-1), 'int32').numpy(), new_pre_n))
        gt_ = tuple(e.numpy() for e in gt)
        tr_gt_ = tuple(e.numpy() for e in tr_gt)
        self.train_cm.record_plus_trans(gt_, pre_n, n_pre, tr_gt_, pre_2)
        # update train metrics
        self.train_metrics[0].update_state(tf.concat(gt, axis=0), tf.concat(new_pre_n, axis=0))
        cm = tf.convert_to_tensor(self.train_cm.matrix_multitask)
        for metric in self.train_metrics[1:]:
            metric.update_state(cm)
        # tackle next_init, next_o[, next_refer]
        return next_ini, next_o, next_refer, logs

    def _test_step(self, inp, gt, tr_gt, mask, ini, last_o, no_begin, refer, logs):
        # chs_attention and wins_attention use to visualize, thus, valid in predicting/evaluating
        if mask is not None:   # 第二种上数据上数据方式的模型 --- 'batch' 该模型还未修改、完善
            predict_n, predict_2, next_ini, next_o, ch_focus, win_focus, next_refer = \
                self.model(inp, mask, ini, last_o, no_begin, False, refer, True)
        else:
            predict_n, predict_2, next_ini, next_o, ch_focus, win_focus, next_refer = \
                self.model(inp, ini, last_o, no_begin, False, refer, True)
        predict_n = list(tf.clip_by_value(p, 1e-8, 1.) for p in predict_n)
        predict_2 = list(tf.clip_by_value(p, 1e-8, 1.) for p in predict_2)
        ce_loss_n = self.losses[0].compute(tf.concat(gt, axis=0), tf.concat(predict_n, axis=0))  # average
        ce_loss_2 = self.losses[0].compute(tf.concat(tr_gt, axis=0), tf.concat(predict_2, axis=0))  # average
        if gt[0].shape[0] > 1:
            new_gt = list(g[1:] for g in gt)
            new_pre_n = list(map(lambda p1, p2: tf.reduce_sum(tf.multiply(
                tf.expand_dims(p1[1:], axis=-2), tf.stack([p2[:-1], p2[1:]], axis=-1)), axis=-1),
                                 predict_2, predict_n))
            new_pre_n = list(tf.clip_by_value(p, 1e-8, 1.) for p in new_pre_n)
            batch_rectify_loss = self.losses[0].compute(tf.concat(new_gt, axis=0), tf.concat(new_pre_n, axis=0))
            # rectify batch_rectify_loss
            batch_rectify_loss *= (logs['batch_size'] - self.file_num) / logs['batch_size']
            new_pre_n = list(map(lambda p1, p2: tf.concat((p1[0: 1], p2), axis=0), predict_n, new_pre_n))
        else:
            batch_rectify_loss = tf.cast(0., ce_loss_n.dtype)
            new_pre_n = predict_n
        ori_loss = [ce_loss_n, ce_loss_2, batch_rectify_loss]
        other_losses = self.model.regu()  # regularizer
        logs['other_losses'] = sum(other_losses).numpy()
        logs['ori_loss'] = sum(ori_loss).numpy()
        # update validation confusion matrix
        pre = tuple(map(lambda e: tf.cast(tf.argmax(e, axis=-1), 'int32').numpy(), predict_n))
        tr_pre = tuple(map(lambda e: tf.cast(tf.argmax(e, axis=-1), 'int32').numpy(), predict_2))
        new_pre = tuple(map(lambda e: tf.cast(tf.argmax(e, axis=-1), 'int32').numpy(), new_pre_n))
        gt_ = tuple(e.numpy() for e in gt)
        tr_gt_ = tuple(e.numpy() for e in tr_gt)
        self.val_cm.record_plus_trans(gt_, pre, new_pre, tr_gt_, tr_pre)
        # update validation metrics
        self.val_metrics[0].update_state(tf.concat(gt, axis=0), tf.concat(new_pre_n, axis=0))
        cm = tf.convert_to_tensor(self.val_cm.matrix_multitask)
        for metric in self.val_metrics[1:]:
            metric.update_state(cm)
        # reserve right or not
        right_old = []
        right_new = []
        for p1, p2, g in zip(pre, new_pre, gt_):
            right_old.append((p1 == g).astype('int32'))
            right_new.append((p2 == g).astype('int32'))
        return next_ini, next_o, next_refer, ch_focus, win_focus, logs, right_old, right_new

    def reset_cm(self, archive, *args, arch_phase='cm_result', arch_obj='all'):
        if archive:
            if arch_obj == 'all':
                self.manager.archive(arch_phase, {'train_old_multi_m': [self.train_cm.matrix_n],
                                                  'train_old_multi_m_acc': [self.train_cm.compute_acc('n')],
                                                  'train_new_multi_m': [self.train_cm.matrix_multitask],
                                                  'train_new_multi_m_acc': [self.train_cm.compute_acc('multitask')],
                                                  'train_2_m': [self.train_cm.matrix_2],
                                                  'train_2_m_acc': [self.train_cm.compute_acc('2')],
                                                  'train_old_transit': [self.train_cm.transit],
                                                  'train_on_old_transit_acc': [self.train_cm.compute_acc('trans')],
                                                  'train_new_transit': [self.train_cm.transit_multitask],
                                                  'train_on_new_transit_acc': [self.train_cm.compute_acc('multi_trans')],
                                                  'val_old_multi_m': [self.val_cm.matrix_n],
                                                  'val_old_multi_m_acc': [self.val_cm.compute_acc('n')],
                                                  'val_new_multi_m': [self.val_cm.matrix_multitask],
                                                  'val_new_multi_m_acc': [self.val_cm.compute_acc('multitask')],
                                                  'val_2_m': [self.val_cm.matrix_2],
                                                  'val_2_m_acc': [self.val_cm.compute_acc('2')],
                                                  'val_old_transit': [self.val_cm.transit],
                                                  'val_on_old_transit_acc': [self.val_cm.compute_acc('trans')],
                                                  'val_new_transit': [self.val_cm.transit_multitask],
                                                  'val_on_new_transit_acc': [self.val_cm.compute_acc('multi_trans')]},
                                     *args)
                self.train_cm.reset()
                self.val_cm.reset()
            elif arch_obj == 'val' or arch_obj == 'train':
                obj = getattr(self, arch_obj + '_cm')
                self.manager.archive(arch_phase, {arch_obj + 'old_multi_m': [obj.matrix_n],
                                                  arch_obj + 'old_multi_m_acc': [obj.compute_acc('n')],
                                                  arch_obj + 'new_multi_m': [obj.matrix_multitask],
                                                  arch_obj + 'new_multi_m_acc': [obj.compute_acc('multitask')],
                                                  arch_obj + '2_m': [obj.matrix_2],
                                                  arch_obj + '2_m_acc': [obj.compute_acc('2')],
                                                  arch_obj + 'old_transit': [obj.transit],
                                                  arch_obj + 'on_old_transit_acc': [obj.compute_acc('trans')],
                                                  arch_obj + 'new_transit': [obj.transit_multitask],
                                                  arch_obj + 'on_new_transit_acc': [
                                                      obj.compute_acc('multi_trans')]}, *args)
                obj.reset()
        else:
            self.train_cm.reset()
            self.val_cm.reset()

    def _train_on_epoch(self, epoch_id, repeat):
        # prepare data --- take the amount of data into consideration --- get iteration by iteration ?
        # note that one file's data constitute a big sequence
        # train
        self.set_phase('training')
        epoch_log = {'total_samples': self.total_samples}
        self.callback.on_epoch_begin(epoch_id, epoch_log)
        loader = SeqDataLoader(self.manager.data_type, self.data_load_mode, self.data_input_type, self.unit_size,
                               self.file_num)
        # to start loader
        loader.start(self.experiment_set[0])
        batch_id = 0
        initial_states = None
        last_outputs = None
        refer = None
        while not loader.is_exhausted:
            # print('batch id: %4d' % batch_id)
            inp, y, tr_y, mask, self.current_ba_frames, no_begin = loader.get_one_batch_data()
            # convert to tensor
            inp = tuple(map(lambda e: tf.constant(e) if e is not None else None, inp))
            # inp = tuple(map(lambda e, n: tf.constant(e, name='input_seq_data_%s' % n), inp,
            #                ('time',) + (('fre',) if len(inp) == 2 else ('fre_des', 'fre_inc'))))
            y = tuple(tf.constant(y_i, dtype='int32') for y_i in y)
            tr_y = tuple(tf.constant(y_i, dtype='int32') for y_i in tr_y)
            sw_n = tuple(tf.gather(self.class_weights['n_class'], y_i) for y_i in y)
            sw_2 = tuple(tf.gather(self.class_weights['trans'], y_i) for y_i in tr_y)
            if mask is not None:
                mask = tf.constant(mask, dtype='bool')
            no_begin = tf.constant(no_begin, dtype='bool')
            batch_log = {'batch_size': self.current_ba_frames}
            initial_states, last_outputs, refer, batch_log = self._train_step(inp, y, tr_y, mask, initial_states,
                                                                              last_outputs, no_begin, refer, sw_n,
                                                                              sw_2, batch_log)

            # tackle next_ini, next_o[, next_refer]
            store_update = tf.constant(loader.exhausted_info(), dtype='bool')
            if not tf.reduce_all(store_update):
                initial_states, last_outputs, refer = self.model.reset_ele(store_update, initial_states, last_outputs,
                                                                           refer)
            metrics_log = OrderedDict()
            for k, fn in zip(self.model.metrics_na[2:], self.train_metrics):
                metrics_log[k] = fn.result().numpy()  # ndarray
            self.callback.on_train_batch_end(batch_id, {'batch_log': batch_log, 'metrics_log': metrics_log})
            # imperative record used to save and restore model
            # =================================================================================================
            self.global_iters += 1
            batch_id += 1
        # archive metric history
        archive_log = {}
        for k, accu_v in zip(self.model.metrics_na, self.model.history):
            archive_log[k] = accu_v
        self.manager.archive(self.phase, archive_log, 'metrics', epoch=str(epoch_id))    # 归档的是含历史epoch在内的
        # archive lr history
        self.manager.archive(self.phase, np.array(self.callback.lr_history), 'lr', epoch=str(epoch_id),
                             columns=list('opt_%d' % i for i in range(len(self.optimizers))))
        # reset metrics
        self.reset_metrics('train')
        # archive weight l2
        weights_log = {'name': [], 'weight_l2_min': [], 'weight_l2_max': []}
        for n, l2 in zip(self.trainable_weights_names, self.trainable_weights_l2):
            weights_log['name'].append(n)
            weights_log['weight_l2_min'].append(np.min(np.asarray(l2)))
            weights_log['weight_l2_max'].append(np.max(np.asarray(l2)))
        self.manager.archive(self.phase, weights_log, 'weight_l2_refer', epoch=str(epoch_id))
        # because of long running time, to save model per epoch
        saved_lr = []
        for opt in self.optimizers:
            saved_lr.append(opt.learning_rate.numpy())
        save_dir = os.sep.join([self.manager.record_dir, 'No.%d_repeat' % (repeat + 1), 'trained_model'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # save trainer
        utils.save_trainer(self.model, os.sep.join([save_dir, str(epoch_id)]), saved_lr, self.opt_weights)
        # clear weight l2 refer
        for l in self.trainable_weights_l2:
            l.clear()
        # close data loader
        loader.stop()

        # validation
        self.set_phase('evaluating')
        self.callback.on_test_begin()
        # to test trained model, so keep equal file_num --- relevant to weights shape --- so use the same loader
        loader.start(self.experiment_set[1])
        batch_id = 0
        initial_states = None
        last_outputs = None
        refer = None
        attention_log = OrderedDict()
        while not loader.is_exhausted:
            # print('batch id: %4d' % batch_id)
            inp, y, tr_y, mask, self.current_ba_frames, no_begin = loader.get_one_batch_data()
            # tackle attention_log
            if len(attention_log) == 0:
                for k in loader.ks:
                    attention_log[k] = {'file_name': ntpath.basename(loader.warehouse[k]['file_path']),
                                        'attentions': {
                                            'ch': np.array((), dtype=self.model.dtype),
                                            'win': np.array((), dtype=self.model.dtype),
                                            'right_old': np.array((), dtype='int32'),
                                            'right_new': np.array((), dtype='int32')}}
            else:
                for i, flag in enumerate(no_begin):
                    k = loader.ks[i]
                    if flag and attention_log[k] is not None:
                        # archive this channel's file attention
                        self.manager.archive(self.phase, attention_log[k], 'attentions')
                        # update
                        if loader.warehouse[k]['id'] != -1:
                            attention_log[k] = {'file_name': ntpath.basename(loader.warehouse[k]['file_path']),
                                                'attentions': {'ch': np.array((), dtype=self.model.dtype),
                                                               'win': np.array((), dtype=self.model.dtype),
                                                               'right_old': np.array((), dtype='int32'),
                                                               'right_new': np.array((), dtype='int32')}}
                        else:
                            attention_log[k] = None   # 已经没有文件可读了
            # convert to tensor
            inp = tuple(map(lambda e: tf.constant(e) if e is not None else None, inp))
            # inp = tuple(map(lambda e, n: tf.constant(e, name='input_seq_data_%s' % n), inp,
            #                ('time',) + (('fre',) if len(inp) == 2 else ('fre_des', 'fre_inc'))))
            y = tuple(tf.constant(y_i, dtype='int32') for y_i in y)
            tr_y = tuple(tf.constant(y_i, dtype='int32') for y_i in tr_y)
            # validation doesn't use sample_weights
            if mask is not None:
                mask = tf.constant(mask, dtype='bool')
            no_begin = tf.constant(no_begin, dtype='bool')
            batch_log = {'batch_size': self.current_ba_frames}
            initial_states, last_outputs, refer, ch_attention, win_attention, batch_log, right_old, right_new = \
                self._test_step(inp, y, tr_y, mask, initial_states, last_outputs, no_begin, refer, batch_log)

            # tackle next_ini, next_o[, next_refer]
            info = loader.exhausted_info()
            store_update = tf.constant(info, dtype='bool')
            if not tf.reduce_all(store_update):
                initial_states, last_outputs, refer = self.model.reset_ele(store_update, initial_states, last_outputs,
                                                                           refer)
            # tackle attention, may be tuple/list type
            for i, (ca, wa, r_o, r_n) in enumerate(zip(ch_attention, win_attention, right_old, right_new)):
                k = loader.ks[i]
                if ca.shape[0] != 0:  # in other words, attention_log[k] is not None, because of judgement
                    # before test and after loaded
                    ca = ca.numpy()
                    wa = wa.numpy()
                    old = attention_log[k]['attentions']['ch']
                    attention_log[k]['attentions']['ch'] = np.concatenate(
                        (old.reshape((old.shape[0],) + ca.shape[1:]),
                         ca), axis=0)
                    old = attention_log[k]['attentions']['win']
                    attention_log[k]['attentions']['win'] = np.concatenate(
                        (old.reshape((old.shape[0],) + wa.shape[1:]),
                         wa), axis=0)
                    old = attention_log[k]['attentions']['right_old']
                    attention_log[k]['attentions']['right_old'] = np.concatenate((old, r_o), axis=0)
                    old = attention_log[k]['attentions']['right_new']
                    attention_log[k]['attentions']['right_new'] = np.concatenate((old, r_n), axis=0)
            metrics_log = OrderedDict()
            for k, fn in zip(self.model.metrics_na[2:], self.val_metrics):
                metrics_log[k] = fn.result().numpy()
            self.callback.on_test_batch_end(batch_id, {'batch_log': batch_log, 'metrics_log': metrics_log})
            batch_id += 1

        loader.stop()
        for k, v in attention_log.items():
            if v is not None:
                self.manager.archive(self.phase, v, 'attentions')
        # archive history
        archive_log = {}
        for k, accu_v in zip(self.model.metrics_na, self.model.history):
            archive_log[k] = accu_v
        self.manager.archive(self.phase, archive_log, 'metrics', epoch=str(epoch_id))
        self.reset_metrics('val')
        self.callback.on_test_end(metrics_log)

        self.callback.on_epoch_end(epoch_id)
        self.reset_cm(True)
        # archive weights   use 'allow_pickle=True' when loading, save train result
        self.manager.archive('training', self.callback.train_hold, 'info')
        self.manager.archive('evaluating', self.callback.val_hold, 'info')
        for idx, opt in enumerate(self.optimizers):
            self.epoch_regulate_lr(self.config.base_lr[idx], self.config.min_lr[idx], opt)

    def _fully_test(self, collect):
        self.set_phase('fully_evaluating')
        self.callback.on_fully_test_begin()
        loader = SeqDataLoader(self.manager.data_type, self.data_load_mode, self.data_input_type, self.unit_size,
                               self.file_num)
        # to start loader
        loader.start(collect)
        batch_id = 0
        initial_states = None
        last_outputs = None
        refer = None
        attention_log = OrderedDict()
        while not loader.is_exhausted:
            # print('batch id: %4d' % batch_id)
            inp, y, tr_y, mask, self.current_ba_frames, no_begin = loader.get_one_batch_data()
            # tackle attention_log
            if len(attention_log) == 0:
                for k in loader.ks:
                    attention_log[k] = {'file_name': ntpath.basename(loader.warehouse[k]['file_path']),
                                        'attentions': {
                                            'ch': np.array((), dtype=self.model.dtype),
                                            'win': np.array((), dtype=self.model.dtype),
                                            'right_old': np.array((), dtype='int32'),
                                            'right_new': np.array((), dtype='int32')}}
            else:
                for i, flag in enumerate(no_begin):
                    k = loader.ks[i]
                    if flag and attention_log[k] is not None:
                        # archive this channel's file attention
                        self.manager.archive(self.phase, attention_log[k], 'attentions')
                        # update
                        if loader.warehouse[k]['id'] != -1:
                            attention_log[k] = {'file_name': ntpath.basename(loader.warehouse[k]['file_path']),
                                                'attentions': {'ch': np.array((), dtype=self.model.dtype),
                                                               'win': np.array((), dtype=self.model.dtype),
                                                               'right_old': np.array((), dtype='int32'),
                                                               'right_new': np.array((), dtype='int32')}}
                        else:
                            attention_log[k] = None
            # convert to tensor
            inp = tuple(map(lambda e: tf.constant(e) if e is not None else None, inp))
            # inp = tuple(map(lambda e, n: tf.constant(e, name='input_seq_data_%s' % n), inp,
            #                ('time',) + (('fre',) if len(inp) == 2 else ('fre_des', 'fre_inc'))))
            y = tuple(tf.constant(y_i, dtype='int32') for y_i in y)
            tr_y = tuple(tf.constant(y_i, dtype='int32') for y_i in tr_y)
            # validation doesn't use sample_weights
            if mask is not None:
                mask = tf.constant(mask, dtype='bool')
            no_begin = tf.constant(no_begin, dtype='bool')
            batch_log = {'batch_size': self.current_ba_frames}
            initial_states, last_outputs, refer, ch_attention, win_attention, batch_log, right_old, right_new = \
                self._test_step(inp, y, tr_y, mask, initial_states, last_outputs, no_begin, refer, batch_log)
            # tackle next_ini, next_o[, next_refer]
            info = loader.exhausted_info()
            store_update = tf.constant(info, dtype='bool')
            if not tf.reduce_all(store_update):
                initial_states, last_outputs, refer = self.model.reset_ele(store_update, initial_states, last_outputs,
                                                                           refer)
            # tackle attention, may be tuple/list type
            for i, (ca, wa, r_o, r_n) in enumerate(zip(ch_attention, win_attention, right_old, right_new)):
                k = loader.ks[i]
                if ca.shape[0] != 0:  # in other words, attention_log[k] is not None
                    ca = ca.numpy()
                    wa = wa.numpy()
                    old = attention_log[k]['attentions']['ch']
                    attention_log[k]['attentions']['ch'] = np.concatenate(
                        (old.reshape((old.shape[0],) + ca.shape[1:]),
                         ca), axis=0)
                    old = attention_log[k]['attentions']['win']
                    attention_log[k]['attentions']['win'] = np.concatenate(
                        (old.reshape((old.shape[0],) + wa.shape[1:]),
                         wa), axis=0)
                    old = attention_log[k]['attentions']['right_old']
                    attention_log[k]['attentions']['right_old'] = np.concatenate((old, r_o), axis=0)
                    old = attention_log[k]['attentions']['right_new']
                    attention_log[k]['attentions']['right_old'] = np.concatenate((old, r_n), axis=0)
            metrics_log = OrderedDict()
            for k, fn in zip(self.model.metrics_na[2:], self.val_metrics):
                metrics_log[k] = fn.result().numpy()
            self.callback.on_fully_test_batch_end(batch_id, {'batch_log': batch_log, 'metrics_log': metrics_log})
            batch_id += 1

        loader.stop()
        for k, v in attention_log.items():
            if v is not None:
                self.manager.archive(self.phase, v, 'attentions')
        # archive history
        archive_log = {}
        for k, accu_v in zip(self.model.metrics_na, self.model.history):
            archive_log[k] = accu_v
        self.manager.archive(self.phase, archive_log, 'metrics', epoch='full_evaluation')
        self.reset_metrics('val')
        self.callback.on_fully_test_end(metrics_log)
        self.reset_cm(True, self.phase, arch_obj='val')







