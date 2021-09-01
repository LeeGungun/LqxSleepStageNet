# -*- coding: utf-8 -*-
from tensorflow import keras
from metrics import *
from config import SeqDataLoader
import numpy as np
from collections import OrderedDict
import ntpath
import utils


class Predictor(object):
    def __init__(self, config, manager, callback, data_load_mode, print_step):
        super(Predictor, self).__init__()
        self.config = config
        self.manager = manager
        self.callback = callback
        if data_load_mode not in ('seq', 'batch'):
            raise Exception('unqualified parameter \'data_loader_mode\'')
        self.data_load_mode = data_load_mode    # ('seq', 'batch')
        self.data_input_type = config.inputs
        self.unit_size = config.batch_size if data_load_mode == 'seq' else config.seq_len
        self.file_num = config.file_num
        self.dtype = config.dtype
        self.class_num = config.classes
        self.metrics = None
        self.compiled = False
        self.model = None
        self.print_step = print_step

    def _register_model(self, model, classifier):
        self.model = model
        if classifier:
            self.callback.set_model(model)

    def compile(self):
        # metrics
        self.metrics = [keras.metrics.SparseCategoricalAccuracy(name='predict_overall_ACC', dtype=self.dtype),
                        MacroF1('predict_macro_f1', self.dtype),
                        F1('predict_multi_f1', n_class=self.class_num, dtype=self.dtype),
                        Kappa('predict_kappa', self.dtype),
                        Rate('predict_overall_recall', 'recall', mode='overall', dtype=self.dtype),
                        Rate('predict_multi_recall', 'recall', n_class=self.class_num, dtype=self.dtype),
                        Rate('predict_overall_ppv', 'precision', mode='overall', dtype=self.dtype),
                        Rate('predict_multi_ppv', 'precision', n_class=self.class_num, dtype=self.dtype)]
        self.stateful_metrics = ['overall_ACC', 'macro_f1', 'multi_f1', 'kappa', 'overall_recall', 'multi_recall',
                                 'overall_ppv', 'multi_ppv']
        self.cm = utils.ConfusionMatrix(utils.LABELS)
        self.compiled = True

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def archive_cm(self, multitask):
        if multitask:
            self.manager.archive('cm_result', {'old_multi_m': [self.cm.matrix_n],
                                               'old_multi_m_acc': [self.cm.compute_acc('n')],
                                               'new_multi_m': [self.cm.matrix_multitask],
                                               'new_multi_m_acc': [self.cm.compute_acc('multitask')],
                                               '2_m': [self.cm.matrix_2],
                                               '2_m_acc': [self.cm.compute_acc('2')],
                                               'old_transit': [self.cm.transit],
                                               'on_old_transit_acc': [self.cm.compute_acc('trans')],
                                               'new_transit': [self.cm.transit_multitask],
                                               'on_new_transit_acc': [self.cm.compute_acc('multi_trans')]},
                                 'predicting')
        else:
            self.manager.archive('cm_result', {'multi_m': [self.cm.matrix_n],
                                               'multi_m_acc': [self.cm.compute_acc('n')],
                                               'transit': [self.cm.transit],
                                               'on_transit_acc': [self.cm.compute_acc('trans')]}, 'predicting')
        self.cm.reset()

    def _predict_normal_step(self, inp, gt, tr_gt, mask, ini, last_o, no_begin):
        # chs_attention and wins_attention use to visualize, thus, valid in predicting/evaluating
        if mask is not None:   # 为第二种上数据方式（即'batch'方式）服务， 模型尚未整合完善
            predict, next_ini, next_o, ch_focus, win_focus = \
                self.model(inp, mask, ini, last_o, no_begin, False, True)
        else:
            predict, next_ini, next_o, ch_focus, win_focus = \
                self.model(inp, ini, last_o, no_begin, False, True)
        #predict = list(tf.clip_by_value(p, 1e-8, 1.) for p in predict)
        # update validation confusion matrix
        pre = tuple(map(lambda e: tf.cast(tf.argmax(e, axis=-1), 'int32').numpy(), predict))
        gt_ = tuple(e.numpy() for e in gt)
        tr_gt_ = tuple(e.numpy() for e in tr_gt)
        self.cm.record(gt_, pre, tr_gt_)
        # reserve right or not
        right = []
        for p, g in zip(pre, gt_):
            right.append((p == g).astype('int32'))
        # update validation metrics
        self.metrics[0].update_state(tf.concat(gt, axis=0), tf.concat(predict, axis=0))
        cm = tf.convert_to_tensor(self.cm.matrix_n)
        for metric in self.metrics[1:]:
            metric.update_state(cm)
        return next_ini, next_o, ch_focus, win_focus, right

    def _predict_normal(self):
        loader = SeqDataLoader(self.manager.data_type, self.data_load_mode, self.data_input_type, self.unit_size,
                               self.file_num)
        loader.start(self.manager.get_predict_set())
        batch_id = 0
        initial_states = None
        last_outputs = None
        attention_log = OrderedDict()
        while not loader.is_exhausted:
            inp, y, tr_y, mask, self.current_ba_frames, no_begin = loader.get_one_batch_data()
            # tackle attention_log
            if len(attention_log) == 0:
                for k in loader.ks:
                    attention_log[k] = {'file_name': ntpath.basename(loader.warehouse[k]['file_path']),
                                        'attentions': {'ch': np.array((), dtype=self.model.dtype),
                                                       'win': np.array((), dtype=self.model.dtype),
                                                       'right': np.array((), dtype='int32')}}
            else:
                for i, flag in enumerate(no_begin):
                    k = loader.ks[i]
                    if flag and attention_log[k] is not None:
                        # archive this channel's file attention
                        self.manager.archive('predicting', attention_log[k], 'attentions')
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
            y = tuple(tf.constant(y_i, dtype='int32') for y_i in y)
            tr_y = tuple(tf.constant(y_i, dtype='int32') for y_i in tr_y)
            # validation doesn't use sample_weights
            if mask is not None:
                mask = tf.constant(mask, dtype='bool')
            no_begin = tf.constant(no_begin, dtype='bool')
            initial_states, last_outputs, ch_attention, win_attention, right = \
                self._predict_normal_step(inp, y, tr_y, mask, initial_states, last_outputs, no_begin)
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
                    attention_log[k]['attentions']['ch'] = np.concatenate((old.reshape((old.shape[0],) + ca.shape[1:]),
                                                                           ca), axis=0)
                    old = attention_log[k]['attentions']['win']
                    attention_log[k]['attentions']['win'] = np.concatenate((old.reshape((old.shape[0],) + wa.shape[1:]),
                                                                            wa), axis=0)
                    old = attention_log[k]['attentions']['right']
                    attention_log[k]['attentions']['right'] = np.concatenate((old, r), axis=0)
            metrics_log = OrderedDict()
            for k, fn in zip(self.stateful_metrics, self.metrics):
                # why? even though custom metrics classes' 'result' methods have set to return numpy, but invalid
                metrics_log[k] = fn.result().numpy()
            self.callback.on_predict_batch_end(batch_id, {'batch_size': self.current_ba_frames,
                                                          'metrics_log': metrics_log})
            batch_id += 1
        loader.stop()
        for k, v in attention_log.items():
            if v is not None:
                self.manager.archive('predicting', v, 'attentions')
        # archive history
        archive_log = {}
        for k, acc_v in zip(self.model.metrics_na, self.model.history):
            archive_log[k] = acc_v
        self.manager.archive('predicting', archive_log, 'metrics')
        self.callback.on_predict_end(metrics_log)
        self.archive_cm(False)

    def _no_predict_step(self, inp, mask, ini, last_o, no_begin, refer, multitask):
        # mask为第二种上数据方式（即'batch'方式）服务， 模型尚未整合完善
        if mask is not None and multitask:
            top_features, _, next_ini, next_o, ch_focus, win_focus, next_refer = \
                self.model(inp, mask, ini, last_o, no_begin, False, refer, False)
        elif mask is None and multitask:
            top_features, _, next_ini, next_o, ch_focus, win_focus, next_refer = \
                self.model(inp, ini, last_o, no_begin, False, refer, False)
        elif mask is not None and not multitask:
            top_features, next_ini, next_o, ch_focus, win_focus = \
                self.model(inp, mask, ini, last_o, no_begin, False, False)
            next_refer = None
        else:
            top_features, next_ini, next_o, ch_focus, win_focus = \
                self.model(inp, ini, last_o, no_begin, False, False)
            next_refer = None
        top_features = list(tf.clip_by_value(p, 1e-8, 1.) for p in top_features)
        return top_features, next_ini, next_o, next_refer, ch_focus, win_focus

    def _no_predict(self, multitask):
        import os
        import sys
        _dynamic_display = (hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()) or \
                           ('ipykernel' in sys.modules) or ('posix' in sys.modules) or \
                           ('PYCHARM_HOSTED' in os.environ)
        _console_width = 0
        print('============================= start fetch top features ================================')
        loader = SeqDataLoader(self.manager.data_type, self.data_load_mode, self.data_input_type, self.unit_size,
                               self.file_num)
        loader.start(self.manager.get_predict_set())
        initial_states = None
        last_outputs = None
        refer = None
        attention_log = OrderedDict()
        feature_log = OrderedDict()
        seen = 0
        batch_id = 0
        while not loader.is_exhausted:
            inp, _, _, mask, self.current_ba_frames, no_begin = loader.get_one_batch_data()
            # tackle attention_log 和 feature_log （二者同步）
            if len(attention_log) == 0:
                for k in loader.ks:
                    file_name = ntpath.basename(loader.warehouse[k]['file_path'])
                    attention_log[k] = {'file_name': file_name,
                                        'attentions': {'ch': np.array((), dtype=self.dtype),
                                                       'win': np.array((), dtype=self.dtype)}}
                    feature_log[k] = {
                        'file_name': file_name,
                        'top_feature': np.array((), dtype=self.dtype)
                    }
            else:
                for i, flag in enumerate(no_begin):
                    k = loader.ks[i]
                    if flag and attention_log[k] is not None:    # attention_log 和 feature_log同步
                        # archive this channel's file attention
                        self.manager.archive('predicting', attention_log[k], 'attentions')
                        self.manager.archive('top_feature', feature_log[k])
                        # update
                        if loader.warehouse[k]['id'] != -1:
                            file_name = ntpath.basename(loader.warehouse[k]['file_path'])
                            attention_log[k] = {'file_name': file_name,
                                                'attentions': {'ch': np.array((), dtype=self.model.dtype),
                                                               'win': np.array((), dtype=self.model.dtype)}}
                            feature_log[k] = {
                                'file_name': file_name,
                                'top_feature': np.array((), dtype=self.dtype)
                            }
                        else:
                            attention_log[k] = None
                            feature_log[k] = None
            # convert to tensor
            inp = tuple(map(lambda e: tf.constant(e) if e is not None else None, inp))
            # validation doesn't use sample_weights
            if mask is not None:
                mask = tf.constant(mask, dtype='bool')
            no_begin = tf.constant(no_begin, dtype='bool')
            top_features, initial_states, last_outputs, refer, ch_attention, win_attention = \
                self._no_predict_step(inp, mask, initial_states, last_outputs, no_begin, refer, multitask)

            # tackle next_ini, next_o[, next_refer]
            info = loader.exhausted_info()
            store_update = tf.constant(info, dtype='bool')
            if not tf.reduce_all(store_update):
                initial_states, last_outputs, refer = self.model.reset_ele(store_update, initial_states,
                                                                           last_outputs, refer)
            # tackle attention, may be tuple/list type
            for i, (ca, wa, f) in enumerate(zip(ch_attention, win_attention, top_features)):
                k = loader.ks[i]
                if ca.shape[0] != 0:  # in other words, attention_log[k] is not None
                    ca = ca.numpy()
                    wa = wa.numpy()
                    f = f.numpy()
                    old = attention_log[k]['attentions']['ch']
                    attention_log[k]['attentions']['ch'] = np.concatenate((old.reshape((old.shape[0],) + ca.shape[1:]),
                                                                           ca), axis=0)
                    old = attention_log[k]['attentions']['win']
                    attention_log[k]['attentions']['win'] = np.concatenate((old.reshape((old.shape[0],) + wa.shape[1:]),
                                                                            wa), axis=0)
                    old = feature_log[k]['top_feature']
                    feature_log[k]['top_feature'] = np.concatenate((old.reshape((old.shape[0],) + f.shape[1:]), f),
                                                                   axis=0)
            seen += self.current_ba_frames
            batch_id += 1
            if batch_id % self.print_step == 0:
                if _dynamic_display:
                    sys.stdout.write('\b' * _console_width)
                    sys.stdout.write('\r')
                else:
                    sys.stdout.write('\n')
                string = 'have dealt with {} samples, continue waiting...'.format(seen)
                if _console_width > len(string):
                    string = ' ' * (_console_width - len(string)) + string
                _console_width = len(string)
                sys.stdout.write(string)
                sys.stdout.flush()

        loader.stop()
        for k, v in attention_log.items():
            if v is not None:
                self.manager.archive('predicting', v, 'attentions')
        for k, v in feature_log.items():
            if v is not None:
                self.manager.archive('top_feature', v)
        sys.stdout.write('\n================================ stop fetch top features ================================\n')
        sys.stdout.flush()

    def _predict_multitask_step(self, inp, gt, tr_gt, mask, ini, last_o, no_begin, refer):
        # chs_attention and wins_attention use to visualize, thus, valid in predicting/evaluating
        if mask is not None:   # 为第二种上数据方式（即'batch'方式）服务， 模型尚未整合完善
            predict_n, predict_2, next_ini, next_o, ch_focus, win_focus, next_refer = \
                self.model(inp, mask, ini, last_o, no_begin, False, refer, True)
        else:
            predict_n, predict_2, next_ini, next_o, ch_focus, win_focus, next_refer = \
                self.model(inp, ini, last_o, no_begin, False, refer, True)
        #predict_n = list(tf.clip_by_value(p, 1e-8, 1.) for p in predict_n)
        #predict_2 = list(tf.clip_by_value(p, 1e-8, 1.) for p in predict_2)
        if gt[0].shape[0] > 1:
            new_pre_n = list(map(lambda p1, p2: tf.reduce_sum(tf.multiply(
                tf.expand_dims(p1[1:], axis=-2), tf.stack([p2[:-1], p2[1:]], axis=-1)), axis=-1),
                                 predict_2, predict_n))
            new_pre_n = list(map(lambda p1, p2: tf.concat((p1[0: 1], p2), axis=0), predict_n, new_pre_n))
        else:
            new_pre_n = predict_n
        # update validation confusion matrix
        pre = tuple(map(lambda e: tf.cast(tf.argmax(e, axis=-1), 'int32').numpy(), predict_n))
        pre_2 = tuple(map(lambda e: tf.cast(tf.argmax(e, axis=-1), 'int32').numpy(), predict_2))
        n_pre = tuple(map(lambda e: tf.cast(tf.argmax(e, axis=-1), 'int32').numpy(), new_pre_n))
        gt_ = tuple(e.numpy() for e in gt)
        tr_gt_ = tuple(e.numpy() for e in tr_gt)
        self.cm.record_plus_trans(gt_, pre, n_pre, tr_gt_, pre_2)
        # reserve right or not
        right_old = []
        right_new = []
        for p1, p2, g in zip(pre, n_pre, gt_):
            right_old.append((p1 == g).astype('int32'))
            right_new.append((p2 == g).astype('int32'))
        # update validation metrics
        self.metrics[0].update_state(tf.concat(gt, axis=0), tf.concat(new_pre_n, axis=0))
        cm = tf.convert_to_tensor(self.cm.matrix_multitask)
        for metric in self.metrics[1:]:
            metric.update_state(cm)
        return next_ini, next_o, next_refer, ch_focus, win_focus, right_old, right_new

    def _predict_multitask(self):
        loader = SeqDataLoader(self.manager.data_type, self.data_load_mode, self.data_input_type, self.unit_size,
                               self.file_num)
        loader.start(self.manager.get_predict_set())
        batch_id = 0
        initial_states = None
        last_outputs = None
        refer = None
        attention_log = OrderedDict()
        while not loader.is_exhausted:
            inp, y, tr_y, mask, self.current_ba_frames, no_begin = loader.get_one_batch_data()
            # tackle attention_log
            if len(attention_log) == 0:
                for k in loader.ks:
                    file_name = ntpath.basename(loader.warehouse[k]['file_path'])
                    attention_log[k] = {'file_name': file_name,
                                        'attentions': {'ch': np.array((), dtype=self.model.dtype),
                                                       'win': np.array((), dtype=self.model.dtype),
                                                       'right_old': np.array((), dtype='int32'),
                                                       'right_new': np.array((), dtype='int32')}}
            else:
                for i, flag in enumerate(no_begin):
                    k = loader.ks[i]
                    if flag and attention_log[k] is not None:
                        # archive this channel's file attention
                        self.manager.archive('predicting', attention_log[k], 'attentions')
                        # update
                        if loader.warehouse[k]['id'] != -1:
                            file_name = ntpath.basename(loader.warehouse[k]['file_path'])
                            attention_log[k] = {'file_name': file_name,
                                                'attentions': {'ch': np.array((), dtype=self.model.dtype),
                                                               'win': np.array((), dtype=self.model.dtype),
                                                               'right_old': np.array((), dtype='int32'),
                                                               'right_new': np.array((), dtype='int32')}}
                        else:
                            attention_log[k] = None
            # convert to tensor
            inp = tuple(map(lambda e: tf.constant(e) if e is not None else None, inp))
            y = tuple(tf.constant(y_i, dtype='int32') for y_i in y)
            tr_y = tuple(tf.constant(y_i, dtype='int32') for y_i in tr_y)
            # validation doesn't use sample_weights
            if mask is not None:
                mask = tf.constant(mask, dtype='bool')
            no_begin = tf.constant(no_begin, dtype='bool')
            initial_states, last_outputs, refer, ch_attention, win_attention, right_old, right_new = \
                self._predict_multitask_step(inp, y, tr_y, mask, initial_states, last_outputs, no_begin, refer)
            # tackle next_ini, next_o[, next_refer]
            info = loader.exhausted_info()
            store_update = tf.constant(info, dtype='bool')
            if not tf.reduce_all(store_update):
                initial_states, last_outputs, refer = self.model.reset_ele(store_update, initial_states,
                                                                           last_outputs, refer)
            # tackle attention, may be tuple/list type
            for i, (ca, wa, r_o, r_n) in enumerate(zip(ch_attention, win_attention, right_old, right_new)):
                k = loader.ks[i]
                if ca.shape[0] != 0:  # in other words, attention_log[k] is not None
                    ca = ca.numpy()
                    wa = wa.numpy()
                    old = attention_log[k]['attentions']['ch']
                    attention_log[k]['attentions']['ch'] = np.concatenate((old.reshape((old.shape[0],) + ca.shape[1:]),
                                                                           ca), axis=0)
                    old = attention_log[k]['attentions']['win']
                    attention_log[k]['attentions']['win'] = np.concatenate((old.reshape((old.shape[0],) + wa.shape[1:]),
                                                                            wa), axis=0)
                    old = attention_log[k]['attentions']['right_old']
                    attention_log[k]['attentions']['right'] = np.concatenate((old, r_o), axis=0)
                    old = attention_log[k]['attentions']['right_new']
                    attention_log[k]['attentions']['right'] = np.concatenate((old, r_n), axis=0)
            metrics_log = OrderedDict()
            for k, fn in zip(self.stateful_metrics, self.metrics):
                # why? even though custom metrics classes' 'result' methods have set to return numpy, but invalid
                metrics_log[k] = fn.result().numpy()
            self.callback.on_predict_batch_end(batch_id, {'batch_size': self.current_ba_frames,
                                                          'metrics_log': metrics_log})
            batch_id += 1
        loader.stop()
        for k, v in attention_log.items():
            if v is not None:
                self.manager.archive('predicting', v, 'attentions')
        # archive history
        archive_log = {}
        for k, acc_v in zip(self.model.metrics_na, self.model.history):
            archive_log[k] = acc_v
        self.manager.archive('predicting', archive_log, 'metrics')
        self.callback.on_predict_end(metrics_log)
        self.archive_cm(True)

    def predict(self, model, multitask=False, classifier=True, top_features=False):
        self._register_model(model, classifier)
        if not classifier:    # 无需编译，直接使用数据 ---》 模型，得到保存的最高层特征
            self._no_predict(multitask)
        else:
            if not self.compiled:    # metrics, confusion matrix
                self.compile()
            self.cm.record_ini(multitask)
            self.callback.register_metrics(
                self.stateful_metrics)  # must after register model and before train/test/predict
            self.callback.on_predict_begin({'print_steps': self.print_step})
            if multitask:
                self._predict_multitask()
            else:
                self._predict_normal()


