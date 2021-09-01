# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.metrics import Metric
# from sklearn.metrics import classification_report    call(y_true, y_pre) --- get pre, recall, f1 .etc information
# from sklearn.metrics import f1_score


# the metrics as following are not directly connected with model outputs, so need to code by oneself to use them
# because in tensorflow, metrics should take advantage of the outputs of models directly like losses
# note that a class with abstractmethod can't be instantiated although not a Abstract class if it inherit
# from a Abstract class
class MacroF1(Metric):
    def __init__(self, name, dtype='float32'):
        super(MacroF1, self).__init__(name=name, dtype=dtype)
        self.mf1 = self.add_weight(name='macro_F1',
                                   shape=(),
                                   initializer=tf.keras.initializers.get('zeros'),
                                   dtype=self.dtype)

    # confusion_matrix needs to be a tensor. --- don't specify dtype from other non-tensor
    # d is x-y directional
    def update_state(self, confusion_matrix, d='gt_pre'):
        if d != 'gt_pre':
            confusion_matrix = tf.transpose(confusion_matrix, (1, 0))
        confusion_matrix = tf.cast(confusion_matrix, self.dtype)
        tp = tf.reduce_sum(tf.multiply(confusion_matrix, tf.eye(confusion_matrix.shape[0], dtype=self.dtype)), axis=-1)
        fp = tf.reduce_sum(confusion_matrix, axis=-1) - tp
        fn = tf.reduce_sum(confusion_matrix, axis=0) - tp
        tp_fp_fn = tf.stack((tp, fp, fn), axis=-1)
        denominator = tp + tf.reduce_sum(tp_fp_fn, axis=-1)
        elems = (tp, denominator)
        new = tf.map_fn(lambda x: 2 * x[0] / x[1] if x[1] > 0 else tf.cast(0, self.dtype), elems, dtype=self.dtype)
        self.mf1.assign(tf.reduce_mean(new))

    def result(self):
        return tf.identity(self.mf1)


# mode and n_class only need one
class F1(Metric):
    def __init__(self, name, mode=None, n_class=None, dtype='float32'):
        super(F1, self).__init__(name=name, dtype=dtype)
        if mode is None and n_class is None:
            raise Exception('lack of parameters, \'mode\' and \'n_class\' need one')
        self.n_class = n_class   # valid in case of bigger than one
        self.mode = mode    # 'micro', None
        if self.n_class is not None and mode is None:
            if self.n_class > 1:
                self.f1 = self.add_weight(name='multi_class_f1',
                                          shape=(self.n_class,),
                                          initializer=tf.keras.initializers.get('zeros'),
                                          dtype=self.dtype)
            else:
                self.f1 = self.add_weight(name='single_class_f1',
                                          shape=(),
                                          initializer=tf.keras.initializers.get('zeros'),
                                          dtype=self.dtype)
                #raise ValueError('multi-class-F1 needs the number of classes bigger than one')
        elif self.n_class is None and mode is not None:
            if mode == 'micro':
                self.f1 = self.add_weight(name='micro_f1',
                                          shape=(),
                                          initializer=tf.keras.initializers.get('zeros'),
                                          dtype=self.dtype)
            else:
                raise ValueError('the \'mode\' can\'t be recognized')
        else:
            raise Exception('don\'t indicate two parameter meanwhile')

    def get_config(self):
        config = super(F1, self).get_config()
        config.update({'n_class': self.n_class, 'mode': self.mode})
        return config

    def update_state(self, confusion_matrix, d='gt_pre'):
        if d != 'gt_pre':
            confusion_matrix = tf.transpose(confusion_matrix, (1, 0))
        confusion_matrix = tf.cast(confusion_matrix, self.dtype)
        tp = tf.reduce_sum(tf.multiply(confusion_matrix, tf.eye(len(confusion_matrix), dtype=self.dtype)), axis=-1)
        fp = tf.reduce_sum(confusion_matrix, axis=-1) - tp
        fn = tf.reduce_sum(confusion_matrix, axis=0) - tp
        tp_fp_fn = tf.stack((tp, fp, fn), axis=-1)
        if self.mode is None:
            if confusion_matrix.shape[0] != self.n_class:
                raise ValueError('confusion-matrix passed doesn\'t match in terms of the number of classes')
            denominator = tp + tf.reduce_sum(tp_fp_fn, axis=-1)
            elems = (tp, denominator)
            new = tf.map_fn(lambda x: 2 * x[0] / x[1] if x[1] > 0 else tf.cast(0, self.dtype), elems, dtype=self.dtype)
        else:
            tp_fp_fn = tf.reduce_sum(tp_fp_fn, axis=0)
            denominator = tp_fp_fn[0] + tf.reduce_sum(tp_fp_fn)
            if denominator > 0:
                new = 2 * tp_fp_fn[0] / denominator
            else:
                new = tf.cast(0, self.dtype)
        self.f1.assign(new)

    def reset_states(self):
        v = tf.zeros_like(self.f1, dtype=self.dtype)
        self.f1.assign(v)

    def result(self):
        return tf.identity(self.f1)


# mode and n_class only need one
# kind in ('recall', 'precision', 'npv', 'spec')
class Rate(Metric):
    def __init__(self, name, kind, mode=None, n_class=None, dtype='float32'):
        super(Rate, self).__init__(name=name, dtype=dtype)
        if mode is None and n_class is None:
            raise Exception('lack of parameters, \'mode\' and \'n_class\' need one')
        if kind not in ('recall', 'precision', 'npv', 'spec'):
            raise ValueError('the \'kind\' can\'t be recognized')
        self.n_class = n_class   # valid in case of bigger than one
        self.mode = mode    # 'overall', None
        self.kind = kind
        if self.n_class is not None and mode is None:
            if self.n_class > 1:
                self.rate = self.add_weight(name='multi_class_' + self.kind + '_rate',
                                            shape=(self.n_class,),
                                            initializer=tf.keras.initializers.get('zeros'),
                                            dtype=self.dtype)
            else:
                raise ValueError('multi-class-{} needs the number of classes bigger than one'.format(self.kind))
        elif self.n_class is None and mode is not None:
            if mode == 'overall':
                self.rate = self.add_weight(name='overall_' + self.kind + '_rate',
                                            shape=(),
                                            initializer=tf.keras.initializers.get('zeros'),
                                            dtype=self.dtype)
            else:
                raise ValueError('the \'mode\' can\'t be recognized')
        else:
            raise Exception('don\'t indicate two parameter meanwhile')

    # the first element of two_elem's one piece is division child
    def update_state(self, confusion_matrix, d='gt_pre'):
        if self.n_class is not None and self.n_class != confusion_matrix.shape[0]:
            raise ValueError('confusion-matrix passed doesn\'t match in terms of the number of classes')
        if d != 'gt_pre':
            confusion_matrix = tf.transpose(confusion_matrix, (1, 0))
        confusion_matrix = tf.cast(confusion_matrix, self.dtype)
        confusion_matrix = tf.cast(confusion_matrix, self.dtype)
        tp = tf.reduce_sum(tf.multiply(confusion_matrix, tf.eye(len(confusion_matrix), dtype=self.dtype)), axis=-1)
        fp = tf.reduce_sum(confusion_matrix, axis=-1) - tp
        fn = tf.reduce_sum(confusion_matrix, axis=0) - tp
        tn = tf.TensorArray(self.dtype, size=confusion_matrix.shape[0])
        for idx in tf.range(confusion_matrix.shape[0]):
            temp = tf.reduce_sum(confusion_matrix) - fp[idx] - fn[idx] - tp[idx]
            tn = tn.write(idx, temp)
        tn = tn.stack()
        if self.kind == 'recall':
            elems = (tp, tp + fn)
        elif self.kind == 'precision':
            elems = (tp, tp + fp)
        elif self.kind == 'npv':
            elems = (tn, tn + fn)
        else:
            elems = (tn, tn + fp)
        new = tf.map_fn(lambda x: x[0] / x[1] if x[1] > 0 else tf.cast(0, self.dtype), elems, dtype=self.dtype)
        if self.mode is not None:
            new = tf.reduce_mean(new)
        self.rate.assign(new)

    def get_config(self):
        config = super(Rate, self).get_config()
        config.update({'n_class': self.n_class, 'mode': self.mode, 'kind': self.kind})
        return config

    def reset_states(self):
        v = tf.zeros_like(self.rate, dtype=self.dtype)
        self.rate.assign(v)

    def result(self):
        return tf.identity(self.rate)


# used to make consistency check, also to be a symbol for classification performance
# values in interval [-1, 1], usually greater than 0
class Kappa(Metric):
    def __init__(self, name, dtype='float32'):
        super(Kappa, self).__init__(name=name, dtype=dtype)
        self.p_o = self.add_weight(name='overall_accuracy',
                                   shape=(),
                                   initializer=tf.keras.initializers.get('zeros'),
                                   dtype=self.dtype)
        self.p_e = self.add_weight(name='penalty_equilibrium',
                                   shape=(),
                                   initializer=tf.keras.initializers.get('zeros'),
                                   dtype=self.dtype)

    # confusion matrix is passed in need
    def update_state(self, cm):
        cm = tf.cast(cm, self.dtype)
        self.p_o.assign(tf.linalg.trace(cm) / tf.reduce_sum(cm))
        col_h = tf.reduce_sum(cm, axis=0)
        row_h = tf.reduce_sum(cm, axis=-1)
        self.p_e.assign(tf.reduce_sum(tf.multiply(row_h, col_h)) / tf.pow(tf.reduce_sum(cm), 2))

    def result(self):
        kappa = (self.p_o - self.p_e) / (1. - self.p_e)
        return kappa


