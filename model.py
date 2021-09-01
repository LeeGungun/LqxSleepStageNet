# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import nn
import gc


class TimeConv(keras.Model):
    def __init__(self, name, config):
        super(TimeConv, self).__init__(name=name, dtype=config.dtype)
        self.config = config
        self.data_format = config.data_format
        self.pattern_conv = nn.PatternConv('pattern_convolution', self.config.seed, self.config.pattern_layers,
                                           self.config.pattern_filters, self.config.pattern_stride,
                                           self.config.pattern_act, self.config.pattern_bias,
                                           self.config.pattern_padding, self.config.pattern_dropout,
                                           self.config.pattern_max_pool, self.config.pattern_wd,
                                           self.config.pattern_he_scale, self.data_format, self.dtype)
        self.depth_conv = list(nn.DepthConv('No.{}_depth_convolution'.format(i + 1), config.seed,
                                            config.time_dc_dfilters[i], config.time_dc_pfilter_num[i],
                                            config.time_dc_strides[i], config.time_dc_act[i], config.time_dc_bias[i],
                                            config.wd, config.time_dc_pd[i], self.data_format,
                                            self.dtype, config.time_dc_act_filter[i], config.time_dc_he_scale[i])
                               for i in range(len(config.time_dc_dfilters)))
        self.var_se = nn.VarianceSE('variance_squeeze_excitation', self.config.se_trans_chs, self.config.seed,
                                    self.config.se_he_scale, self.config.se_bias, self.config.se_wd, self.dtype)
        self.regulate_layers = None

    def regu(self):
        if self.regulate_layers is None:
            regulate_layers = []
            for l in self.layers:
                if hasattr(l, 'regu'):
                    regulate_layers.append(l)
            self.regulate_layers = regulate_layers
        r = []
        for l in self.regulate_layers:
           r += l.regu()
        return r

    def get_config(self):
        config = {'name': self.name, 'config': self.config}
        return config

    @classmethod
    def from_config(cls, kwargs):
        return cls(**kwargs)

    def get_input(self):
        prod = self.config.epoch_second * self.config.fs
        inputs = tf.zeros((self.config.batch_size, self.config.chs, prod, 1), dtype=self.dtype)
        return inputs, self.config.batch_size    # need to append 'training'

    # inputs --- T  --- (epochs, chs, epoch_second * fs, 1)
    def call(self, inputs, training=None, heap=None):
        inputs = tf.cast(inputs, self.dtype)
        chs_axis = 3 - self.data_format.find('H')
        ori_chs = inputs.shape[chs_axis]
        inputs = self.pattern_conv(inputs, training, heap, self.config.relu_leaky, self.config.relu_threshold)
        # output: (epochs, chs * pfilter_depth_prod, h', 3 * 1)
        inputs = tf.concat(inputs, axis=-1)
        for dc in self.depth_conv:
            inputs = dc(inputs, training, heap, relu=self.config.relu_leaky, threshold=self.config.relu_threshold)
        # output: (epochs, wins, chs * pfilter_depth_prod, h' // wins, 3 * 1)
        split_into_wins = tf.stack(tf.split(inputs, self.config.wins, axis=(3 - chs_axis)), axis=1)
        # output: (epochs, wins, chs, pfilter_depth_prod, h' // wins, 3 * 1)
        split_into_wins = tf.stack(tf.split(split_into_wins, ori_chs, axis=(chs_axis + 1)), axis=2)
        del inputs
        gc.collect()

        # track out pattern: 3 patterns integrate into one
        temp = tf.transpose(tf.reduce_mean(split_into_wins, axis=(5 - chs_axis)), (4, 0, 2, 3, 1))
        # (epochs, chs, 3)
        var = tf.transpose(self.var_se(temp, self.config.relu_leaky, self.config.relu_threshold, heap), (1, 2, 0))
        var = tf.reshape(var, (var.shape[0], 1, var.shape[1], 1, var.shape[2]))
        split_into_wins = tf.reduce_max(split_into_wins, axis=(5 - chs_axis))
        # output shape: (epochs. wins, chs, pfilter_depth_prod, 1)
        split_into_wins = tf.reduce_sum(tf.multiply(split_into_wins, var), axis=-1, keepdims=True)

        return split_into_wins


class STimeConv(keras.Model):
    def __init__(self, name, config):
        super(STimeConv, self).__init__(name=name, dtype=config.dtype)
        self.config = config
        self.data_format = config.data_format
        self.depth_conv = list(nn.DepthConv('No.{}_depth_convolution'.format(i + 1), config.seed,
                                            config.time_dc_dfilters[i], config.time_dc_pfilter_num[i],
                                            config.time_dc_strides[i], config.time_dc_act[i], config.time_dc_bias[i],
                                            config.wd, config.time_dc_pd[i], self.data_format,
                                            self.dtype, config.time_dc_act_filter[i], config.time_dc_he_scale[i])
                               for i in range(len(config.time_dc_dfilters)))
        self.insert_max = list(nn.MaxPool('No.{}_max_pooling'.format(i + 1), e[0], e[1], e[2], self.data_format)
                               for i, e in enumerate(config.time_dc_insert_max_info))
        self.regulate_layers = None

    def regu(self):
        if self.regulate_layers is None:
            regulate_layers = []
            for l in self.layers:
                if hasattr(l, 'regu'):
                    regulate_layers.append(l)
            self.regulate_layers = regulate_layers
        r = []
        for l in self.regulate_layers:
           r += l.regu()
        return r

    def get_config(self):
        config = {'name': self.name, 'config': self.config}
        return config

    @classmethod
    def from_config(cls, kwargs):
        return cls(**kwargs)

    def get_input(self):
        prod = self.config.epoch_second * self.config.fs
        inputs = tf.zeros((self.config.batch_size, self.config.chs, prod, 1), dtype=self.dtype)
        return inputs, self.config.batch_size    # need to append 'training'

    # inputs --- T  --- (epochs, chs, epoch_second * fs, 1)
    def call(self, inputs, training=None, heap=None):
        inputs = tf.cast(inputs, self.dtype)
        chs_axis = 3 - self.data_format.find('H')
        ori_chs = inputs.shape[chs_axis]
        #inputs = self.pattern_conv(inputs, training, heap, self.config.relu_leaky, self.config.relu_threshold)
        # output: (epochs, chs * pfilter_depth_prod, h', 3 * 1)
        #inputs = tf.concat(inputs, axis=-1)
        max_flag = 0
        for dc, flag, d in zip(self.depth_conv, self.config.time_dc_insert_max, self.config.time_dc_dropout):
            inputs = dc(inputs, training, heap, relu=self.config.relu_leaky, threshold=self.config.relu_threshold)
            if flag == 1:
                inputs = self.insert_max[max_flag](inputs, heap)
                max_flag += 1
            if training and d:
                inputs = tf.nn.dropout(inputs, d, noise_shape=[1, None, None, 1])
        # output: (epochs, wins, chs * pfilter_depth_prod, h' // wins, 1)
        split_into_wins = tf.stack(tf.split(inputs, self.config.wins, axis=(3 - chs_axis)), axis=1)
        # output: (epochs, wins, chs, pfilter_depth_prod, h' // wins, 1)
        split_into_wins = tf.stack(tf.split(split_into_wins, ori_chs, axis=(chs_axis + 1)), axis=2)
        del inputs
        gc.collect()

        # output shape: (epochs. wins, chs, pfilter_depth_prod, 1)
        return tf.reduce_max(split_into_wins, axis=(5 - chs_axis))


class SVDTimeConv(keras.Model):
    def __init__(self, name, config):
        super(SVDTimeConv, self).__init__(name=name, dtype=config.dtype)
        self.config = config
        self.data_format = config.data_format
        self.pattern_conv = nn.PatternConv('pattern_convolution', self.config.seed, self.config.pattern_layers,
                                           self.config.pattern_filters, self.config.pattern_stride,
                                           self.config.pattern_act, self.config.pattern_bias,
                                           self.config.pattern_padding, self.config.pattern_dropout,
                                           self.config.pattern_max_pool, self.config.pattern_wd,
                                           self.config.pattern_he_scale, self.data_format, self.dtype)
        self.depth_conv = list(nn.DepthConv('No.{}_depth_convolution'.format(i + 1), config.seed,
                                            config.time_dc_dfilters[i], config.time_dc_pfilter_num[i],
                                            config.time_dc_strides[i], config.time_dc_act[i], config.time_dc_bias[i],
                                            config.wd, config.time_dc_pd[i], self.data_format,
                                            self.dtype, config.time_dc_act_filter[i], config.time_dc_he_scale[i])
                               for i in range(len(config.time_dc_dfilters)))
        self.svd_se = nn.SVDSE('svd_squeeze_excitation', self.config.seed, self.config.se_he_scale,
                               self.config.se_bias, self.config.se_wd, self.dtype)
        self.regulate_layers = None

    def regu(self):
        if self.regulate_layers is None:
            regulate_layers = []
            for l in self.layers:
                if hasattr(l, 'regu'):
                    regulate_layers.append(l)
            self.regulate_layers = regulate_layers
        r = []
        for l in self.regulate_layers:
           r += l.regu()
        return r

    def get_config(self):
        config = {'name': self.name, 'config': self.config}
        return config

    @classmethod
    def from_config(cls, kwargs):
        return cls(**kwargs)

    def get_input(self):
        prod = self.config.epoch_second * self.config.fs
        inputs = tf.zeros((self.config.batch_size, self.config.chs, prod, 1), dtype=self.dtype)
        return inputs, self.config.batch_size    # need to append 'training'

    # inputs --- T  --- (epochs, chs, epoch_second * fs, 1)
    def call(self, inputs, training=None, heap=None):
        inputs = tf.cast(inputs, self.dtype)
        chs_axis = 3 - self.data_format.find('H')
        ori_chs = inputs.shape[chs_axis]
        # outputs type --- tuple, the element keeps the original shape
        inputs = self.pattern_conv(inputs, training, heap, self.config.relu_leaky, self.config.relu_threshold)
        # output: (epochs, chs * pfilter_depth_prod, h', 3 * 1)  have no connection with data format
        inputs = tf.concat(inputs, axis=-1)
        for dc in self.depth_conv:
            inputs = dc(inputs, training, heap, relu=self.config.relu_leaky, threshold=self.config.relu_threshold)
        # output: (epochs, wins, chs * pfilter_depth_prod, h' // wins, 3 * 1)
        split_into_wins = tf.stack(tf.split(inputs, self.config.wins, axis=(3 - chs_axis)), axis=1)
        # output: (epochs, wins, chs, pfilter_depth_prod, h' // wins, 3 * 1)
        split_into_wins = tf.stack(tf.split(split_into_wins, ori_chs, axis=(chs_axis + 1)), axis=2)
        del inputs
        gc.collect()

        # track out pattern: 3 patterns integrate into one
        temp = tf.reduce_mean(split_into_wins, axis=(5 - chs_axis))
        # (epochs, wins, chs, pfilter_depth_prod, 3 * 1)
        svd = self.svd_se(temp, self.config.relu_leaky, self.config.relu_threshold, heap)
        split_into_wins = tf.reduce_max(split_into_wins, axis=(5 - chs_axis))
        # output shape: (epochs. wins, chs, pfilter_depth_prod, 1)
        split_into_wins = tf.reduce_sum(tf.multiply(split_into_wins, svd), axis=-1, keepdims=True)

        return split_into_wins


class NTimeConv(keras.Model):
    def __init__(self, name, config):
        super(NTimeConv, self).__init__(name=name, dtype=config.dtype)
        self.config = config
        self.data_format = config.data_format
        self.pattern_conv = nn.PatternConv('pattern_convolution', self.config.seed, self.config.pattern_layers,
                                           self.config.pattern_filters, self.config.pattern_stride,
                                           self.config.pattern_act, self.config.pattern_bias,
                                           self.config.pattern_padding, self.config.pattern_dropout,
                                           self.config.pattern_max_pool, self.config.pattern_wd,
                                           self.config.pattern_he_scale, self.data_format, self.dtype)
        self.depth_conv = list(nn.DepthConv('No.{}_depth_convolution'.format(i + 1), config.seed,
                                            config.time_dc_dfilters[i], config.time_dc_pfilter_num[i],
                                            config.time_dc_strides[i], config.time_dc_act[i], config.time_dc_bias[i],
                                            config.wd, config.time_dc_pd[i], self.data_format,
                                            self.dtype, config.time_dc_act_filter[i], config.time_dc_he_scale[i])
                               for i in range(len(config.time_dc_dfilters)))
        self.regulate_layers = None

    def regu(self):
        if self.regulate_layers is None:
            regulate_layers = []
            for l in self.layers:
                if hasattr(l, 'regu'):
                    regulate_layers.append(l)
            self.regulate_layers = regulate_layers
        r = []
        for l in self.regulate_layers:
           r += l.regu()
        return r

    def get_config(self):
        config = {'name': self.name, 'config': self.config}
        return config

    @classmethod
    def from_config(cls, kwargs):
        return cls(**kwargs)

    def get_input(self):
        prod = self.config.epoch_second * self.config.fs
        inputs = tf.zeros((self.config.batch_size, self.config.chs, prod, 1), dtype=self.dtype)
        return inputs, self.config.batch_size    # need to append 'training'

    # inputs --- T  --- (epochs, chs, epoch_second * fs, 1)
    def call(self, inputs, training=None, heap=None):
        inputs = tf.cast(inputs, self.dtype)
        chs_axis = 3 - self.data_format.find('H')
        ori_chs = inputs.shape[chs_axis]
        # outputs type --- tuple, the element keeps the original shape
        inputs = self.pattern_conv(inputs, training, heap, self.config.relu_leaky, self.config.relu_threshold)
        # output: (epochs, chs * pfilter_depth_prod, h', 3 * 1)  have no connection with data format
        inputs = tf.concat(inputs, axis=-1)
        for dc in self.depth_conv:
            inputs = dc(inputs, training, heap, relu=self.config.relu_leaky, threshold=self.config.relu_threshold)
        # output: (epochs, wins, chs * pfilter_depth_prod, h' // wins, 3 * 1)
        split_into_wins = tf.stack(tf.split(inputs, self.config.wins, axis=(3 - chs_axis)), axis=1)
        # output: (epochs, wins, chs, pfilter_depth_prod, h' // wins, 3 * 1)
        split_into_wins = tf.stack(tf.split(split_into_wins, ori_chs, axis=(chs_axis + 1)), axis=2)
        del inputs
        gc.collect()

        # track out pattern: 3 patterns integrate into one
        temp = tf.reduce_max(split_into_wins, axis=(5 - chs_axis))
        # output shape: (epochs. wins, chs, 3*pfilter_depth_prod, 1)
        split_into_wins = tf.concat(tf.split(temp, 3, axis=-1), axis=-2)

        return split_into_wins


class ABiGRUBaseForStackBi(keras.Model):
    def __init__(self, name, config):
        super(ABiGRUBaseForStackBi, self).__init__(name=name, dtype=config.dtype)
        self.config = config
        self.data_format = self.config.data_format
        self.has_dynamics = 'df' in config.inputs
        if 't' in config.inputs:
            self.select_t = TimeConv('time_conv_net', self.config)
            self.dropout_t = keras.layers.Dropout(self.config.dropout_t, noise_shape=(1, None, None, None, None),
                                                  name='time_dropout_before_attention', dtype=self.dtype)
        else:
            self.select_t = None
            self.dropout_t = None
        if 'f' in config.inputs:
            self.select_fre = nn.SelectF('select_fre', self.config.f_low_num, self.config.f_mid_num,
                                         self.config.select_f, self.config.seed, self.has_dynamics, self.config.f_wd,
                                         self.config.f_he_scale, self.dtype, self.data_format)
            self.dropout_fre = keras.layers.Dropout(self.config.dropout_f, noise_shape=(1, None, None, None, None),
                                                    name='fre_dropout_before_attention', dtype=self.dtype)
        else:
            self.select_fre = None
            self.dropout_fre = None
        self.disassemble = nn.Disassemble('disassemble', self.config.seq_len, self.config.seq_batch_size,
                                          self.config.padding_mode)
        self.stack_bi_gru = nn.StackBiGRUPlusAttention('stack_bi_gru', self.config.seq_len, self.config.seed,
                                                       self.config.rnn_h_units, self.config.focus_units,
                                                       self.config.rnn_mmode, self.config.rnn_layers,
                                                       self.config.focus_bias, self.config.focus_wd, self.dtype,
                                                       self.config.rnn_h_wd, self.config.rnn_input_dropout,
                                                       self.config.rnn_hidden_dropout)
        self.classification = nn.Classification('classification', self.config.classes, self.config.seed,
                                                self.config.classifier_dr_rate, self.dtype, self.config.classifier_bias,
                                                self.config.wd)
        self.regulate_layers = None

    def regu(self):
        if self.regulate_layers is None:
            regulate_layers = []
            for l in self.layers:
                if hasattr(l, 'regu'):
                    regulate_layers.append(l)
            self.regulate_layers = regulate_layers
        r = []
        for l in self.regulate_layers:
           r += l.regu()
        return r

    def get_config(self):
        config = {'name': self.name, 'config': self.config}
        return config

    @classmethod
    def from_config(cls, kwargs):
        return cls(**kwargs)

    def get_input(self):
        if 't' in self.config.inputs:
            t = tf.zeros((self.config.batch_size, self.config.chs, self.config.epoch_second * self.config.fs),
                         dtype=self.dtype)
        else:
            t = None
        if 'f' in self.config.inputs:
            f = tf.zeros((self.config.batch_size, self.config.chs, self.config.wins, self.config.fft), dtype=self.dtype)
        else:
            f = None
        if self.has_dynamics:
            df = tf.zeros((self.config.batch_size, self.config.chs, self.config.wins, self.config.fft, 2),
                          dtype=self.dtype)
        else:
            df = None
        inputs = (t, f, df)
        initial_states = None
        last_output = None
        no_begin = tf.constant([True], dtype='bool')
        return inputs, initial_states, last_output, no_begin   # when use please append 'training' & 'classifier'

    # inputs --- (T, F) or (T, F, DF)     per fre element --- (epochs, chs, wins, points)
    #                                             time element --- (epochs, chs, epoch_sampling_points)
    # initial_states --- (((1, h_units, f) or None, None) or None) * layers or None
    # last_outputs --- ((1, bottom_h_units, f) or None, None) or None
    # no_begin --- None or 1-D Tensor (bool) True indicates needs to use self-attention
    def call(self, inputs, initial_states=None, last_outputs=None, no_begin=None, training=None,
             classifier=True):
        if self.config.heap is not None:
            time_heap = self.config.heap
            fre_heap = int(self.config.heap * self.config.wins)
        else:
            time_heap = None
            fre_heap = None

        if inputs[0] is not None and self.select_t is not None:
            t = tf.expand_dims(inputs[0], axis=-1)
            t = self.select_t(t, training, time_heap)
            t = self.dropout_t(t, training)
        else:
            t = None
        if inputs[1] is not None and self.select_fre is not None:
            epochs = inputs[1].shape[0]
            f = tf.expand_dims(inputs[1], axis=-1)
            if inputs[2] is not None and self.has_dynamics:
                f = tf.concat([f, inputs[2]], axis=-1)
            del inputs
            gc.collect()
            f = tf.transpose(f, (0, 2, 1, 3, 4))
            f = tf.concat(tf.unstack(f, f.shape[0], axis=0), axis=0)
            f = self.select_fre(f, training, fre_heap)
            f = tf.stack(tf.split(f, epochs, axis=0), axis=0)
            f = self.dropout_fre(f, training)
        else:
            del inputs
            gc.collect()
            f = None
        if t is not None and f is not None:
            all_in = tf.concat((t, f), axis=-1)
        elif t is not None:
            all_in = t
        elif f is not None:
            all_in = f
        del t, f
        gc.collect()
        _f = all_in.shape[-1]
        # (seqs, steps, 1, wins, chs, xxx, _f)  mask output shape: (seqs, steps, 1)
        all_in, mask = self.disassemble(all_in)
        ini_s = initial_states
        last_o = last_outputs
        seqs = len(all_in)
        # the last seq may have different steps
        outputs = tf.TensorArray(self.dtype, size=seqs, infer_shape=False)
        ch_attention = tf.TensorArray(self.dtype, size=seqs, infer_shape=False)
        win_attention = tf.TensorArray(self.dtype, size=seqs, infer_shape=False)
        exp_no_begin = (no_begin,) + (None,) * (seqs - 1)  # can't use tf.range
        if classifier:
            for i in range(seqs):
                if mask is not None:
                    temp_m = mask[i]
                else:
                    temp_m = None
                gru_o = self.stack_bi_gru(all_in[i], ini_s, last_o, temp_m, exp_no_begin[i], training,
                                          self.config.relu_leaky, self.config.relu_threshold)
                result = self.classification(tf.concat(tf.unstack(gru_o[0], _f, axis=-1), axis=-1), temp_m, training)
                outputs = outputs.write(i, result)
                ch_attention = ch_attention.write(i, gru_o[3])
                win_attention = win_attention.write(i, gru_o[4])
                ini_s = tuple((s_i[0], None) for s_i in gru_o[1])
                last_o = (gru_o[2][0], None)  # bottom layer
        else:
            for i in range(seqs):
                if mask is not None:
                    temp_m = mask[i]
                else:
                    temp_m = None
                gru_o = self.stack_bi_gru(all_in[i], ini_s, last_o, temp_m, exp_no_begin[i], training,
                                          self.config.relu_leaky, self.config.relu_threshold)
                result = tf.concat(tf.unstack(gru_o[0], _f, axis=-1), axis=-1)
                outputs = outputs.write(i, result)
                ch_attention = ch_attention.write(i, gru_o[3])
                win_attention = win_attention.write(i, gru_o[4])
                ini_s = tuple((s_i[0], None) for s_i in gru_o[1])
                last_o = (gru_o[2][0], None)  # bottom layer

        next_ini = ini_s
        next_last = last_o
        outputs = outputs.concat()
        # (batch, n_class/top_units)
        outputs = tf.reshape(outputs, (-1, outputs.shape[-1]))
        ch_attention = tf.concat(tuple(ch_attention.read(i) for i in range(seqs)), axis=1)
        ch_attention = tf.transpose(ch_attention, (1, 2, 0, 4, 3))
        # (batch, 2, f, chs)
        ch_attention = tf.reshape(ch_attention, tf.concat([tf.constant((-1,), dtype='int32'),
                                                           tf.constant(ch_attention.shape[-3:])], axis=0))
        win_attention = tf.concat(tuple(win_attention.read(i) for i in range(seqs)), axis=1)
        win_attention = tf.transpose(win_attention, (1, 2, 0, 5, 3, 4))
        # (batch, 2, f, wins, chs)
        win_attention = tf.reshape(win_attention, tf.concat([tf.constant((-1,), dtype='int32'),
                                                            tf.constant(win_attention.shape[-4:])], axis=0))
        # 已经全部使用mask对齐过结果
        # output --- (epochs, n_class/top_units)    ch_att --- (epochs, 2, 1/3, chs)
        # win_att --- (epochs, 2, 1/3, wins, chs)
        # ini_s returned and last_o returned need tackling furthermore (should reset zeros or None when new_file starts)
        return [outputs], next_ini, next_last, [ch_attention], [win_attention]

    def reset_ele(self, ch_info, ini, o, refer=None):
        ch_info = tf.cast(ch_info, self.dtype)
        exp_dims = o[0].shape.ndims - ch_info.shape.ndims
        exp_ch_info = tf.reshape(ch_info, ch_info.shape + (1,) * exp_dims)
        new_o = (tf.multiply(o[0], exp_ch_info), None)
        new_ini = tuple((tf.multiply(e[0], exp_ch_info), None) for e in ini)
        if refer is not None:
            new_refer = tf.multiply(refer, tf.reshape(ch_info, ch_info.shape + (1,) * (refer.shape.ndims -
                                                                                       ch_info.shape.ndims)))
        else:
            new_refer = None
        return new_ini, new_o, new_refer


class ABiGRUBaseForBiWrapper(keras.Model):
    def __init__(self, name, config):
        super(ABiGRUBaseForBiWrapper, self).__init__(name=name, dtype=config.dtype)
        self.config = config
        self.data_format = self.config.data_format
        self.has_dynamics = 'df' in config.inputs
        if 't' in config.inputs:
            self.select_t = TimeConv('time_conv_net', self.config)
            self.dropout_t = keras.layers.Dropout(self.config.dropout_t, noise_shape=(1, None, None, None, None),
                                                  name='time_dropout_before_attention', dtype=self.dtype)
        else:
            self.select_t = None
            self.dropout_t = None
        if 'f' in config.inputs:
            self.select_fre = nn.SelectF('select_fre', self.config.f_low_num, self.config.f_mid_num,
                                         self.config.select_f, self.config.seed, self.has_dynamics, self.config.f_wd,
                                         self.config.f_he_scale, self.dtype, self.data_format)
            self.dropout_fre = keras.layers.Dropout(self.config.dropout_f, noise_shape=(1, None, None, None, None),
                                                    name='fre_dropout_before_attention', dtype=self.dtype)
        else:
            self.select_fre = None
            self.dropout_fre = None
        self.disassemble = nn.Disassemble('disassemble', self.config.seq_len, self.config.seq_batch_size,
                                          self.config.padding_mode)
        self.bi_wrapper = nn.BidirectionalWrapper('bi_wrapper_for_gru_plus_attention', self.config.seq_len,
                                                  self.config.seed, self.config.rnn_h_units, self.config.focus_units,
                                                  self.config.rnn_layers, self.config.bi_wrapper_mmode,
                                                  self.config.focus_bias, self.config.focus_wd, self.dtype,
                                                  self.config.rnn_h_wd, self.config.rnn_input_dropout,
                                                  self.config.rnn_hidden_dropout)
        self.classification = nn.Classification('classification', self.config.classes, self.config.seed,
                                                self.config.classifier_dr_rate, self.dtype, self.config.classifier_bias,
                                                self.config.wd)
        self.regulate_layers = None

    def regu(self):
        if self.regulate_layers is None:
            regulate_layers = []
            for l in self.layers:
                if hasattr(l, 'regu'):
                    regulate_layers.append(l)
            self.regulate_layers = regulate_layers
        r = []
        for l in self.regulate_layers:
           r += l.regu()
        return r

    def get_config(self):
        config = {'name': self.name, 'config': self.config}
        return config

    @classmethod
    def from_config(cls, kwargs):
        return cls(**kwargs)

    def get_input(self):
        if 't' in self.config.inputs:
            t = tf.zeros((self.config.batch_size, self.config.chs, self.config.epoch_second * self.config.fs),
                         dtype=self.dtype)
        else:
            t = None
        if 'f' in self.config.inputs:
            f = tf.zeros((self.config.batch_size, self.config.chs, self.config.wins, self.config.fft), dtype=self.dtype)
        else:
            f = None
        if self.has_dynamics:
            df = tf.zeros((self.config.batch_size, self.config.chs, self.config.wins, self.config.fft, 2),
                          dtype=self.dtype)
        else:
            df = None
        inputs = (t, f, df)
        initial_states = None
        last_output = None
        no_begin = tf.constant([True], dtype='bool')
        return inputs, initial_states, last_output, no_begin   # when use please append 'training' & 'classifier'

    # inputs --- (T, F) or (T, F, DF)     per fre element --- (epochs, chs, wins, points)
    #                                             time element --- (epochs, chs, epoch_sampling_points)
    # initial_states --- ((((1, h_units, f) or None) * layers or None), None) or None
    # last_outputs --- ((1, top_h_units, f) or None, None) or None
    # no_begin --- None or 1-D Tensor (bool) True indicates needs to use self-attention
    def call(self, inputs, initial_states=None, last_outputs=None, no_begin=None, training=None,
             classifier=True):
        if self.config.heap is not None:
            time_heap = self.config.heap
            fre_heap = int(self.config.heap * self.config.wins)
        else:
            time_heap = None
            fre_heap = None

        if inputs[0] is not None and self.select_t is not None:
            t = tf.expand_dims(inputs[0], axis=-1)
            t = self.select_t(t, training, time_heap)
            t = self.dropout_t(t, training)
        else:
            t = None
        if inputs[1] is not None and self.select_fre is not None:
            epochs = inputs[1].shape[0]
            f = tf.expand_dims(inputs[1], axis=-1)
            if inputs[2] is not None and self.has_dynamics:
                f = tf.concat([f, inputs[2]], axis=-1)
            del inputs
            gc.collect()
            f = tf.transpose(f, (0, 2, 1, 3, 4))
            f = tf.concat(tf.unstack(f, f.shape[0], axis=0), axis=0)
            f = self.select_fre(f, training, fre_heap)
            f = tf.stack(tf.split(f, epochs, axis=0), axis=0)
            f = self.dropout_fre(f, training)
        else:
            del inputs
            gc.collect()
            f = None
        if t is not None and f is not None:
            all_in = tf.concat((t, f), axis=-1)
        elif t is not None:
            all_in = t
        elif f is not None:
            all_in = f
        del t, f
        gc.collect()
        _f = all_in.shape[-1]
        # (seqs, steps, 1, wins, chs, xxx, _f)  mask output shape: (seqs, steps, 1)
        all_in, mask = self.disassemble(all_in)
        ini_s = initial_states
        last_o = last_outputs
        seqs = len(all_in)
        # the last seq may have different steps
        outputs = tf.TensorArray(self.dtype, size=seqs, infer_shape=False)
        ch_attention = tf.TensorArray(self.dtype, size=seqs, infer_shape=False)
        win_attention = tf.TensorArray(self.dtype, size=seqs, infer_shape=False)
        exp_no_begin = (no_begin,) + (None,) * (seqs - 1)  # can't use tf.range
        if classifier:
            for i in range(seqs):
                if mask is not None:
                    temp_m = mask[i]
                else:
                    temp_m = None
                gru_o = self.bi_wrapper(all_in[i], ini_s, last_o, temp_m, exp_no_begin[i], training,
                                        self.config.relu_leaky, self.config.relu_threshold)
                result = self.classification(tf.concat(tf.unstack(gru_o[0], _f, axis=-1), axis=-1), temp_m, training)
                outputs = outputs.write(i, result)
                ch_attention = ch_attention.write(i, gru_o[3])
                win_attention = win_attention.write(i, gru_o[4])
                ini_s = (gru_o[1][0], None)
                last_o = (gru_o[2][0], None)  # top layer
        else:
            for i in range(seqs):
                if mask is not None:
                    temp_m = mask[i]
                else:
                    temp_m = None
                gru_o = self.bi_wrapper(all_in[i], ini_s, last_o, temp_m, exp_no_begin[i], training,
                                        self.config.relu_leaky, self.config.relu_threshold)
                result = tf.concat(tf.unstack(gru_o[0], _f, axis=-1), axis=-1)
                outputs = outputs.write(i, result)
                ch_attention = ch_attention.write(i, gru_o[3])
                win_attention = win_attention.write(i, gru_o[4])
                ini_s = (gru_o[1][0], None)
                last_o = (gru_o[2][0], None)  # top layer
        next_ini = ini_s
        next_last = last_o
        outputs = outputs.concat()
        # (batch, n_class/top_units)
        outputs = tf.reshape(outputs, (-1, outputs.shape[-1]))
        ch_attention = tf.transpose(ch_attention.concat(), (0, 1, 2, 4, 3))
        # (batch, 2, f, chs)
        ch_attention = tf.reshape(ch_attention, tf.concat([tf.constant((-1,), dtype='int32'),
                                                           tf.constant(ch_attention.shape[-3:])], axis=0))
        win_attention = tf.transpose(win_attention.concat(), (0, 1, 2, 5, 3, 4))
        # (batch, 2, f, wins, chs)
        win_attention = tf.reshape(win_attention, tf.concat([tf.constant((-1,), dtype='int32'),
                                                            tf.constant(win_attention.shape[-4:])], axis=0))
        # 已经全部使用mask对齐过结果
        # output --- (epochs, n_class/top_units)    ch_att --- (epochs, 2, 1/3, chs)
        # win_att --- (epochs, 2, 1/3, wins, chs)
        # ini_s returned and last_o returned need tackling furthermore (should reset zeros or None when new_file starts)
        return [outputs], next_ini, next_last, [ch_attention], [win_attention]

    def reset_ele(self, ch_info, ini, o, refer=None):
        ch_info = tf.cast(ch_info, self.dtype)
        exp_dims = o[0].shape.ndims - ch_info.shape.ndims
        exp_ch_info = tf.reshape(ch_info, ch_info.shape + (1,) * exp_dims)
        new_o = (tf.multiply(o[0], exp_ch_info), None)
        new_ini = (tuple(tf.multiply(e, exp_ch_info) for e in ini[0]), None)
        if refer is not None:
            new_refer = tf.multiply(refer, tf.reshape(ch_info, ch_info.shape + (1,) * (refer.shape.ndims -
                                                                                       ch_info.shape.ndims)))
        else:
            new_refer = None
        return new_ini, new_o, new_refer


class ABiGRURectifyForStackBi(keras.Model):
    def __init__(self, name, config):
        super(ABiGRURectifyForStackBi, self).__init__(name=name, dtype=config.dtype)
        self.config = config
        self.data_format = self.config.data_format
        self.has_dynamics = 'df' in config.inputs
        if 't' in config.inputs:
            self.select_t = TimeConv('time_conv_net', self.config)
            self.dropout_t = keras.layers.Dropout(self.config.dropout_t, noise_shape=(1, None, None, None, None),
                                                  name='time_dropout_before_attention', dtype=self.dtype)
        else:
            self.select_t = None
            self.dropout_t = None
        if 'f' in config.inputs:
            self.select_fre = nn.SelectF('select_fre', self.config.f_low_num, self.config.f_mid_num,
                                         self.config.select_f, self.config.seed, self.has_dynamics, self.config.f_wd,
                                         self.config.f_he_scale, self.dtype, self.data_format)
            self.dropout_fre = keras.layers.Dropout(self.config.dropout_f, noise_shape=(1, None, None, None, None),
                                                    name='fre_dropout_before_attention', dtype=self.dtype)
        else:
            self.select_fre = None
            self.dropout_fre = None
        self.disassemble = nn.Disassemble('disassemble', self.config.seq_len, self.config.seq_batch_size,
                                          self.config.padding_mode)
        self.stack_bi_gru = nn.StackBiGRUPlusAttention('stack_bi_gru', self.config.seq_len, self.config.seed,
                                                       self.config.rnn_h_units, self.config.focus_units,
                                                       self.config.rnn_mmode, self.config.rnn_layers,
                                                       self.config.focus_bias, self.config.focus_wd, self.dtype,
                                                       self.config.rnn_h_wd, self.config.rnn_input_dropout,
                                                       self.config.rnn_hidden_dropout)
        self.classification = nn.RectifyClassification('multi_task_classification', self.config.classes,
                                                       self.config.classifier_dr_rate, self.config.seed,
                                                       self.config.classifier_bias, self.dtype, self.config.wd)
        self.regulate_layers = None

    def regu(self):
        if self.regulate_layers is None:
            regulate_layers = []
            for l in self.layers:
                if hasattr(l, 'regu'):
                    regulate_layers.append(l)
            self.regulate_layers = regulate_layers
        r = []
        for l in self.regulate_layers:
           r += l.regu()
        return r

    def get_config(self):
        config = {'name': self.name, 'config': self.config}
        return config

    @classmethod
    def from_config(cls, kwargs):
        return cls(**kwargs)

    def get_input(self):
        if 't' in self.config.inputs:
            t = tf.zeros((self.config.batch_size, self.config.chs, self.config.epoch_second * self.config.fs),
                         dtype=self.dtype)
        else:
            t = None
        if 'f' in self.config.inputs:
            f = tf.zeros((self.config.batch_size, self.config.chs, self.config.wins, self.config.fft), dtype=self.dtype)
        else:
            f = None
        if self.has_dynamics:
            df = tf.zeros((self.config.batch_size, self.config.chs, self.config.wins, self.config.fft, 2),
                          dtype=self.dtype)
        else:
            df = None
        inputs = (t, f, df)
        initial_states = None
        last_output = None
        refer = None
        no_begin = tf.constant([True], dtype='bool')
        return inputs, initial_states, last_output, no_begin, refer   # when use please append 'training' & 'classifier'

    # inputs --- (T, F) or (T, F, DF)     per fre element --- (epochs, chs, wins, points)
    #                                             time element --- (epochs, chs, epoch_sampling_points)
    # initial_states --- (((1, h_units, f) or None, None) or None) * layers or None
    # last_outputs --- ((1, bottom_h_units, f) or None, None) or None
    # refer --- (1, integrated_top_h_units) or None
    # no_begin --- None or 1-D Tensor (bool) True indicates needs to use self-attention
    def call(self, inputs, initial_states=None, last_outputs=None, no_begin=None, training=None, refer=None,
             classifier=True):
        n_class = self.config.classes
        if self.config.heap is not None:
            time_heap = self.config.heap
            fre_heap = int(self.config.heap * self.config.wins)
        else:
            time_heap = None
            fre_heap = None

        if inputs[0] is not None and self.select_t is not None:
            t = tf.expand_dims(inputs[0], axis=-1)
            t = self.select_t(t, training, time_heap)
            t = self.dropout_t(t, training)
        else:
            t = None
        if inputs[1] is not None and self.select_fre is not None:
            epochs = inputs[1].shape[0]
            f = tf.expand_dims(inputs[1], axis=-1)
            if inputs[2] is not None and self.has_dynamics:
                f = tf.concat([f, inputs[2]], axis=-1)
            del inputs
            gc.collect()
            f = tf.transpose(f, (0, 2, 1, 3, 4))
            f = tf.concat(tf.unstack(f, f.shape[0], axis=0), axis=0)
            f = self.select_fre(f, training, fre_heap)
            f = tf.stack(tf.split(f, epochs, axis=0), axis=0)
            f = self.dropout_fre(f, training)
        else:
            del inputs
            gc.collect()
            f = None
        if t is not None and f is not None:
            all_in = tf.concat((t, f), axis=-1)
        elif t is not None:
            all_in = t
        elif f is not None:
            all_in = f
        del t, f
        gc.collect()
        _f = all_in.shape[-1]
        # (seqs, steps, 1, wins, chs, xxx, _f)  mask output shape: (seqs, steps, 1)
        all_in, mask = self.disassemble(all_in)
        ini_s = initial_states
        last_o = last_outputs
        last_refer = refer
        seqs = len(all_in)
        # the last seq may have different steps
        outputs = tf.TensorArray(self.dtype, size=seqs, infer_shape=False)
        ch_attention = tf.TensorArray(self.dtype, size=seqs, infer_shape=False)
        win_attention = tf.TensorArray(self.dtype, size=seqs, infer_shape=False)
        exp_no_begin = (no_begin,) + (None,) * (seqs - 1)  # can't use tf.range
        if classifier:
            for i in range(seqs):
                if mask is not None:
                    temp_m = mask[i]
                else:
                    temp_m = None
                gru_o = self.stack_bi_gru(all_in[i], ini_s, last_o, temp_m, exp_no_begin[i], training,
                                          self.config.relu_leaky, self.config.relu_threshold)
                result = self.classification(tf.concat(tf.unstack(gru_o[0], _f, axis=-1), axis=-1), last_refer, temp_m,
                                             training)
                last_refer = result[-1]
                result = tf.concat(result[:-1], axis=-1)  # (steps, 1, n_class + 2)
                outputs = outputs.write(i, result)
                ch_attention = ch_attention.write(i, gru_o[3])
                win_attention = win_attention.write(i, gru_o[4])
                ini_s = tuple((s_i[0], None) for s_i in gru_o[1])
                last_o = (gru_o[2][0], None)  # bottom layer
            outputs = outputs.concat()  # (steps * seqs, 1, n_class + 2)
            # (batch, n_class + 2)
            outputs = tf.reshape(outputs, (-1, outputs.shape[-1]))
            outputs = tf.split(outputs, n_class + 2, axis=-1)
            out1 = [tf.concat(outputs[:n_class], axis=-1)]
            out2 = [tf.concat(outputs[-2:], axis=-1)]
        else:
            last_refer = None
            for i in range(seqs):
                if mask is not None:
                    temp_m = mask[i]
                else:
                    temp_m = None
                gru_o = self.stack_bi_gru(all_in[i], ini_s, last_o, temp_m, exp_no_begin[i], training,
                                          self.config.relu_leaky, self.config.relu_threshold)
                result = tf.concat(tf.unstack(gru_o[0], _f, axis=-1), axis=-1)  # (steps, 1, integrated_top_units)
                outputs = outputs.write(i, result)
                ch_attention = ch_attention.write(i, gru_o[3])
                win_attention = win_attention.write(i, gru_o[4])
                ini_s = tuple((s_i[0], None) for s_i in gru_o[1])
                last_o = (gru_o[2][0], None)  # bottom layer
            outputs = outputs.concat()  # (steps * seqs, 1, integrated_top_units)
            # (batch, integrated_top_units)
            outputs = tf.reshape(outputs, (-1, outputs.shape[-1]))
            out1 = [outputs]
            out2 = None

        next_ini = ini_s
        next_last = last_o
        next_refer = last_refer
        ch_attention = tf.concat(tuple(ch_attention.read(i) for i in range(seqs)), axis=1)
        ch_attention = tf.transpose(ch_attention, (1, 2, 0, 4, 3))
        # (batch, 2, f, chs)
        ch_attention = tf.reshape(ch_attention, tf.concat([tf.constant((-1,), dtype='int32'),
                                                           tf.constant(ch_attention.shape[-3:])], axis=0))
        win_attention = tf.concat(tuple(win_attention.read(i) for i in range(seqs)), axis=1)
        win_attention = tf.transpose(win_attention, (1, 2, 0, 5, 3, 4))
        # (batch, 2, f, wins, chs)
        win_attention = tf.reshape(win_attention, tf.concat([tf.constant((-1,), dtype='int32'),
                                                            tf.constant(win_attention.shape[-4:])], axis=0))
        # 已经全部使用mask对齐过结果
        # output --- (epochs, n_class/top_units)    ch_att --- (epochs, 2, 1/3, chs)
        # win_att --- (epochs, 2, 1/3, wins, chs)
        # ini_s returned and last_o returned need tackling furthermore (should reset zeros or None when new_file starts)
        return out1, out2, next_ini, next_last, [ch_attention], [win_attention], next_refer

    def reset_ele(self, ch_info, ini, o, refer=None):
        ch_info = tf.cast(ch_info, self.dtype)
        exp_dims = o[0].shape.ndims - ch_info.shape.ndims
        exp_ch_info = tf.reshape(ch_info, ch_info.shape + (1,) * exp_dims)
        new_o = (tf.multiply(o[0], exp_ch_info), None)
        new_ini = tuple((tf.multiply(e[0], exp_ch_info), None) for e in ini)
        if refer is not None:
            new_refer = tf.multiply(refer, tf.reshape(ch_info, ch_info.shape + (1,) * (refer.shape.ndims -
                                                                                       ch_info.shape.ndims)))
        else:
            new_refer = None
        return new_ini, new_o, new_refer


class ABiGRURectifyForBiWrapper(keras.Model):
    def __init__(self, name, config):
        super(ABiGRURectifyForBiWrapper, self).__init__(name=name, dtype=config.dtype)
        self.config = config
        self.data_format = self.config.data_format
        self.has_dynamics = 'df' in config.inputs
        if 't' in config.inputs:
            self.select_t = TimeConv('time_conv_net', self.config)
            self.dropout_t = keras.layers.Dropout(self.config.dropout_t, noise_shape=(1, None, None, None, None),
                                                  name='time_dropout_before_attention', dtype=self.dtype)
        else:
            self.select_t = None
            self.dropout_t = None
        if 'f' in config.inputs:
            self.select_fre = nn.SelectF('select_fre', self.config.f_low_num, self.config.f_mid_num,
                                         self.config.select_f, self.config.seed, self.has_dynamics, self.config.f_wd,
                                         self.config.f_he_scale, self.dtype, self.data_format)
            self.dropout_fre = keras.layers.Dropout(self.config.dropout_f, noise_shape=(1, None, None, None, None),
                                                    name='fre_dropout_before_attention', dtype=self.dtype)
        else:
            self.select_fre = None
            self.dropout_fre = None
        self.disassemble = nn.Disassemble('disassemble', self.config.seq_len, self.config.seq_batch_size,
                                          self.config.padding_mode)
        self.bi_wrapper = nn.BidirectionalWrapper('bi_wrapper_for_gru_plus_attention', self.config.seq_len,
                                                  self.config.seed, self.config.rnn_h_units, self.config.focus_units,
                                                  self.config.rnn_layers, self.config.bi_wrapper_mmode,
                                                  self.config.focus_bias, self.config.focus_wd, self.dtype,
                                                  self.config.rnn_h_wd, self.config.rnn_input_dropout,
                                                  self.config.rnn_hidden_dropout)
        self.classification = nn.RectifyClassification('multi_task_classification', self.config.classes,
                                                       self.config.classifier_dr_rate, self.config.seed,
                                                       self.config.classifier_bias, self.dtype, self.config.wd)
        self.regulate_layers = None

    def regu(self):
        if self.regulate_layers is None:
            regulate_layers = []
            for l in self.layers:
                if hasattr(l, 'regu'):
                    regulate_layers.append(l)
            self.regulate_layers = regulate_layers
        r = []
        for l in self.regulate_layers:
           r += l.regu()
        return r

    def get_config(self):
        config = {'name': self.name, 'config': self.config}
        return config

    @classmethod
    def from_config(cls, kwargs):
        return cls(**kwargs)

    def get_input(self):
        if 't' in self.config.inputs:
            t = tf.zeros((self.config.batch_size, self.config.chs, self.config.epoch_second * self.config.fs),
                         dtype=self.dtype)
        else:
            t = None
        if 'f' in self.config.inputs:
            f = tf.zeros((self.config.batch_size, self.config.chs, self.config.wins, self.config.fft), dtype=self.dtype)
        else:
            f = None
        if self.has_dynamics:
            df = tf.zeros((self.config.batch_size, self.config.chs, self.config.wins, self.config.fft, 2),
                          dtype=self.dtype)
        else:
            df = None
        inputs = (t, f, df)
        initial_states = None
        last_output = None
        no_begin = tf.constant([True], dtype='bool')
        refer = None
        return inputs, initial_states, last_output, no_begin, refer   # when use please append 'training' & 'classifier'

    # inputs --- (T, F) or (T, F, DF)     per fre element --- (epochs, chs, wins, points)
    #                                             time element --- (epochs, chs, epoch_sampling_points)
    # initial_states --- ((((1, h_units, f) or None) * layers or None), None) or None
    # last_outputs --- ((1, top_h_units, f) or None, None) or None
    # refer --- (1, integrated_top_h_unit) or None
    # no_begin --- None or 1-D Tensor (bool) True indicates needs to use self-attention
    def call(self, inputs, initial_states=None, last_outputs=None, no_begin=None, training=None, refer=None,
             classifier=True):
        n_class = self.config.classes
        if self.config.heap is not None:
            time_heap = self.config.heap
            fre_heap = int(self.config.heap * self.config.wins)
        else:
            time_heap = None
            fre_heap = None

        # for time
        if inputs[0] is not None and self.select_t is not None:
            t = tf.expand_dims(inputs[0], axis=-1)
            t = self.select_t(t, training, time_heap)
            t = self.dropout_t(t, training)
        else:
            t = None
        # for frequency
        if inputs[1] is not None and self.select_fre is not None:
            epochs = inputs[1].shape[0]
            f = tf.expand_dims(inputs[1], axis=-1)
            if inputs[2] is not None and self.has_dynamics:
                f = tf.concat([f, inputs[2]], axis=-1)
            del inputs
            gc.collect()
            f = tf.transpose(f, (0, 2, 1, 3, 4))
            f = tf.concat(tf.unstack(f, f.shape[0], axis=0), axis=0)
            f = self.select_fre(f, training, fre_heap)
            f = tf.stack(tf.split(f, epochs, axis=0), axis=0)
            f = self.dropout_fre(f, training)
        else:
            del inputs
            gc.collect()
            f = None
        if t is not None and f is not None:
            all_in = tf.concat((t, f), axis=-1)
        elif t is not None:
            all_in = t
        elif f is not None:
            all_in = f
        del t, f
        gc.collect()
        _f = all_in.shape[-1]
        # (seqs, steps, 1, wins, chs, xxx, _f)  mask output shape: (seqs, steps, 1)
        all_in, mask = self.disassemble(all_in)
        ini_s = initial_states
        last_o = last_outputs
        last_refer = refer
        seqs = len(all_in)
        # the last seq may have different steps
        outputs = tf.TensorArray(self.dtype, size=seqs, infer_shape=False)
        ch_attention = tf.TensorArray(self.dtype, size=seqs, infer_shape=False)
        win_attention = tf.TensorArray(self.dtype, size=seqs, infer_shape=False)
        exp_no_begin = (no_begin,) + (None,) * (seqs - 1)  # can't use tf.range
        if classifier:
            for i in range(seqs):
                if mask is not None:
                    temp_m = mask[i]
                else:
                    temp_m = None
                gru_o = self.bi_wrapper(all_in[i], ini_s, last_o, temp_m, exp_no_begin[i], training,
                                        self.config.relu_leaky, self.config.relu_threshold)
                result = self.classification(tf.concat(tf.unstack(gru_o[0], _f, axis=-1), axis=-1), last_refer,
                                             temp_m, training)
                last_refer = result[-1]
                result = tf.concat(result[:-1], axis=-1)  # (steps, 1, n_class + 2)
                outputs = outputs.write(i, result)
                ch_attention = ch_attention.write(i, gru_o[3])
                win_attention = win_attention.write(i, gru_o[4])
                ini_s = (gru_o[1][0], None)
                last_o = (gru_o[2][0], None)  # top layer
            outputs = outputs.concat()  # (steps * seqs, 1, n_class + 2)
            # (batch, n_class + 2)
            outputs = tf.reshape(outputs, (-1, outputs.shape[-1]))
            outputs = tf.split(outputs, n_class + 2, axis=-1)
            out1 = [tf.concat(outputs[:n_class], axis=-1)]
            out2 = [tf.concat(outputs[-2:], axis=-1)]
        else:
            last_refer = None
            for i in range(seqs):
                if mask is not None:
                    temp_m = mask[i]
                else:
                    temp_m = None
                gru_o = self.bi_wrapper(all_in[i], ini_s, last_o, temp_m, exp_no_begin[i], training,
                                        self.config.relu_leaky, self.config.relu_threshold)
                result = tf.concat(tf.unstack(gru_o[0], _f, axis=-1), axis=-1)   # (steps, 1, integrated_top_units)
                outputs = outputs.write(i, result)
                ch_attention = ch_attention.write(i, gru_o[3])
                win_attention = win_attention.write(i, gru_o[4])
                ini_s = (gru_o[1][0], None)
                last_o = (gru_o[2][0], None)  # top layer
            outputs = outputs.concat()  # (steps * seqs, 1, integrated_top_units)
            # (batch, integrated_top_units)
            outputs = tf.reshape(outputs, (-1, outputs.shape[-1]))
            out1 = [outputs]
            out2 = None

        next_ini = ini_s
        next_last = last_o
        next_refer = last_refer
        ch_attention = tf.transpose(ch_attention.concat(), (0, 1, 2, 4, 3))
        # (batch, 2, f, chs)
        ch_attention = tf.reshape(ch_attention, tf.concat([tf.constant((-1,), dtype='int32'),
                                                           tf.constant(ch_attention.shape[-3:])], axis=0))
        win_attention = tf.transpose(win_attention.concat(), (0, 1, 2, 5, 3, 4))
        # (batch, 2, f, wins, chs)
        win_attention = tf.reshape(win_attention, tf.concat([tf.constant((-1,), dtype='int32'),
                                                            tf.constant(win_attention.shape[-4:])], axis=0))
        # 已经全部使用mask对齐过结果
        # output --- (epochs, n_class/top_units)    ch_att --- (epochs, 2, 1/3, chs)
        # win_att --- (epochs, 2, 1/3, wins, chs)
        # ini_s returned and last_o returned need tackling furthermore (should reset zeros or None when new_file starts)
        return out1, out2, next_ini, next_last, [ch_attention], [win_attention], next_refer

    def reset_ele(self, ch_info, ini, o, refer=None):
        ch_info = tf.cast(ch_info, self.dtype)
        exp_dims = o[0].shape.ndims - ch_info.shape.ndims
        exp_ch_info = tf.reshape(ch_info, ch_info.shape + (1,) * exp_dims)
        new_o = (tf.multiply(o[0], exp_ch_info), None)
        new_ini = (tuple(tf.multiply(e, exp_ch_info) for e in ini[0]), None)
        if refer is not None:
            new_refer = tf.multiply(refer, tf.reshape(ch_info, ch_info.shape + (1,) * (refer.shape.ndims -
                                                                                       ch_info.shape.ndims)))
        else:
            new_refer = None
        return new_ini, new_o, new_refer
