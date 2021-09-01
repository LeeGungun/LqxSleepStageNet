# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import numpy as np
from math import ceil, sin, cos, pi, pow, sqrt
import gc


class MyHeInitializer(keras.initializers.Initializer):
    def __init__(self, seed, distribution, fan_in, scale, dtype=tf.float32):
        super(MyHeInitializer, self).__init__()
        self.seed = seed
        self.distribution = distribution   # ('normal', 'uniform')
        self.dtype = dtype
        self.fan_in = fan_in    # list or int
        if isinstance(self.fan_in, int):
            self.fan_in = [self.fan_in]
        self.scale = scale

    def __call__(self, shape, dtype=None, partition_info=None):
        for i, d in enumerate(self.fan_in):
            if d < 0:
                d += len(shape)
            if d < 0 or d >= len(shape):
                raise ValueError('initializer calling fails, initializer\'s have some wrong in axes of shape')
            self.fan_in[i] = d
        if dtype is None:
            dtype = self.dtype
        dims = 1
        for d in self.fan_in:
            dims *= shape[d]
        scale = self.scale / dims
        if self.distribution == 'normal':
            # divided by scipy.stats.truncnorm.std(-2, 2, loc=0, scale=1) to
            # promise the variance equal to scale after truncated
            stddev = tf.cast(tf.sqrt(scale) / .87962566103423978, dtype)
            return tf.random.truncated_normal(
                shape, 0.0, stddev, dtype, seed=self.seed)
        if self.distribution == 'uniform':
            limit = tf.cast(tf.sqrt(3 * scale), dtype)
            return tf.random.uniform(shape, -limit, limit, dtype, self.seed)


class MyGlorotInitializer(keras.initializers.Initializer):
    def __init__(self, seed, distribution, fan_in, fan_out, dtype=tf.float32):
        super(MyGlorotInitializer, self).__init__()
        self.seed = seed
        self.distribution = distribution   # ('normal', 'uniform')
        self.dtype = dtype
        # fan_in and fan_out --- list or int
        self.fan_in = fan_in
        if isinstance(self.fan_in, int):
            self.fan_in = [self.fan_in]
        self.fan_out = fan_out
        if isinstance(self.fan_out, int):
            self.fan_out = [self.fan_out]

    def __call__(self, shape, dtype=None, partition_info=None):
        for i, d in enumerate(self.fan_in):
            if d < 0:
                d += len(shape)
            if d < 0 or d >= len(shape):
                raise ValueError('initializer calling fails, initializer\'s have some wrong in axes of shape')
            self.fan_in[i] = d
        for i, d in enumerate(self.fan_out):
            if d < 0:
                d += len(shape)
            if d < 0 or d >= len(shape):
                raise ValueError('initializer calling fails, initializer\'s have some wrong in axes of shape')
            self.fan_out[i] = d
        if dtype is None:
            dtype = self.dtype
        dims_in = 1
        for d in self.fan_in:
            dims_in *= shape[d]
        dims_out = 1
        for d in self.fan_out:
            dims_out *= shape[d]
        scale = 2 / (dims_in + dims_out)
        if self.distribution == 'normal':
            # divided by scipy.stats.truncnorm.std(-2, 2, loc=0, scale=1) to
            # promise the variance equal to scale after truncated
            stddev = tf.cast(tf.sqrt(scale) / .87962566103423978, dtype)
            return tf.random.truncated_normal(
                shape, 0.0, stddev, dtype, seed=self.seed)
        if self.distribution == 'uniform':
            limit = tf.cast(tf.sqrt(3 * scale), dtype)
            return tf.random.uniform(shape, -limit, limit, dtype, self.seed)


class MyOrthogonalInitializer(keras.initializers.Initializer):
    def __init__(self, seed=[None], gain=1., dtype=tf.float32):
        super(MyOrthogonalInitializer, self).__init__()
        self.seed = seed
        if isinstance(self.seed, int):
            self.seed = [self.seed]
        if self.seed is None:
            self.seed = [None]
        self.gain = gain
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        num = 1
        if len(shape) < 2:
            raise ValueError('shape must two-dimensional at least')
        if shape[-2] != shape[-1]:
            raise ValueError('orthogonal initializer needs shape meeting square matrix!')
        if len(shape) > 2:
            for a in shape[:-2]:
                num *= a
        shape_ = shape[-2:]
        if dtype is None:
            dtype = self.dtype
        seeds = len(self.seed)
        if num > 1:
            t = tf.TensorArray(dtype, num, dynamic_size=False)
            for i in range(num):
                ori_matrix = tf.random.normal(shape_, dtype=dtype, seed=self.seed[i % seeds])
                q, r = tf.linalg.qr(ori_matrix, full_matrices=False)
                d = tf.linalg.diag_part(r)
                q *= tf.sign(d)
                q *= self.gain
                t = t.write(i, q)
            return tf.reshape(t.stack(), shape)
        else:
            ori_matrix = tf.random.normal(shape_, dtype=dtype, seed=self.seed[0])
            q, r = tf.linalg.qr(ori_matrix, full_matrices=False)
            d = tf.linalg.diag_part(r)
            q *= tf.sign(d)
            q *= self.gain
            q = tf.reshape(q, shape)
            return q


# can't be used in Graph mode
class BN(keras.layers.Layer):
    # note that 'axes'&'r_axes' need to be in ascending order
    def __init__(self, name, axes=None, r_axes=None, epislon=1e-4, momentum=0.99, is_renorm=False,
                 renorm_momentum=0.99, renorm_clip=None, dtype=tf.float32):
        super(BN, self).__init__(name=name, dtype=dtype)
        self.axes = axes   # 'all' or int or list or tuple, need to be normalized
        self.r_axes = r_axes   # int or list or tuple, when normalizing without regard to them
        if axes is None and r_axes is None:
            raise Exception('should indicate one ax at least')
        if isinstance(self.axes, int):
            self.axes = [self.axes]
        elif isinstance(axes, tuple):
            self.axes = list(self.axes)
        if isinstance(self.r_axes, int):
            self.r_axes = [self.r_axes]
        self.is_renorm = is_renorm
        # a minimum epsilon is 1.001e-5, which is a requirement by CUDNN to
        # prevent exception (see cudnn.h).
        if epislon < 1.001e-5:
            self.epislon = 1.001e-5
        else:
            self.epislon = epislon
        self.momentum = momentum
        self.renorm_momentum = renorm_momentum
        self.renorm_clip = renorm_clip      # if not None, is tuple(rmin, rmax, dmax) eg: (0.95, 1.05, 0.1)
        self.supports_masking = True

    def get_config(self):
        config = super(BN, self).get_config()
        config.update({'axes': tuple(self.axes) if hasattr(self.axes, '__iter__') else self.axes,
                       'r_axes': tuple(self.r_axes) if hasattr(self.r_axes, '__iter__') else self.r_axes,
                       'epislon': self.epislon, 'momentum': self.momentum, 'is_renorm': self.is_renorm,
                       'renorm_momentum': self.renorm_momentum,
                       'renorm_clip': (tuple(self.renorm_clip)
                                       if hasattr(self.renorm_clip, '__iter__') else self.renorm_clip)})
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'axes', 'r_axes', 'epislon', 'momentum', 'is_renorm', 'renorm_momentum', 'renorm_clip']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    # in fact, if hasattr(self.build, '_is_default') is True, indicates the 'build' method has not been overridden
    def build(self, input_shape):
        dims = input_shape.ndims
        if self.axes is not None:
            if self.axes == 'all':
                self.axes = [i for i in range(dims)]
            else:
                for i, x in enumerate(self.axes):
                    if x < 0:
                        x += dims
                    if x < 0 or x >= dims:
                        raise ValueError('invalid ax: {}'.format(x))
                    self.axes[i] = x
        else:
            self.axes = [i for i in range(dims)]
            for x in self.r_axes:
                if x < 0:
                    x += dims
                if x < 0 or x >= dims:
                    raise ValueError('invalid ax: {}'.format(x))
                self.axes.remove(x)
        params_shape = [input_shape[i] if i not in self.axes else 1 for i in range(dims)]
        self.gamma = self.add_weight(name='gamma', shape=params_shape, dtype=self.dtype, initializer='Ones',
                                     regularizer=None, trainable=True, constraint=None)
        self.beta = self.add_weight(name='beta', shape=params_shape, dtype=self.dtype, initializer='Zeros',
                                    regularizer=None, trainable=True, constraint=None)
        self.moving_mean = self.add_weight(name='moving_mean', shape=params_shape, dtype=self.dtype,
                                           initializer='Zeros', regularizer=None, trainable=False, constraint=None)
        self.moving_variance = self.add_weight(name='moving_variance', shape=params_shape, dtype=self.dtype,
                                               initializer='Ones', regularizer=None, trainable=False, constraint=None)
        if self.is_renorm:
            self.moving_stddev = self.add_weight(name='moving_stddev', shape=params_shape, dtype=self.dtype,
                                                 initializer='Ones', regularizer=None, trainable=False, constraint=None)
            self.renorm_mean = self.add_weight(name='renorm_mean', shape=params_shape, dtype=self.dtype,
                                               initializer='Zeros', regularizer=None, trainable=False, constraint=None)
            self.renorm_stddev = self.add_weight(name='renorm_stddev', shape=params_shape, dtype=self.dtype,
                                                 initializer='Ones', regularizer=None, trainable=False, constraint=None)
        self.built = True

    def _moving_average(self, var, value, momentum, training):
        delta = tf.multiply(tf.subtract(var, value), tf.cast(tf.subtract(1., momentum), var.dtype))
        return var.assign_sub(tf.cond(training, lambda: delta, lambda: tf.zeros_like(delta)))

    def _renorm(self, mean, variance, training):
        stddev = tf.sqrt(variance + self.epislon)
        renorm_stddev = tf.maximum(self.renorm_stddev, tf.sqrt(tf.cast(self.epislon, self.dtype)))
        r = tf.divide(stddev, renorm_stddev)
        d = tf.divide(tf.subtract(mean, self.renorm_mean), renorm_stddev)
        with tf.control_dependencies([r, d]):    # r, d need use value before update
            # use mean, stddev to update self.renorm_mean, self.renorm_stddev separately if training
            self._moving_average(self.renorm_mean, mean, self.renorm_momentum, training)
            self._moving_average(self.renorm_stddev, stddev, self.renorm_momentum, training)
        # clip
        if self.renorm_clip is not None and len(self.renorm_clip) == 3:
            clip_r = tf.clip_by_value(r, self.renorm_clip[0], self.renorm_clip[1])
            clip_d = tf.clip_by_value(d, -self.renorm_clip[2], self.renorm_clip[2])
        else:
            clip_r = r
            clip_d = d
        # after_corrected_value = normalized_value * r + d ---- is training
        r, d = tf.cond(training, lambda: (clip_r, clip_d), lambda: (tf.ones_like(clip_r), tf.zeros_like(clip_d)))
        return r, d

    # use moving_stddev modify moving_variance
    def _update_renorm_mv(self, value, training):
        stddev = self._moving_average(self.moving_stddev, tf.sqrt(value + self.epislon), self.momentum, training)
        return self.moving_variance.assign(tf.relu(tf.pow(stddev, 2) - self.epislon))  # use relu promise to be positive

    def _use_renorm(self, mean, variance, training):
        r, d = self._renorm(mean, variance, training)
        r = tf.stop_gradient(r)
        d = tf.stop_gradient(d)
        gamma = tf.multiply(self.gamma, r)
        beta = tf.add(tf.multiply(self.gamma, d), self.beta)
        return gamma, beta

    def call(self, inputs, mask=None, training=None):
        inputs = tf.cast(inputs, self.dtype)
        if training is None:
            training = tf.constant(False)
        else:
            training = tf.constant(training)

        # get current mean&variance
        def _moments(inp=inputs, msk=mask):
            if msk is None:
                m, v = tf.nn.moments(inp, self.axes, keepdims=True)
            else:
                exp_dims = inp.shape.ndims - msk.shape.ndims
                if tf.reduce_any(tf.not_equal(inp.shape[: -exp_dims]), msk.shape):
                    raise ValueError('the shapes of inputs and mask must be equal at the corresponding axes '
                                     'beginning with the first one')
                exp_mask = tf.broadcast_to(tf.reshape(msk, msk.shape + (1,) * exp_dims), inp.shape)
                exp_mask = tf.cast(exp_mask, inp.dtype)
                fact = tf.reduce_sum(exp_mask, axis=self.axes, keepdims=True)
                m = tf.reduce_sum(inp * exp_mask, axis=self.axes, keepdims=True)
                m = m / fact
                v = tf.reduce_sum(tf.square(inp - m) * exp_mask, axis=self.axes, keepdims=True)
                v = v / fact
                # if use @tf.function，whatever this branch will be accessed during tracing
                # even if use 'with tf.init_scope()' so --- transform executing mode also need tracing
                if tf.reduce_any(v < 0):
                    raise ValueError('variance generated by BN with mask has some elements small than zero')
            return m, v

        mean, variance = tf.cond(training, lambda: _moments(),
                                 lambda: (tf.identity(self.moving_mean), tf.identity(self.moving_variance)))

        if self.is_renorm:
            # renormalization
            gamma, beta = self._use_renorm(mean, variance, training)
        else:
            gamma = tf.identity(self.gamma)
            beta = tf.identity(self.beta)

        # update moving_mean&moving_variance
        self._moving_average(self.moving_mean, mean, self.momentum, training)
        if self.is_renorm:
            self._update_renorm_mv(variance, training)
        else:
            self._moving_average(self.moving_variance, variance, self.momentum, training)

        # apply parameters to correct inputs
        outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, self.epislon,
                                            name='outputs')
        return outputs


# data_format = 'NCHW'
class SelectF(keras.layers.Layer):
    def __init__(self, name, low_num, mid_num, select, seed, has_dynamics=False, wd=0., he_scale=2,
                 dtype=tf.float32, data_format='NCHW'):
        '''

        :param name:
        :param low_num: int, the number of what is categorized into low frequency bands
        :param mid_num: int, the number of what is categorized into middle frequency bands
        :param select: 2-element nested tuple/list, per element also a 3-element tuple/list
        :param seed:
        :param has_dynamics: whether have the dynamic frequency features in 'call' method as input
        :param wd:
        :param he_scale:
        :param dtype:
        :param data_format:
        '''
        super(SelectF, self).__init__(name=name, dtype=dtype)
        self.low_num = low_num
        self.mid_num = mid_num
        if not hasattr(select, '__iter__') or len(select) != 2:
            raise ValueError('the param \'select\' must be iterative and have length of 2')
        for s in select:
            if not hasattr(s, '__iter__') or len(s) != 3:
                raise ValueError('the each element in param \'select\' should have information corresponding of three '
                                 'frequency ranges')
            # in fact, need to judge the relationship of size between per element and
            #  corresponding current frequency ranges
        self.select = select
        self.seed = seed
        self.has_dynamics = has_dynamics
        self.wd = wd
        self.data_format = data_format
        if self.wd:
            self.regularizer = keras.regularizers.l2(self.wd)
        else:
            self.regularizer = None
        self.regulate_vars = []
        self.spectrum_bn = BN('spectrum_bn', r_axes=[1, -1], dtype=self.dtype)  # because 'NHWC' ---> 'NCHW'
        if self.has_dynamics:
            self.delta_bn = BN('delta_bn', r_axes=[1, -1], dtype=self.dtype)
            self.d_delta_bn = BN('delta_delta_bn', r_axes=[1, -1], dtype=self.dtype)
            self.sublayer_num = 3
        else:
            self.delta_bn = None
            self.d_delta_bn = None
            self.sublayer_num = 1
        self.he_scale = he_scale

    def get_config(self):
        config = super(SelectF, self).get_config()
        config.update({'low_num': self.low_num,
                       'mid_num': self.mid_num,
                       'select': tuple(self.select),
                       'seed': self.seed,
                       'has_dynamics': self.has_dynamics,
                       'wd': self.wd,
                       'he_scale': self.he_scale,
                       'data_format': self.data_format})
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'low_num', 'mid_num', 'select', 'seed', 'has_dynamics', 'wd', 'he_scale', 'data_format']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def build(self, input_shape):
        c = 3 - self.data_format.find('H')
        self.power_high_w = self.add_weight(name='high_spectrum_weights',
                                            shape=(input_shape[c], self.select[0][0], self.select[1][0]),
                                            dtype=self.dtype,
                                            initializer=MyHeInitializer(self.seed, 'normal', 1, self.he_scale),
                                            regularizer=self.regularizer,
                                            trainable=True,
                                            constraint=None)
        self.regulate_vars.append(self.power_high_w)
        self.power_mid_w = self.add_weight(name='mid_spectrum_weights',
                                           shape=(input_shape[c], self.select[0][1], self.select[1][1]),
                                           dtype=self.dtype,
                                           initializer=MyHeInitializer(self.seed, 'normal', 1, self.he_scale),
                                           regularizer=self.regularizer,
                                           trainable=True,
                                           constraint=None)
        self.regulate_vars.append(self.power_mid_w)
        self.power_low_w = self.add_weight(name='low_spectrum_weights',
                                           shape=(input_shape[c], self.select[0][2], self.select[1][2]),
                                           dtype=self.dtype,
                                           initializer=MyHeInitializer(self.seed, 'normal', 1, self.he_scale),
                                           regularizer=self.regularizer,
                                           trainable=True,
                                           constraint=None)
        self.regulate_vars.append(self.power_low_w)
        if self.has_dynamics:
            high_in = input_shape[3 - c] - self.low_num - self.mid_num
            self.dynamic_high_w = self.add_weight(name='high_dynamic_weights',
                                                  shape=(input_shape[c], high_in, self.select[1][0]),
                                                  dtype=self.dtype,
                                                  initializer=MyGlorotInitializer(self.seed, 'normal', 1, 2),
                                                  regularizer=self.regularizer,
                                                  trainable=True,
                                                  constraint=None)
            self.regulate_vars.append(self.dynamic_high_w)
            self.dynamic_mid_w = self.add_weight(name='mid_dynamic_weights',
                                                 shape=(input_shape[c], self.mid_num, self.select[1][1]),
                                                 dtype=self.dtype,
                                                 initializer=MyGlorotInitializer(self.seed, 'normal', 1, 2),
                                                 regularizer=self.regularizer,
                                                 trainable=True,
                                                 constraint=None)
            self.regulate_vars.append(self.dynamic_mid_w)
            self.dynamic_low_w = self.add_weight(name='low_dynamic_weights',
                                                 shape=(input_shape[c], self.low_num, self.select[1][2]),
                                                 dtype=self.dtype,
                                                 initializer=MyGlorotInitializer(self.seed, 'normal', 1, 2),
                                                 regularizer=self.regularizer,
                                                 trainable=True,
                                                 constraint=None)
            self.regulate_vars.append(self.dynamic_low_w)
        else:
            self.dynamic_high_w = None
            self.dynamic_mid_w = None
            self.dynamic_low_w = None
        self.built = True

    def regu(self):
        if self.regularizer is not None:
            r = []
            for v in self.regulate_vars:
                r.append(tf.clip(self.regularizer(v), 1e-8, np.Inf))
            return r
        return []

    def get_input(self, batch_epochs, chs, wins, fft, f_features=1):
        if self.data_format == 'NCHW':
            f = tf.zeros((batch_epochs * wins, chs, fft, f_features), dtype=self.dtype)
        else:
            f = tf.zeros((batch_epochs * wins, fft, chs, f_features), dtype=self.dtype)
        return f    # when use please append 'heap'

    # 1D
    @staticmethod
    def prepare_sliding(input, output):
        if input >= output:  # down-sampling
            inp = input
            out = output
            down = True  # mask for input
        else:  # up-sampling
            inp = output
            out = input
            down = False  # mask for output
        stride = int(inp / (out - 1))
        window = int(inp - (out - 1) * stride)
        if window < stride:
            padding = stride - window
            window = stride
            mask = tf.concat([tf.zeros((int(padding / 2),), dtype='bool'), tf.ones((int(inp),), dtype='bool'),
                              tf.zeros((int(ceil(padding / 2)),), dtype='bool')], axis=0)
        else:
            mask = None
        return window, stride, mask, down

    # 1D data shape: (epochs*wins, chs, xxx, f), only for down-sampling
    @classmethod
    def self_adjust(cls, data, input, output):

        window, stride, mask, _ = cls.prepare_sliding(input, output)

        def __apply_kernel(d):
            out = int((d.shape[0] - window) / stride) + 1
            _output = tf.TensorArray(d.dtype, size=out)
            for i in range(out):
                item = d[i * stride: i * stride + window]
                o = tf.nn.softmax(item, axis=0)
                o = tf.reduce_sum(tf.multiply(item, o), axis=0, keepdims=True)
                _output = _output.write(i, o)
            return _output.concat()

        if mask is not None and data.shape[-2] < mask.shape[-1]:  # need padding
            indices = tf.where(mask)
            start = indices[0, 0]  # tensor scalar  'int64' by default
            end = indices[-1, 0]
            padding_h = tf.zeros(data.shape[:2] + (start,) + data.shape[-1:], dtype=data.dtype)
            padding_t = tf.zeros(data.shape[:2] + (mask.shape[-1] - 1 - end,) + data.shape[-1:], dtype=data.dtype)
            _data = tf.concat([padding_h, data, padding_t], axis=-2)
        else:
            _data = data
        # take advantage of factual value
        result = __apply_kernel(tf.transpose(_data, (2, 0, 1, 3)))
        result = tf.transpose(result, (1, 2, 0, 3))
        return result

    # inputs --- F(1) or F(3)      (epochs*wins, chs, xxx, 1/3)
    # return --- F(1)_select_num or F(3)_select_num
    def call(self, inputs, training=None, heap=None):
        fre = tf.cast(inputs, self.dtype)
        # promised shape --- (epochs*wins, chs, xxx, 1/3)
        if self.data_format == 'NHWC':
            fre = tf.transpose(fre, (0, 2, 1, 3))

        low = fre[:, :, :self.low_num, :]
        mid = fre[:, :, self.low_num: (self.mid_num + self.low_num), :]
        high = fre[:, :, (self.low_num + self.mid_num):, :]

        # for power spectrum
        #@tf.function
        def _compute_for_power(_low, _mid, _high):
            power_low = self.self_adjust(_low, self.low_num, self.select[0][-1])
            power_mid = self.self_adjust(_mid, self.mid_num, self.select[0][1])
            power_high = self.self_adjust(_high, _high.shape[-2], self.select[0][0])
            power_low = tf.matmul(tf.transpose(power_low, (0, 1, 3, 2)), tf.sigmoid(self.power_low_w))
            power_low = tf.transpose(power_low, (0, 1, 3, 2))
            power_mid = tf.matmul(tf.transpose(power_mid, (0, 1, 3, 2)), tf.sigmoid(self.power_mid_w))
            power_mid = tf.transpose(power_mid, (0, 1, 3, 2))
            power_high = tf.matmul(tf.transpose(power_high, (0, 1, 3, 2)), tf.sigmoid(self.power_high_w))
            power_high = tf.transpose(power_high, (0, 1, 3, 2))
            return power_low, power_mid, power_high

        # for dynamic features
        #F@tf.function
        def _comput_for_dynamic(_low, _mid, _high):
            dynamic_low = tf.matmul(tf.transpose(_low, (0, 1, 3, 2)), self.dynamic_low_w)
            dynamic_low = tf.transpose(dynamic_low, (0, 1, 3, 2))
            dynamic_mid = tf.matmul(tf.transpose(_mid, (0, 1, 3, 2)), self.dynamic_mid_w)
            dynamic_mid = tf.transpose(dynamic_mid, (0, 1, 3, 2))
            dynamic_high = tf.matmul(tf.transpose(_high, (0, 1, 3, 2)), self.dynamic_high_w)
            dynamic_high = tf.transpose(dynamic_high, (0, 1, 3, 2))
            return dynamic_low, dynamic_mid, dynamic_high

        if heap is not None:
            remaining = int(fre.shape[0] % heap)
            heaps = ceil(fre.shape[0] / heap)
            if remaining == 0:
                split = heaps
            else:
                split = [heap] * (heaps - 1) + [remaining]
            inv_low = tf.split(low, split, axis=0)
            del low
            inv_mid = tf.split(mid, split, axis=0)
            del mid
            inv_high = tf.split(high, split, axis=0)
            del high
            gc.collect()
            ta_low = tf.TensorArray(self.dtype, size=heaps, infer_shape=False)
            ta_mid = tf.TensorArray(self.dtype, size=heaps, infer_shape=False)
            ta_high = tf.TensorArray(self.dtype, size=heaps, infer_shape=False)
            if not self.has_dynamics:
                for i in range(heaps):
                    l, m, h = _compute_for_power(inv_low[i][:, :, :, 0: 1], inv_mid[i][:, :, :, 0: 1],
                                                 inv_high[i][:, :, :, 0: 1])
                    ta_low = ta_low.write(i, l)
                    ta_mid = ta_mid.write(i, m)
                    ta_high = ta_high.write(i, h)
                result = self.spectrum_bn(tf.concat([ta_low.concat(), ta_mid.concat(), ta_high.concat()], axis=-2),
                                          None, training)
            else:
                for i in range(heaps):
                    l, m, h = _compute_for_power(inv_low[i][:, :, :, 0: 1], inv_mid[i][:, :, :, 0: 1],
                                                 inv_high[i][:, :, :, 0: 1])
                    l_d, m_d, h_d = _comput_for_dynamic(inv_low[i][:, :, :, 1:], inv_mid[i][:, :, :, 1:],
                                                        inv_high[i][:, :, :, 1:])
                    ta_low = ta_low.write(i, tf.concat([l, l_d], axis=-1))
                    ta_mid = ta_mid.write(i, tf.concat([m, m_d], axis=-1))
                    ta_high = ta_high.write(i, tf.concat([h, h_d], axis=-1))
                l = ta_low.concat()
                del inv_low
                m = ta_mid.concat()
                del inv_mid
                h = ta_high.concat()
                del inv_high
                gc.collect()
                power = self.spectrum_bn(tf.concat([l[:, :, :, 0: 1], m[:, :, :, 0: 1], h[:, :, :, 0: 1]], axis=-2),
                                         None, training)
                delta = self.delta_bn(tf.concat([l[:, :, :, 1: 2], m[:, :, :, 1: 2], h[:, :, :, 1: 2]], axis=-2),
                                      None, training)
                delta_delta = self.d_delta_bn(tf.concat([l[:, :, :, 2:], m[:, :, :, 2:], h[:, :, :, 2:]], axis=-2),
                                              None, training)
                result = tf.concat([power, tf.tanh(tf.concat([delta, delta_delta], axis=-1))], axis=-1)
        else:
            power = self.spectrum_bn(tf.concat(_compute_for_power(low[:, :, :, 0: 1], mid[:, :, :, 0: 1], 
                                                                  high[:, :, :, 0: 1]), axis=-2), None, training)
            if self.has_dynamics:
                l, m, h = _comput_for_dynamic(low[:, :, :, 1:], mid[:, :, :, 1:], high[:, :, :, 1:])
                delta = self.delta_bn(tf.concat([l[:, :, :, 0: 1], m[:, :, :, 0: 1], h[:, :, :, 0: 1]], axis=-2),
                                      None, training)
                delta_delta = self.d_delta_bn(tf.concat([l[:, :, :, 1:], m[:, :, :, 1:], h[:, :, :, 1:]], axis=-2),
                                              None, training)
                del low
                del mid
                del high
                gc.collect()
                result = tf.concat([power, tf.tanh(tf.concat([delta, delta_delta], axis=-1))], axis=-1)
            else:
                result = power

        if self.data_format == 'NHWC':
            result = tf.transpose(result, (0, 2, 1, 3))
        return result


class Conv(keras.layers.Layer):
    def __init__(self, name, seed, filters, stride, activate, padding, conv,
                 wd=0.001, he_scale=2, data_format='NCHW', dtype=tf.float32):
        '''
            customized to time data --- 1D data
        :param name:
        :param seed:
        :param filters: tuple/list, is 2-element — kernel size, filters
        :param stride: int
        :param activate: False or str in ('relu', 'tanh', 'sigmoid'), indicates the corresponding layer whether
                         need relu activation or not, or use what kind of activation
        :param padding: str, 'SAME' or 'VALID'
        :param conv: str in ('in_chs', 'normal', 'separate')
        :param wd: float, indicates the convolution of the
                      corresponding layer whether need weights regularizer or not
        :param he_scale:
        :param data_format:
        :param dtype:
        '''
        super(Conv, self).__init__(name=name, dtype=dtype)
        self.seed = seed

        if not hasattr(filters, '__iter__') or len(filters) not in (2, 3):
            raise ValueError('parameter \'filters\' must be 2-element or 3-element iterative')
        else:
            self.filters = filters    # per element --- (kernel, filters)

        if not isinstance(stride, int):
            raise ValueError('parameter \'stride\' must be int')
        else:
            self.stride = stride

        if not isinstance(activate, bool) and activate not in ('tanh', 'relu', 'sigmoid'):
            raise ValueError('parameter \'activate\' has wrong value')
        elif isinstance(activate, bool) and activate:
            raise ValueError('parameter \'activate\' has wrong value')
        else:
            self.activate = activate

        if padding not in ('SAME', 'VALID'):
            raise ValueError('parameter \'padding\' has wrong value')
        else:
            self.padding = padding

        if conv not in ('in_chs', 'normal', 'separate'):
            raise ValueError('parameter \'conv\' has wrong value')
        else:
            if conv in ('in_chs', 'separate') and len(self.filters) != 3:
                raise Exception('parameters \'conv\' and \'filters\' do not match with each other')
            elif conv == 'normal' and len(self.filters) != 2:
                raise Exception('parameters \'conv\' and \'filters\' do not match with each other')
            else:
                self.conv = conv

        self.wd = wd
        if wd:
            self.regularizer = keras.regularizers.l2(wd)
        else:
            self.regularizer = None
        self.regulate_vars = []

        self.he_scale = he_scale
        self.data_format = data_format
        # transform to ---> 'NHWC', reserve axes: pattern, f, ch
        self.bn = keras.layers.BatchNormalization(axis=-1)
        self.sublayer_num = 1

    def get_config(self):
        config = super(Conv, self).get_config()
        config.update({'seed': self.he_seed,
                       'filters': tuple(self.d_filters),
                       'stride': self.stride,
                       'activate': self.activate,
                       'padding': self.padding,
                       'conv': self.conv,
                       'wd': self.wd,
                       'he_scale': self.he_scale,
                       'data_format': self.data_format})
        return config

    @classmethod
    def form_config(cls, kwargs):
        ks = ['name', 'dtype', 'seed', 'filters', 'stride', 'activate', 'padding', 'conv', 'wd',
              'he_scale', 'data_format']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def build(self, input_shape):
        c = 3 - self.data_format.find('H')
        self.w = []
        if self.conv == 'in_chs':
            self.w.append(
                self.add_weight(name='depth_weights',
                                shape=(input_shape[-1], self.filters[0], 1, input_shape[c], self.filters[1]),
                                dtype=self.dtype,
                                initializer=MyHeInitializer(self.seed, 'normal', [1, 2, 3], self.he_scale),
                                regularizer=self.regularizer,
                                trainable=True,
                                constraint=None)
            )
            self.regulate_vars.append(self.w[-1])
            self.w.append(
                self.add_weight(name='in_chs_weights',
                                shape=(input_shape[-1], 1, self.filters[2], input_shape[c], self.filters[1]),
                                dtype=self.dtype,
                                initializer=MyHeInitializer(self.seed, 'normal', [-1, -2], self.he_scale),
                                regularizer=self.regularizer,
                                trainable=True,
                                constraint=None)
            )
            self.regulate_vars.append(self.w[-1])
        elif self.conv == 'separate':
            self.w.append(
                self.add_weight(name='depth_weights',
                                shape=(input_shape[-1], self.filters[0], 1, input_shape[c], self.filters[1]),
                                dtype=self.dtype,
                                initializer=MyHeInitializer(self.seed, 'normal', [1, 2, 3], self.he_scale),
                                regularizer=self.regularizer,
                                trainable=True,
                                constraint=None)
            )
            self.regulate_vars.append(self.w[-1])
            self.w.append(
                self.add_weight(name='point_filters',
                                shape=(input_shape[-1], 1, 1, input_shape[c] * self.filters[1], self.filters[2]),
                                dtype=self.dtype,
                                initializer=MyHeInitializer(self.seed, 'normal', [1, 2, 3], self.he_scale),
                                regularizer=self.regularizer,
                                trainable=True,
                                constraint=None)
            )
            self.regulate_vars.append(self.w[-1])
        else:
            self.w.append(
                self.add_weight(name='filters',
                                shape=(input_shape[-1], self.filters[0], 1, input_shape[c], self.filters[1]),
                                dtype=self.dtype,
                                initializer=MyHeInitializer(self.seed, 'normal', [1, 2, 3], self.he_scale),
                                regularizer=self.regularizer,
                                trainable=True,
                                constraint=None)
            )
            self.regulate_vars.append(self.w[-1])
        self.built = True

    def regu(self):
        if not self.regu_flag:
            r = []
            for i, rg in enumerate(self.regularizer):
                if rg is not None:
                    for v in self.regulate_vars[i]:
                        r.append(tf.clip_by_value(rg(v), 1e-8, np.Inf))
            return r
        return []

    def get_input(self, batch_epochs, chs, width, f):
        if self.data_format == 'NCHW':
            t = tf.zeros((batch_epochs, chs, width, f), dtype=self.dtype)
        else:
            t = tf.zeros((batch_epochs, width, chs, f), dtype=self.dtype)
        return t

    # inputs  (batch_size, in_chs, xxx, f)
    # outputs (batch_size, out_chs, xxx, f)
    def call(self, inputs, training=None, heap=None, **kwargs):
        inputs = tf.cast(inputs, self.dtype)
        if self.data_format == 'NCHW':
            inputs = tf.transpose(inputs, (0, 2, 1, 3))
        in_chs = inputs.shape[-2]
        features = inputs.shape[-1]

        # @tf.function
        def _call(inp):
            # output (epochs, fs * epoch_seconds, 1, chs, f)
            inp = tf.expand_dims(inp, axis=-3)
            # output (epochs, fs * epoch_seconds, 1, chs) * f
            inp = tf.unstack(inp, features, axis=-1)
            # output (epochs, h', 1, filters, f)
            if self.conv == 'in_chs':
                # output (epochs, f, h', 1, in_chs * filters)
                _result = tf.stack(tuple(tf.nn.depthwise_conv2d(inp[i], self.w[0][i],
                                                                [1, self.stride, self.stride, 1], self.padding, 'NHWC')
                                         for i in range(features)), axis=1)
                # output (epochs, f, h', p_filters, in_chs, filters)
                _result = tf.multiply(tf.stack(tf.split(_result, in_chs, axis=-1), axis=-2), self.w[1])
                _result = tf.reduce_sum(_result, axis=-1)
                # output (epochs, h', in_chs * p_filters, f)
                _result = tf.transpose(tf.concat(tf.unstack(_result, axis=-1), axis=-1), (0, 2, 3, 1))
            elif self.conv == 'separate':
                # output (epochs, h', 1, p_filters, f)
                _result = tf.stack(tuple(tf.nn.separable_conv2d(inp[i], self.w[0][i], self.w[1][i],
                                                                [1, self.stride, self.stride, 1], self.padding, 'NHWC')
                                         for i in range(features)), axis=-1)
                # (epochs, h', p_filters, f)
                _result = tf.squeeze(_result, axis=-3)
            else:
                _result = tf.stack(tuple(tf.nn.conv2d(inp[i], self.w[0][i], [1, self.stride, 1, 1], self.padding,
                                                      'NHWC') for i in range(features)), axis=-1)
                # (epochs, h', filters, f)
                _result = tf.squeeze(_result, axis=-3)
            return _result

        if heap is not None:
            remaining = int(inputs.shape[0] % heap)
            heaps = ceil(inputs.shape[0] / heap)
            if remaining == 0:
                split = heaps
            else:
                split = [heap] * (heaps - 1) + [remaining]
            inventory = tf.split(inputs, split, axis=0)
            del inputs
            gc.collect()
            ta = tf.TensorArray(self.dtype, size=heaps, infer_shape=False)
            for i in range(heaps):
                ta = ta.write(i, _call(inventory[i]))
            result = ta.concat()
        else:
            result = _call(inputs)
            del inputs
            gc.collect()
        _f = result.shape[-1]
        result = self.bn(tf.concat(tf.unstack(result, axis=-1), axis=-1), training)
        result = tf.stack(tf.split(result, _f, axis=-1), axis=-1)
        if self.activate == 'relu':
            result = keras.activations.relu(result, alpha=kwargs.get('relu_leaky', 0.),
                                            threshold=kwargs.get('relu_threshold', 0.))
        elif self.activate == 'tanh':
            result = tf.tanh(result)
        elif self.activate == 'sigmoid':
            result = tf.sigmoid(result)
        if self.data_format == 'NCHW':
            result = tf.transpose(result, (0, 2, 1, 3))
        return result


# customized to time data --- 1D ---- depth_conv / normal conv / separate conv
class PatternConv(keras.layers.Layer):
    def __init__(self, name, seed, filters, stride, suppression, padding, conv,
                 wd=0.001, he_scale=2, data_format='NCHW', dtype=tf.float32):
        '''
            customized to time data --- 1D data
        :param name:
        :param seed:
        :param filters: tuple/list, is 3-element — kernel size, depth filter depth and point filter depth,
                        or 2-element — kernel size, output filter depth
        :param stride: int
        :param suppression: 2-element tuple/list, indicates the suppression weight of activation of the two patterns
                            except normal pattern
        :param padding: str, 'SAME' or 'VALID'
        :param conv: str in ('in_chs', 'normal', 'separate')
        :param wd: float, indicates the convolution whether need weights regularizer or not
        :param he_scale:
        :param data_format:
        :param dtype:
        '''
        super(PatternConv, self).__init__(name=name, dtype=dtype)
        self.seed = seed

        if not hasattr(filters, '__iter__') or len(filters) not in (2, 3):
            raise ValueError('parameter \'filters\' must be 2-element or 3-element iterative')
        else:
            self.filters = filters    # per element --- (kernel, filters) / (kernel, depth_filters, point_filters)

        if not isinstance(stride, int):
            raise ValueError('parameter \'stride\' must be int')
        else:
            self.stride = stride

        if not hasattr(suppression, '__iter__') or len(suppression) != 2:
            raise ValueError('parameter \'suppression\' must be 2-element iterative')
        else:
            self.suppression = suppression   # (for bi_pattern, for tri_pattern)

        if padding not in ('SAME', 'VALID'):
            raise ValueError('parameter \'padding\' has wrong value')
        else:
            self.padding = padding

        if conv not in ('in_chs', 'normal', 'separate'):
            raise ValueError('parameter \'conv\' has wrong value')
        else:
            if conv in ('in_chs', 'separate') and len(self.filters) != 3:
                raise Exception('parameters \'conv\' and \'filters\' do not match with each other')
            elif conv == 'normal' and len(self.filters) != 2:
                raise Exception('parameters \'conv\' and \'filters\' do not match with each other')
            else:
                self.conv = conv

        self.wd = wd
        if wd:
            self.regularizer = keras.regularizers.l2(wd)
        else:
            self.regularizer = None
        self.regulate_vars = []

        self.he_scale = he_scale
        self.data_format = data_format
        # transform to ---> 'NHWC', reserve axes: pattern, f, ch
        self.bn = keras.layers.BatchNormalization(axis=-1)
        self.sublayer_num = 1

    def get_config(self):
        config = super(PatternConv, self).get_config()
        config.update({'seed': self.seed,
                       'filters': tuple(self.filters),
                       'stride': self.stride,
                       'suppression': tuple(self.suppression),
                       'padding': self.padding,
                       'conv': self.conv,
                       'wd': self.wd,
                       'he_scale': self.he_scale,
                       'data_format': self.data_format})
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'seed', 'filters', 'stride', 'suppression', 'padding', 'conv',
              'he_scale', 'wd', 'data_format']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def build(self, input_shape):
        c = 3 - self.data_format.find('H')
        self.w = []
        if self.conv == 'in_chs':
            self.w.append(
                self.add_weight(name='depth_weights',
                                shape=(6, input_shape[0][-1], self.filters[0], 1, input_shape[0][c], self.filters[1]),
                                dtype=self.dtype,
                                initializer=MyHeInitializer(self.seed, 'normal', [2, 3, 4], self.he_scale),
                                regularizer=self.regularizer,
                                trainable=True,
                                constraint=None)
            )
            self.regulate_vars.append(self.w[-1])
            self.w.append(
                self.add_weight(name='in_chs_weights',
                                shape=(6, input_shape[0][-1], 1, self.filters[2], input_shape[0][c], self.filters[1]),
                                dtype=self.dtype,
                                initializer=MyHeInitializer(self.seed, 'normal', [-1, -2], self.he_scale),
                                regularizer=self.regularizer,
                                trainable=True,
                                constraint=None)
            )
            self.regulate_vars.append(self.w[-1])
        elif self.conv == 'separate':
            self.w.append(
                self.add_weight(name='depth_weights',
                                shape=(6, input_shape[0][-1], self.filters[0], 1, input_shape[0][c], self.filters[1]),
                                dtype=self.dtype,
                                initializer=MyHeInitializer(self.seed, 'normal', [2, 3, 4], self.he_scale),
                                regularizer=self.regularizer,
                                trainable=True,
                                constraint=None)
            )
            self.regulate_vars.append(self.w[-1])
            self.w.append(
                self.add_weight(name='point_filters',
                                shape=(6, input_shape[0][-1], 1, 1, input_shape[0][c] * self.filters[1],
                                       self.filters[2]),
                                dtype=self.dtype,
                                initializer=MyHeInitializer(self.seed, 'normal', [2, 3, 4], self.he_scale),
                                regularizer=self.regularizer,
                                trainable=True,
                                constraint=None)
            )
            self.regulate_vars.append(self.w[-1])
        else:
            self.w.append(
                self.add_weight(name='filters',
                                shape=(6, input_shape[0][-1], self.filters[0], 1, input_shape[0][c], self.filters[1]),
                                dtype=self.dtype,
                                initializer=MyHeInitializer(self.seed, 'normal', [2, 3, 4], self.he_scale),
                                regularizer=self.regularizer,
                                trainable=True,
                                constraint=None)
            )
            self.regulate_vars.append(self.w[-1])
        self.built = True

    def regu(self):
        if not self.regu_flag:
            r = []
            for i, rg in enumerate(self.regularizer):
                if rg is not None:
                    for v in self.regulate_vars[i]:
                        r.append(tf.clip_by_value(rg(v), 1e-8, np.Inf))
            return r
        return []

    def get_input(self, batch_epochs, chs, width, f):
        if self.data_format == 'NCHW':
            t = tf.zeros((batch_epochs, chs, width, f), dtype=self.dtype)
        else:
            t = tf.zeros((batch_epochs, width, chs, f), dtype=self.dtype)
        return t

    # time data input : (epochs, chs, fs * epoch_seconds, 1)
    # output: normal_patter, bi_pattern, tri_pattern --- (epochs, chs * all_points_prod, h', 1)
    def call(self, inputs, training=None, heap=None, relu_leaky=0.01, relu_threshold=0):
        inputs = tf.cast(inputs, self.dtype)
        if self.data_format == 'NCHW':
            inputs = tf.transpose(inputs, (0, 2, 1, 3))
        f = inputs.shape[-1]
        in_chs = inputs.shape[-2]
        shape = inputs.shape

        # @tf.function
        def _call(inp, idx):
            # output (epochs, fs * epoch_seconds, 1, chs, f)
            inp = tf.expand_dims(inp, axis=-3)
            # output (epochs, fs * epoch_seconds, 1, chs) * f
            inp = tf.unstack(inp, f, axis=-1)
            # output (epochs, h', 1, filters, f)
            if self.conv == 'in_chs':
                # output (epochs, f, h', 1, in_chs * filters)
                _result = tf.stack(tuple(tf.nn.depthwise_conv2d(inp[i], self.w[0][idx][i],
                                                                [1, self.stride, self.stride, 1], self.padding, 'NHWC')
                                         for i in range(f)), axis=1)
                # output (epochs, f, h', p_filters, in_chs, filters)
                _result = tf.multiply(tf.stack(tf.split(_result, in_chs, axis=-1), axis=-2), self.w[1][idx])
                _result = tf.reduce_sum(_result, axis=-1)
                # output (epochs, h', in_chs * p_filters, f)
                _result = tf.transpose(tf.concat(tf.unstack(_result, axis=-1), axis=-1), (0, 2, 3, 1))
            elif self.conv == 'separate':
                # output (epochs, h', 1, p_filters, f)
                _result = tf.stack(tuple(tf.nn.separable_conv2d(inp[i], self.w[0][idx][i], self.w[1][idx][i],
                                                                [1, self.stride, self.stride, 1], self.padding, 'NHWC')
                                         for i in range(f)), axis=-1)
                # (epochs, h', p_filters, f)
                _result = tf.squeeze(_result, axis=-3)
            else:
                _result = tf.stack(tuple(tf.nn.conv2d(inp[i], self.w[0][idx][i], [1, self.stride, 1, 1], self.padding,
                                                      'NHWC') for i in range(f)), axis=-1)
                # (epochs, h', filters, f)
                _result = tf.squeeze(_result, axis=-3)
            # (epochs, h', f * xxx)
            _result = tf.concat(tf.unstack(_result, axis=-1), axis=-1)
            return _result

        def _suppress(inp, suppression):
            neg_part = tf.nn.relu(-inp)
            sup_part = tf.nn.relu(inp) * float(suppression)
            return tf.add(-neg_part, sup_part)

        # inp (epochs, h, p * chs)
        def _pattern_add(inp, idx, p, suppression):
            inp = tf.split(inp, p, axis=-1)
            inp = tuple(inp[i] if idx == i else _suppress(inp[i], suppression) for i in range(p))
            # (epochs, h, chs)
            return sum(inp)

        def _inner_tackle(inp):
            out = inp.shape[1]
            split_bi = (out // 2, int(out - out // 2))
            split_tri = (out // 3, int(out - 2 * (out // 3)), out // 3)
            # output (epochs, h', 6 * f * chs)
            _result = self.bn(inp, training)
            del inp
            gc.collect()
            _result = keras.activations.relu(_result, alpha=relu_leaky, threshold=relu_threshold)
            _result = tf.split(_result, [1, 2, 3], axis=-1)
            # (epochs, h', chs, f)
            nor_ = tf.stack(tf.split(_result[0], f, axis=-1), axis=-1)
            bi_ = tf.split(_result[1], split_bi, axis=1)
            # (epochs, h'/ 2, f * chs)_2
            bi_ = tf.concat(tuple(_pattern_add(bi_[i], i, 2, self.suppression[0]) for i in range(2)), axis=1)
            # (epochs, h', chs, f)
            bi_ = tf.stack(tf.split(tf.concat(bi_, axis=1), f, axis=-1), axis=-1)
            tri_ = tf.split(_result[2], split_tri, axis=1)
            # (epochs, h'/ 3, f * chs)_3
            tri_ = tf.concat(tuple(_pattern_add(tri_[i], i, 3, self.suppression[1]) for i in range(3)), axis=1)
            # (epochs, h', chs, f)
            tri_ = tf.stack(tf.split(tf.concat(tri_, axis=1), f, axis=-1), axis=-1)
            return nor_, bi_, tri_

        if heap is not None:
            remaining = int(shape[0] % heap)
            heaps = ceil(shape[0] / heap)
            if remaining == 0:
                split = heaps
            else:
                split = [heap] * (heaps - 1) + [remaining]
            normal = tf.split(inputs, split, axis=0)
            bi = tf.split(inputs, split, axis=0)
            tri = tf.split(inputs, split, axis=0)
            del inputs
            gc.collect()
            ta = tf.TensorArray(self.dtype, size=heaps, infer_shape=False)
            for i in range(heaps):
                #                shape (heap_size, h', 6 * f * in_chs)
                ta = ta.write(i, tf.concat((_call(normal[i], 0),) + tuple(_call(bi[i], 1 + i) for i in range(2))
                                           + tuple(_call(tri[i], 3 + i) for i in range(3)), axis=-1))
            del normal, bi, tri
            gc.collect()
            # output (epochs, h', 6 * f * in_chs)
            result = ta.concat()
            result = _inner_tackle(result)
        else:
            result = _inner_tackle(tf.concat((_call(inputs, 0),) + tuple(_call(inputs, 1 + i) for i in range(2))
                                             + tuple(_call(inputs, 3 + i) for i in range(3)), axis=-1))
        if self.data_format == 'NCHW':
            result = tuple(tf.transpose(result[i], (0, 2, 1, 3)) for i in range(3))
        return result


# note to adjust input shape first
class VarianceSE(keras.layers.Layer):
    def __init__(self, name, trans_channels, seed, he_scale, bias=False, wd=0.001, dtype=tf.float32):
        super(VarianceSE, self).__init__(name=name, dtype=dtype)
        self.trans_channels = trans_channels
        self.seed = seed
        self.he_scale = he_scale
        self.bias = bias
        self.wd = wd
        if self.wd:
            self.regularizer = keras.regularizers.l2(self.wd)
        else:
            self.regularizer = None
        self.regulate_vars = []

    def get_config(self):
        config = super(VarianceSE, self).get_config()
        config.update({'trans_channels': self.trans_channels,
                       'seed': self.seed,
                       'he_scale': self.he_scale,
                       'bias': self.bias,
                       'wd': self.wd})
        return config

    @classmethod
    def form_config(cls, kwargs):
        ks = ['name', 'dtype', 'trans_channels', 'seed', 'he_scale', 'bias', 'wd']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def build(self, input_shape):
        self.trans_w = self.add_weight(name='trans_weights',
                                       shape=(input_shape[-1], self.trans_channels),
                                       dtype=self.dtype,
                                       initializer=MyHeInitializer(self.seed, 'normal', 0, self.he_scale),
                                       regularizer=self.regularizer,
                                       trainable=True,
                                       constraint=None)
        self.regulate_vars.append(self.trans_w)
        self.trans_b = self.add_weight(name='trans_bias',
                                       shape=(self.trans_channels,),
                                       dtype=self.dtype,
                                       initializer=keras.initializers.get('zeros'),
                                       regularizer=None,
                                       trainable=self.bias,
                                       constraint=None)
        self.restore_w = self.add_weight(name='restore_weights',
                                         shape=(self.trans_channels, input_shape[-1]),
                                         dtype=self.dtype,
                                         initializer=MyGlorotInitializer(self.seed, 'normal', 0, 1),
                                         regularizer=self.regularizer,
                                         trainable=True,
                                         constraint=None)
        self.regulate_vars.append(self.restore_w)
        self.built = True

    def regu(self):
        if self.regularizer is not None:
            r = []
            for v in self.regulate_vars:
                r.append(tf.clip_by_value(self.regularizer(v), 1e-8, np.Inf))
            return r
        return []

    def get_input(self, batch, wins, f, use_chs, patterns):
        return tf.zeros((batch, wins, f, use_chs, patterns), dtype=self.dtype)

    # inputs  --- (epochs, f, wins, xxx, patterns)
    # outputs --- (epochs, f, patterns)
    @tf.function
    def call(self, inputs, relu_leaky=0.01, relu_threshold=0., heap=None):
        inp = tf.cast(inputs, self.dtype)

        # position encode
        C = inp.shape[-2]
        base = 2 * (ceil(C / 2) - 1)
        encoded = np.zeros(inp.shape[2:-1], dtype=self.dtype)
        for i in range(ceil(C / 2)):
            for pos in range(inp.shape[-3]):
                encoded[2 * i][pos] = sin(pi * (pos + 0.5) / pow(C, 2 * i / base))
                if 2 * i + 1 < C:
                    encoded[2 * i + 1][pos] = cos(pi * (pos + 0.5) / pow(C, 2 * i / base))
        encoded = tf.convert_to_tensor(encoded)
        encoded = tf.multiply(inp, encoded)
        encoded = tf.reduce_mean(encoded, axis=-3)
        # output shape --- (epochs, f, xxx, patterns)
        encoded = tf.transpose(encoded, (1, 2, 0, 3))

        def _variance_se(portion):
            # get variance
            # output shape --- (epochs, f, xxx, trans)
            portion = tf.add(tf.matmul(portion, self.trans_w), self.trans_b)
            portion = keras.activations.relu(portion, alpha=relu_leaky, threshold=relu_threshold)
            # output shape --- (epochs, f, trans_chs)
            _variance = tf.math.reduce_variance(portion, axis=-2)
            _variance = tf.matmul(_variance, self.restore_w)
            _variance = tf.nn.softmax(_variance, axis=-1)
            return _variance

        if heap is not None:
            remaining = int(encoded.shape[0] % heap)
            heaps = ceil(encoded.shape[0] / heap)
            if remaining == 0:
                split = heaps
            else:
                split = [heap] * (heaps - 1) + [remaining]
            encoded = tf.split(encoded, split, axis=0)
            ta = tf.TensorArray(self.dtype, size=heaps, infer_shape=False)
            for i in range(heaps):
                ta = ta.write(i, _variance_se(encoded[i]))
            variance = ta.concat()
        else:
            variance = _variance_se(encoded)
        return variance


# note to adjust input shape first
class SVDSE(keras.layers.Layer):
    def __init__(self, name, seed, bias=False, wd=0.001, dtype=tf.float32):
        super(SVDSE, self).__init__(name=name, dtype=dtype)
        self.seed = seed
        self.bias = bias
        self.wd = wd
        if self.wd:
            self.regularizer = keras.regularizers.l2(self.wd)
        else:
            self.regularizer = None
        self.regulate_vars = []

    def get_config(self):
        config = super(SVDSE, self).get_config()
        config.update({'seed': self.seed,
                       'bias': self.bias,
                       'wd': self.wd})
        return config

    @classmethod
    def form_config(cls, kwargs):
        ks = ['name', 'dtype', 'seed', 'bias', 'wd']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def build(self, input_shape):
        self.association_w = []
        self.association_b = []
        self.svd_w = []
        axes = len(input_shape)
        for i in range(1, 3):
            self.association_w.append(self.add_weight(
                name=f'No.{axes - i + 1}_axis_encode_weights',
                shape=(input_shape[2], input_shape[-i], input_shape[-3 + i]),
                dtype=self.dtype,
                initializer=MyHeInitializer(self.seed, 'normal', -2, 2),
                regularizer=self.regularizer,
                trainable=True,
                constraint=None
            ))
            self.association_b.append(self.add_weight(
                name=f'No.{axes - i + 1}_axis_encode_bias',
                shape=(input_shape[2], 1, input_shape[-3 + i]),
                dtype=self.dtype,
                initializer=keras.initializers.get('zeros'),
                regularizer=None,
                trainable=self.bias,
                constraint=None
            ))
            self.svd_w.append(self.add_weight(
                name=f'No.{axes - i + 1}_axis_svd_weights',
                shape=(input_shape[2], 1, input_shape[-3 + i]),
                dtype=self.dtype,
                initializer=MyHeInitializer(self.seed, 'normal', -2, 1),
                regularizer=None,
                trainable=True,
                constraint=None
            ))
        self.regulate_vars.extend(self.association_w)
        self.built = True

    def regu(self):
        if self.regularizer is not None:
            r = []
            for v in self.regulate_vars:
                r.append(tf.clip_by_value(self.regularizer(v), 1e-8, np.Inf))
            return r
        return []

    def get_input(self, group_fs, batch, ori_chs, use_chs, points):
        return tf.zeros((group_fs, batch, ori_chs, use_chs, points), dtype=self.dtype)

    # inputs  --- (epochs, 1, f*in_chs, chs, patterns)
    # outputs --- (epochs, 1, f*in_chs, chs, patterns)
    #@tf.function
    def call(self, inputs, heap=None):
        inp = tf.cast(inputs, self.dtype)
        epochs = inp.shape[0]
        if heap is not None:
            heap = int(heap * 10)
        else:
            heap = int(epochs * 10)
        inp = tf.reshape(inp, (int(inp.shape[1] * epochs),) + inp.shape[2:])
        ndims = inp.shape.ndims

        def _encode(axis, portion):
            w = tf.reshape(self.association_w[axis - 1],
                           (self.association_w[axis - 1].shape[0],) + (1,) * (ndims - 4) +
                           self.association_w[axis - 1].shape[1:])
            b = tf.reshape(self.association_b[axis - 1],
                           (self.association_b[axis - 1].shape[0],) + (1,) * (ndims - 4) +
                           self.association_b[axis - 1].shape[1:])
            ii = tf.transpose(portion,
                              tuple(range(ndims - axis)) + tuple(range(ndims - axis + 1, ndims)) + (ndims - axis,))
            r = tf.add(tf.matmul(ii, w), b)
            r = keras.activations.relu(r)
            t = tf.nn.softmax(self.svd_w[axis - 1], axis=-1)
            t = tf.reshape(t, (t.shape[0],) + (1,) * (ndims - 4) + t.shape[2:])
            s = tf.multiply(tf.add(tf.expand_dims(tf.linalg.trace(r), axis=-1),
                                   tf.ones_like(t)), t)
            return tf.expand_dims(s, axis=-axis)

        remaining = int(inp.shape[0] % heap)
        heaps = ceil(inp.shape[0] / heap)
        if remaining == 0:
            split = heaps
        else:
            split = [heap] * (heaps - 1) + [remaining]
        inp = tf.split(inp, split, axis=0)
        ta = tf.TensorArray(self.dtype, size=heaps, infer_shape=False)
        for i in range(heaps):
            ta = ta.write(i, tf.matmul(_encode(1, inp[i]), _encode(2, inp[i])))
        result = ta.concat()
        result = tf.stack(tf.split(result, epochs, axis=0), axis=0)
        return result


class MaxPool(keras.layers.Layer):
    def __init__(self, name, filters, strides, padding='SAME', data_format='NCHW'):
        super(MaxPool, self).__init__(name=name, trainable=False)
        self.filters = filters    # integer or list    [H, W]
        self.strides = strides    # integer or list    [H, W]
        self.padding = padding
        self.data_format = data_format

    def get_config(self):
        config = super(MaxPool, self).get_config()
        config.update({'filters': tuple(self.filters) if hasattr(self.filters, '__iter__') else self.filters,
                       'strides': tuple(self.strides) if hasattr(self.strides, '__iter__') else self.strides,
                       'padding': self.padding, 'data_format': self.data_format})
        return config

    @classmethod
    def form_config(cls, kwargs):
        ks = ['name', 'filters', 'strides', 'padding', 'data_format']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def get_input(self, epochs, wins, chs, features, dtype):
        if self.data_format == 'NHWC':
            return tf.zeros((epochs, wins, chs, features), dtype=dtype)
        return tf.zeros((epochs, chs, wins, features), dtype=dtype)

    # inputs --- (N, chs, xxx, f) / (N, xxx, chs, f)
    #@tf.function
    def call(self, inputs, heap=None):
        inputs = tf.cast(inputs, self.dtype)
        #_format = self.data_format.replace('H', '')
        if self.data_format == 'NCHW':
            inputs = tf.transpose(inputs, (0, 2, 1, 3))

        #@tf.function
        def _call(inp):
            features = inp.shape[-1]
            inp = tf.transpose(inp, (3, 0, 1, 2))
            inp = tf.reshape(inp, (-1, inp.shape[2], inp.shape[3]))
            _result = tf.nn.max_pool1d(inp, self.filters, self.strides, self.padding, 'NWC')
            return tf.stack(tf.split(_result, features, axis=0), axis=-1)

        if heap is not None:
            remaining = int(inputs.shape[0] % heap)
            heaps = ceil(inputs.shape[0] / heap)
            if remaining == 0:
                split = heaps
            else:
                split = [heap] * (heaps - 1) + [remaining]
            inventory = tf.split(inputs, split, axis=0)
            ta = tf.TensorArray(self.dtype, size=heaps, infer_shape=False)
            for i in range(heaps):
                ta = ta.write(i, _call(inventory[i]))
            result = ta.concat()
        else:
            result = _call(inputs)
        if self.data_format == 'NCHW':
            result = tf.transpose(result, (0, 2, 1, 3))
        return result


class AvgPool(keras.layers.Layer):
    def __init__(self, name, filters, strides, padding='SAME', data_format='NCWH'):
        super(AvgPool, self).__init__(name=name, trainable=False)
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

    def get_config(self):
        config = super(AvgPool, self).get_config()
        config.update({'filters': tuple(self.filters) if hasattr(self.filters, '__iter__') else self.filters,
                       'strides': tuple(self.strides) if hasattr(self.strides, '__iter__') else self.strides,
                       'padding': self.padding, 'data_format': self.data_format})
        return config

    @classmethod
    def form_config(cls, kwargs):
        ks = ['name', 'filters', 'strides', 'padding', 'data_format']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def get_input(self, epochs, wins, chs, features, dtype):
        if self.data_format == 'NHWC':
            return tf.zeros((epochs, wins, chs, features), dtype=dtype)
        return tf.zeros((epochs, chs, wins, features), dtype=dtype)

    # inputs --- (N, chs, xxx, f) / (N, xxx, chs, f)
    #@tf.function
    def call(self, inputs, heap=None):
        inputs = tf.cast(inputs, self.dtype)
        #_format = self.data_format.replace('H', '')
        if self.data_format == 'NCHW':
            inputs = tf.transpose(inputs, (0, 2, 1, 3))

        @tf.function
        def _call(inp):
            features = inp.shape[-1]
            inp = tf.transpose(inp, (3, 0, 1, 2))
            inp = tf.reshape(inp, (-1, inp.shape[2], inp.shape[3]))
            _result = tf.nn.avg_pool1d(inp, self.filters, self.strides, self.padding, 'NWC')
            return tf.stack(tf.split(_result, features, axis=0), axis=-1)

        if heap is not None:
            remaining = int(inputs.shape[0] % heap)
            heaps = ceil(inputs.shape[0] / heap)
            if remaining == 0:
                split = heaps
            else:
                split = [heap] * (heaps - 1) + [remaining]
            inventory = tf.split(inputs, split, axis=0)
            ta = tf.TensorArray(self.dtype, size=heaps, infer_shape=False)
            for i in range(heaps):
                ta = ta.write(i, _call(inventory[i]))
            result = ta.concat()
        else:
            result = _call(inputs)
        if self.data_format == 'NCHW':
            result = tf.transpose(result, (0, 2, 1, 3))
        return result


# output fixed data_format --- 'NCHW'
class Disassemble(keras.layers.Layer):
    def __init__(self, name, seq_len, batch_size, padding_mode):
        super(Disassemble, self).__init__(name=name, trainable=False)
        self.seq_len = seq_len
        self.batch_size = batch_size
        if padding_mode not in ('seq', 'batch'):
            raise ValueError('unqualified value passed to parameter \'padding_mode\'')
        self.padding_mode = padding_mode

    def get_config(self):
        config = super(Disassemble, self).get_config()
        config.update({
            'seq_len': self.seq_len,
            'batch_size': self.batch_size,
            'padding_mode': self.padding_mode,
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'seq_len', 'batch_size', 'padding_mode']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    # have no weights, so no get_input method

    # (epochs, wins, chs, h, f) --> ((steps, step_batch_size, wins, chs, h, f), ...)
    @tf.function
    def call(self, inputs):
        epochs = inputs.shape[0]
        ndims = inputs.shape.ndims
        seqs = int(epochs / (self.seq_len * self.batch_size))
        tail = epochs - seqs * self.seq_len * self.batch_size
        if seqs > 0:
            no_padded = tuple(map(lambda e: tf.transpose(tf.reshape(e, (self.batch_size, self.seq_len) + e.shape[1:]),
                                                         (1, 0) + tuple(range(2, ndims + 1))),
                                  tf.split(inputs[: seqs * self.seq_len * self.batch_size], seqs, axis=0)))
        else:
            no_padded = ()
        if tail > 0:
            if self.padding_mode == 'batch':
                if tail <= self.seq_len:
                    if self.batch_size > 1:
                        padded = tf.concat((tf.expand_dims(inputs[-tail:], axis=1),
                                            tf.zeros((tail, self.batch_size - 1) + inputs.shape[1:],
                                                     dtype=inputs.dtype)), axis=1)
                        mask = (None,) * seqs + (tf.concat((tf.ones((tail, 1), dtype=bool),
                                                            tf.zeros((tail, self.batch_size - 1), dtype=bool)),
                                                           axis=1),)
                        padded = no_padded + (padded,)
                        return padded, mask
                    else:
                        return no_padded + (tf.expand_dims(inputs[-tail:], axis=1),), (None,) * (seqs + 1)
            add = self.seq_len * self.batch_size * int(ceil(epochs / (self.seq_len * self.batch_size))) - epochs
            padded = tf.concat((inputs[-tail:], tf.zeros((add,) + inputs.shape[1:], dtype=inputs.dtype)), axis=0)
            mask = tf.concat((tf.ones((tail,), dtype=bool), tf.zeros((add,), dtype=bool)), axis=0)
            padded = tf.stack(tf.split(padded, self.batch_size, axis=0), axis=1)
            mask = tf.stack(tf.split(mask, self.batch_size, axis=0), axis=1)
            return no_padded + (padded,), (None,) * seqs + (mask,)
        else:
            return no_padded, (None,) * seqs


# ChannelAttention after WindowAttention
class ChannelAttention(keras.layers.Layer):
    def __init__(self, name, seed, dims, dtype=tf.float32, wd=0.001):
        super(ChannelAttention, self).__init__(name=name, dtype=dtype)
        self.dims = dims
        self.seed = seed
        self.wd = wd
        if self.wd:
            self.regularizer = keras.regularizers.l2(self.wd)
        else:
            self.regularizer = None
        self.regulate_vars = []

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            'seed': self.seed,
            'dims': self.dims,
            'wd': self.wd
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'seed', 'dims', 'wd']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def build(self, input_shape):
        assert input_shape[0][-1] == input_shape[1][-1], 'special types of rnn outputs and attention inputs must ' \
                                                         'match with each other'
        self.query_w = self.add_weight(name='query_weights',
                                       shape=(input_shape[0][-1], input_shape[0][-2], self.dims),
                                       dtype=self.dtype,
                                       initializer=MyGlorotInitializer(self.seed, 'normal', -2, -1),
                                       regularizer=self.regularizer,
                                       trainable=True,
                                       constraint=None)
        self.regulate_vars.append(self.query_w)
        self.key_w = self.add_weight(name='key_weights',
                                     shape=(input_shape[1][-1], input_shape[1][-2], self.dims),
                                     dtype=self.dtype,
                                     initializer=MyGlorotInitializer(self.seed, 'normal', -2, -1),
                                     regularizer=self.regularizer,
                                     trainable=True,
                                     constraint=None)
        self.regulate_vars.append(self.key_w)
        self.key_b = self.add_weight(name='key_bias',
                                     shape=(input_shape[1][-1], 1, self.dims),
                                     dtype=self.dtype,
                                     initializer=
                                     keras.initializers.RandomNormal(
                                         seed=self.seed, stddev=sqrt(2. / (input_shape[1][-2] + self.dims))),
                                     regularizer=self.regularizer,
                                     trainable=True,
                                     constraint=None)
        self.regulate_vars.append(self.key_b)
        self.built = True

    def regu(self):
        if self.regularizer is not None:
            r = []
            for v in self.regulate_vars:
                r.append(tf.clip_by_value(self.regularizer(v), 1e-8, np.Inf))
            return r
        return []

    def get_input(self, batch_size, output, chs, obj_p, features):
        o = tf.zeros((batch_size, output, features), dtype=self.dtype)
        f = tf.zeros((batch_size, chs, obj_p, features), dtype=self.dtype)
        inputs = (o, f)
        no_begin = None
        mask = tf.constant([True] * batch_size, dtype='bool')
        return inputs, mask, no_begin

    # GRU's the last step's output vector ----(step_batch_size, o_dims, f)，
    # window_attention_tensor------(step_batch_size, chs, dims, f)
    # inputs ---- (query vector, window attention synthesis)
    @tf.function
    def call(self, inputs, mask=None, no_begin=None):
        inputs = (tf.cast(inputs[0], self.dtype), tf.cast(inputs[1], self.dtype))
        if mask is not None and not tf.reduce_all(mask):
            inputs = (tf.multiply(inputs[0], tf.cast(tf.reshape(mask, mask.shape + (1,) * 2), self.dtype)),
                      tf.multiply(inputs[1], tf.cast(tf.reshape(mask, mask.shape + (1,) * 3), self.dtype)))
        # (batch_size, f, chs, dims)
        key = tf.tanh(tf.matmul(tf.transpose(inputs[1], (0, 3, 1, 2)), self.key_w) + self.key_b)

        # (batch_size, f, chs, 1)
        def _compute_foreign_simi(inp):
            # (f, batch_size, dims)
            query = tf.matmul(tf.transpose(inp, (2, 0, 1)), self.query_w)
            query = tf.expand_dims(query, axis=-1)
            return tf.divide(tf.matmul(key, tf.transpose(query, (1, 0, 2, 3))),
                             tf.sqrt(tf.constant(self.dims, dtype=self.dtype)))

        # (batch_size, f, chs, 1)
        def _compute_self_simi(inp):
            return tf.divide(tf.reduce_sum(tf.multiply(inp, inp), axis=-1, keepdims=True),
                             tf.sqrt(tf.constant(self.dims, dtype=self.dtype)))

        def _compute_simi():
            if no_begin is None:
                return _compute_foreign_simi(inputs[0])
            elif tf.reduce_all(no_begin):
                return _compute_simi(key)
            else:
                for_foreign = tf.cast(tf.logical_not(no_begin), self.dtype)
                exp_d1 = inputs[0].shape.ndims - for_foreign.shape.ndims
                for_self = tf.cast(no_begin, self.dtype)
                exp_d2 = key.shape.ndims - for_self.shape.ndims
                foreign = _compute_foreign_simi(tf.multiply(inputs[0],
                                                            tf.reshape(for_foreign, for_foreign.shape + (1,) * exp_d1)))
                foreign = tf.multiply(foreign, tf.reshape(for_foreign, for_foreign.shape + (1,) * 3))
                oneself = _compute_self_simi(tf.multiply(key, tf.reshape(for_self, for_self.shape + (1,) * exp_d2)))
                return foreign + oneself

        similarity = _compute_simi()
        attention = tf.nn.softmax(similarity, axis=-2)
        attention = tf.transpose(attention, (0, 2, 3, 1))
        # (batch_size, chs, f)
        focus_chs = tf.stop_gradient(tf.squeeze(attention, axis=-2))
        # (batch_size, dims, f)
        synthesis = tf.reduce_sum(tf.multiply(attention, inputs[1]), axis=-3)
        if mask is not None and not tf.reduce_all(mask):
            exp_mask = tf.cast(tf.reshape(mask, mask.shape + (1,) * 2), self.dtype)
            focus_chs = tf.multiply(focus_chs, exp_mask)
            synthesis = tf.multiply(synthesis, exp_mask)
        return focus_chs, synthesis


class WindowAttention(keras.layers.Layer):
    def __init__(self, name, seed, dims, dtype=tf.float32, wd=0.001):
        super(WindowAttention, self).__init__(name=name, dtype=dtype)
        self.dims = dims
        self.seed = seed
        self.wd = wd
        if self.wd:
            self.regularizer = keras.regularizers.l2(self.wd)
        else:
            self.regularizer = None
        self.regulate_vars = []

    def get_config(self):
        config = super(WindowAttention, self).get_config()
        config.update({
            'seed': self.seed,
            'dims': self.dims,
            'wd': self.wd
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'seed', 'dims', 'wd']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def build(self, input_shape):
        assert input_shape[0][-1] == input_shape[1][-1], 'special types of rnn outputs and attention inputs must ' \
                                                         'match with each other'
        self.query_w = self.add_weight(name='query_weights',
                                       shape=(input_shape[0][-1], input_shape[1][-3], input_shape[0][-2], self.dims),
                                       dtype=self.dtype,
                                       initializer=MyGlorotInitializer(self.seed, 'normal', -2, -1),
                                       regularizer=self.regularizer,
                                       trainable=True,
                                       constraint=None)
        self.regulate_vars.append(self.query_w)
        self.key_w = self.add_weight(name='key_weights',
                                     shape=(input_shape[1][-1], input_shape[1][-3], input_shape[1][-2], self.dims),
                                     dtype=self.dtype,
                                     initializer=MyGlorotInitializer(self.seed, 'normal', -2, -1),
                                     regularizer=self.regularizer,
                                     trainable=True,
                                     constraint=None)
        self.regulate_vars.append(self.key_w)
        self.key_b = self.add_weight(name='key_bias',
                                     shape=(input_shape[1][-1], input_shape[1][-3], 1, self.dims),
                                     dtype=self.dtype,
                                     initializer=keras.initializers.RandomNormal(
                                         seed=self.seed, stddev=sqrt(2. / (input_shape[1][-2] + self.dims))),
                                     regularizer=self.regularizer,
                                     trainable=True,
                                     constraint=None)
        self.regulate_vars.append(self.key_b)
        self.built = True

    def regu(self):
        if self.regularizer is not None:
            r = []
            for v in self.regulate_vars:
                r.append(tf.clip_by_value(self.regularizer(v), 1e-8, np.Inf))
            return r
        return []

    def get_input(self, batch_size, output, wins, chs, obj_p, features):
        o = tf.zeros((batch_size, output, features), dtype=self.dtype)
        f = tf.zeros((batch_size, wins, chs, obj_p, features), dtype=self.dtype)
        inputs = (o, f)
        no_begin = None
        mask = tf.constant([True] * batch_size, dtype='bool')
        return inputs, mask, no_begin

    # GRU's the last step's output vector ----(step_batch_size, output_dims, f)，
    # disassembled_tensor------(step_batch_size, wins, chs, dims, f)
    # inputs ---- (output vector, primary features)
    @tf.function
    def call(self, inputs, mask=None, no_begin=None):
        inputs = (tf.cast(inputs[0], self.dtype), tf.cast(inputs[1], self.dtype))
        if mask is not None and not tf.reduce_all(mask):
            inputs = (tf.multiply(inputs[0], tf.cast(tf.reshape(mask, mask.shape + (1,) * 2), self.dtype)),
                      tf.multiply(inputs[1], tf.cast(tf.reshape(mask, mask.shape + (1,) * 4), self.dtype)))
        # (batch_size, f, chs, wins, dims)
        key = tf.tanh(tf.matmul(tf.transpose(inputs[1], (0, 4, 2, 1, 3)), self.key_w) + self.key_b)

        # (batch_size, f, chs, wins, 1)
        def _compute_foreign_simi(inp):
            # (f, chs, batch_size, dims)
            query = tf.matmul(tf.expand_dims(tf.transpose(inp, (2, 0, 1)), axis=1), self.query_w)
            query = tf.expand_dims(query, axis=-1)
            return tf.divide(tf.matmul(key, tf.transpose(query, (2, 0, 1, 3, 4))),
                             tf.sqrt(tf.constant(self.dims, dtype=self.dtype)))

        def _compute_self_simi(inp):
            return tf.divide(tf.reduce_sum(tf.multiply(inp, inp), axis=-1, keepdims=True),
                             tf.sqrt(tf.constant(self.dims, dtype=self.dtype)))

        def _compute_simi():
            if no_begin is None:
                return _compute_foreign_simi(inputs[0])
            elif tf.reduce_all(no_begin):
                return _compute_simi(key)
            else:
                for_foreign = tf.cast(tf.logical_not(no_begin), self.dtype)
                exp_d1 = inputs[0].shape.ndims - for_foreign.shape.ndims
                for_self = tf.cast(no_begin, self.dtype)
                exp_d2 = key.shape.ndims - for_self.shape.ndims
                foreign = _compute_foreign_simi(
                    tf.multiply(inputs[0], tf.reshape(for_foreign, for_foreign.shape + (1,) * exp_d1))
                )
                foreign = tf.multiply(foreign, tf.reshape(for_foreign, for_foreign.shape + (1,) * 4))
                oneself = _compute_self_simi(
                    tf.multiply(key, tf.reshape(for_self, for_self.shape + (1,) * exp_d2))
                )
                return foreign + oneself

        # (batch_size, f, chs, wins, 1)
        similarity = _compute_simi()
        attention = tf.nn.softmax(similarity, axis=-2)
        attention = tf.transpose(attention, (0, 3, 2, 4, 1))
        # (batch_size, wins, chs, f)
        focus_wins = tf.stop_gradient(tf.squeeze(attention, axis=-2))
        # (batch_size, chs, dims, f)
        synthesis = tf.reduce_sum(tf.multiply(attention, inputs[1]), axis=-4)
        if mask is not None and not tf.reduce_all(mask):
            exp_mask = tf.cast(tf.reshape(mask, mask.shape + (1,) * 3), self.dtype)
            synthesis = tf.multiply(synthesis, exp_mask)
            focus_wins = tf.multiply(focus_wins, exp_mask)
        return focus_wins, synthesis


# (step_batch_size, wins, chs, dims, f)  -->  (step_batch_size, wins, chs, f) compute win_attention first
#  --> (step_batch_size, chs, f) then compute ch_attention
class Attention(keras.layers.Layer):
    def __init__(self, name, seed, dims, dtype=tf.float32, wd=0.001):
        '''

        :param name:
        :param seed:
        :param dims: int, 2-element tuple/list(one for window attention and the other for channel attention)
        :param dtype:
        :param wd: float, 2-element tuple/list(one for window attention and the other for channel attention)
        '''
        super(Attention, self).__init__(name=name, dtype=dtype)
        self.seed = seed

        if isinstance(dims, int):
            self.dims = (dims,) * 2
        elif hasattr(dims, '__iter__') and len(dims) == 2:
            self.dims = tuple(dims)
        else:
            raise ValueError('parameter \'dims\' is unqualified')

        if isinstance(wd, float):
            self.wd = (wd,) * 2
        elif hasattr(wd, '__iter__') and len(wd) == 2:
            self.wd = tuple(wd)
        else:
            raise ValueError('parameter \'wd\' is unqualified')

        self.regularizer = list(keras.regularizers.l2(w) if w != 0 else None for w in self.wd)
        if self.regularizer[0] is None and self.regularizer[1] is None:
            self.regularizer = None
        self.regulate_vars = [[], []]

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            'seed': self.seed,
            'dims': tuple(self.dims),
            'wd': tuple(self.wd)
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'seed', 'dims', 'wd']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def build(self, input_shape):
        assert input_shape[0][-1] == input_shape[1][-1], 'special types of rnn outputs and attention inputs must ' \
                                               'match with each other'
        self.win_query_w = self.add_weight(name='window_attention_query_weights',
                                           shape=(input_shape[1][-3], input_shape[0][-1], input_shape[0][-2],
                                                  self.dims[0]),
                                           dtype=self.dtype,
                                           initializer=MyGlorotInitializer(self.seed, 'normal', -2, -1),
                                           regularizer=
                                           self.regularizer[0] if hasattr(self.regularizer, '__iter__') else None,
                                           trainable=True,
                                           constraint=None)
        self.regulate_vars[0].append(self.win_query_w)
        self.win_key_w = self.add_weight(name='window_attention_key_weights',
                                         shape=(input_shape[1][-3], input_shape[1][-1], input_shape[1][-2],
                                                self.dims[0]),
                                         dtype=self.dtype,
                                         initializer=MyGlorotInitializer(self.seed, 'normal', -2, -1),
                                         regularizer=
                                         self.regularizer[0] if hasattr(self.regularizer, '__iter__') else None,
                                         trainable=True,
                                         constraint=None)
        self.regulate_vars[0].append(self.win_key_w)
        self.win_key_b = self.add_weight(name='window_attention_key_bias',
                                         shape=(input_shape[1][-3], input_shape[1][-1], 1, self.dims[0]),
                                         dtype=self.dtype,
                                         initializer=keras.initializers.RandomNormal(
                                             seed=self.seed, stddev=sqrt(2. / (input_shape[1][-2] + self.dims[0]))),
                                         regularizer=
                                         self.regularizer[0] if hasattr(self.regularizer, '__iter__') else None,
                                         trainable=True,
                                         constraint=None)
        self.regulate_vars[0].append(self.win_key_b)
        self.ch_query_w = self.add_weight(name='channel_attention_query_weights',
                                          shape=(input_shape[0][-1], input_shape[0][-2], self.dims[1]),
                                          dtype=self.dtype,
                                          initializer=MyGlorotInitializer(self.seed, 'normal', -2, -1),
                                          regularizer=
                                          self.regularizer[1] if hasattr(self.regularizer, '__iter__') else None,
                                          trainable=True,
                                          constraint=None)
        self.regulate_vars[1].append(self.ch_query_w)
        self.ch_key_w = self.add_weight(name='channel_attention_key_weights',
                                        shape=(input_shape[1][-1], input_shape[1][-2], self.dims[1]),
                                        dtype=self.dtype,
                                        initializer=MyGlorotInitializer(self.seed, 'normal', -2, -1),
                                        regularizer=
                                        self.regularizer[1] if hasattr(self.regularizer, '__iter__') else None,
                                        trainable=True,
                                        constraint=None)
        self.regulate_vars[1].append(self.ch_key_w)
        self.ch_key_b = self.add_weight(name='channel_attention_key_bias',
                                        shape=(input_shape[1][-1], 1, self.dims[1]),
                                        dtype=self.dtype,
                                        initializer=keras.initializers.RandomNormal(
                                            seed=self.seed, stddev=sqrt(2. / (input_shape[1][-2] + self.dims[1]))),
                                        regularizer=
                                        self.regularizer[1] if hasattr(self.regularizer, '__iter__') else None,
                                        trainable=True,
                                        constraint=None)
        self.regulate_vars[1].append(self.ch_key_b)
        self.built = True

    def regu(self):
        if self.regularizer is not None:
            r = []
            for i, reg in enumerate(self.regularizer):
                if reg is not None:
                    for v in self.regulate_vars[i]:
                        r.append(tf.clip_by_value(reg(v), 1e-8, np.Inf))
            return r
        return []

    def get_input(self, batch_size, dims1, wins, chs, dims2, features):
        inputs = (tf.zeros((batch_size, dims1, features), dtype=self.dtype),
                  tf.zeros((batch_size, wins, chs, dims2, features), dtype=self.dtype))
        no_begin = None
        mask = tf.constant([True] * batch_size, dtype='bool')
        return inputs, no_begin, mask

    # inputs ---- (output_vector, primary_features)
    #              former shape: (step_batch_size, o_dims, f); latter shape: (step_batch_size, wins, chs, dims, f)
    # outputs --- chs_focus:(batch_size, chs, f) wins_focus:(batch_size, wins, chs, f) synthesis:(batch_size, dims, f)
    # no_begin indicates whether needs to use self-similarity  --- None or 1-D tensor_(bool)
    #@tf.function
    def call(self, inputs, no_begin=None, mask=None):
        inputs = (tf.cast(inputs[0], self.dtype), tf.cast(inputs[1], self.dtype))
        if mask is not None and not tf.reduce_all(mask):
            inputs = (tf.multiply(inputs[0], tf.cast(tf.reshape(mask, mask.shape + (1,) * 2), self.dtype)),
                      tf.multiply(inputs[1], tf.cast(tf.reshape(mask, mask.shape + (1,) * 4), self.dtype)))
        # ValueError: Length of branch outputs of cond must match ???
        #if no_begin is not None and not tf.reduce_any(no_begin):
        #    no_begin = None

        # (batch, chs, f, wins, 1)    (batch, f, chs, 1)
        def _compute_foreign_simi(w, key, q):     # (batch_size, chs, f, wins, dims)  (batch_size, f, chs, dims)
            # (chs, f, batch_size, dims)  (f, batch_size, dims)
            query = tf.matmul(tf.transpose(q, (2, 0, 1)), w)
            dims = tuple(range(query.shape.ndims))
            # (batch_size, chs, f, dims)  (batch_size, f, dims)
            query = tf.transpose(query, (dims[-2],) + dims[:-2] + dims[-1:])
            # (batch_size, chs, f, dims, 1)  (batch_size, f, dims, 1)
            query = tf.expand_dims(query, axis=-1)
            return tf.divide(tf.matmul(key, query),
                             tf.sqrt(tf.constant(key.shape[-1], dtype=self.dtype)))

        # (batch, chs, f, wins, 1)   (batch, f, chs, 1)
        # 不算正经自注意力
        def _compute_self_simi(key):
            return tf.divide(tf.reduce_sum(tf.multiply(key, key), axis=-1, keepdims=True),
                             tf.sqrt(tf.constant(key.shape[-1], dtype=self.dtype)))

        def _compute_simi(*args, expand):
            if no_begin is None:
                return _compute_foreign_simi(*args, inputs[0])
            else:
                if tf.reduce_all(no_begin):
                    return _compute_self_simi(args[-1])
                else:
                    for_foreign = tf.cast(tf.logical_not(no_begin), self.dtype)
                    foreign_inp = tf.multiply(inputs[0], tf.reshape(for_foreign, for_foreign.shape + (1,) * 2))
                    for_self = tf.cast(no_begin, self.dtype)
                    exp_d = args[-1].shape.ndims - for_self.shape.ndims
                    foreign = _compute_foreign_simi(*args, foreign_inp)
                    foreign = tf.multiply(foreign, tf.reshape(for_foreign, for_foreign.shape + (1,) * expand))
                    oneself = _compute_self_simi(tf.multiply(args[-1], tf.reshape(for_self,
                                                                                  for_self.shape + (1,) * exp_d)))
                return foreign + oneself

        # (batch_size, chs, f, wins, dims)
        win_key = tf.tanh(tf.matmul(tf.transpose(inputs[1], (0, 2, 4, 1, 3)), self.win_key_w) + self.win_key_b)
        # (batch_size, chs, f, wins, 1)
        win_similarity = _compute_simi(self.win_query_w, win_key, expand=4)
        # (batch_size, wins, chs, 1, f)
        win_attention = tf.transpose(tf.nn.softmax(win_similarity, axis=-2), (0, 3, 1, 4, 2))
        # (batch_size, wins, chs, f)
        # focus_wins = tf.argsort(win_attention, axis=1, direction='DESCENDING')[:, :, :, 0, :] --- result is index 'int32'
        focus_wins = tf.stop_gradient(tf.squeeze(win_attention, axis=-2))
        # (batch_size, chs, dims, f)
        win_synthesis = tf.reduce_sum(tf.multiply(win_attention, inputs[1]), axis=1)

        # (batch_size, f, chs, dims)
        ch_key = tf.tanh(tf.matmul(tf.transpose(win_synthesis, (0, 3, 1, 2)), self.ch_key_w) + self.ch_key_b)
        # (batch_size, f, chs, 1)
        ch_similarity = _compute_simi(self.ch_query_w, ch_key, expand=3)
        # (batch_size, chs, 1, f)
        ch_attention = tf.transpose(tf.nn.softmax(ch_similarity, axis=-2), (0, 2, 3, 1))
        # (batch_size, chs, f)
        # focus_chs = tf.argsort(ch_attention, axis=1, direction='DESCENDING')[:, :, 0, :]
        focus_chs = tf.stop_gradient(tf.squeeze(ch_attention, axis=-2))
        # (batch_size, dims, f)
        synthesis = tf.reduce_sum(tf.multiply(ch_attention, win_synthesis), axis=1)

        if mask is not None and not tf.reduce_all(mask):
            exp_mask = tf.cast(tf.reshape(mask, mask.shape + (1,) * 2), self.dtype)
            synthesis = tf.multiply(synthesis, exp_mask)
            focus_chs = tf.multiply(focus_chs, exp_mask)
            exp_mask = tf.expand_dims(exp_mask, axis=-1)
            focus_wins = tf.multiply(focus_wins, exp_mask)

        return focus_chs, focus_wins, synthesis


class SelfWindowAttention(keras.layers.Layer):
    def __init__(self, name, seed, dims, dtype=tf.float32, wd=0.001):
        super(SelfWindowAttention, self).__init__(name=name, dtype=dtype)
        self.dims = dims
        self.seed = seed
        self.wd = wd
        if self.wd:
            self.regularizer = keras.regularizers.l2(self.wd)
        else:
            self.regularizer = None
        self.regulate_vars = []

    def get_config(self):
        config = super(SelfWindowAttention, self).get_config()
        config.update({
            'seed': self.seed,
            'dims': self.dims,
            'wd': self.wd
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'seed', 'dims', 'wd']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def build(self, input_shape):
        self.query = self.add_weight(name='query',
                                     shape=(input_shape[-1], input_shape[-3], self.dims, 1),
                                     dtype=self.dtype,
                                     initializer=keras.initializers.RandomNormal(
                                         seed=self.seed, stddev=sqrt(2. / (input_shape[-2] + self.dims))),
                                     regularizer=self.regularizer,
                                     trainable=True,
                                     constraint=None)
        self.regulate_vars.append(self.query)
        self.key_w = self.add_weight(name='key_weights',
                                     shape=(input_shape[-1], input_shape[-3], input_shape[-2], self.dims),
                                     dtype=self.dtype,
                                     initializer=MyGlorotInitializer(self.seed, 'normal', -2, -1),
                                     regularizer=self.regularizer,
                                     trainable=True,
                                     constraint=None)
        self.regulate_vars.append(self.key_w)
        self.key_b = self.add_weight(name='key_bias',
                                     shape=(input_shape[-1], input_shape[-3], 1, self.dims),
                                     dtype=self.dtype,
                                     initializer=keras.initializers.RandomNormal(
                                         seed=self.seed, stddev=sqrt(2. / (input_shape[-2] + self.dims))),
                                     regularizer=self.regularizer,
                                     trainable=True,
                                     constraint=None)
        self.regulate_vars.append(self.key_b)
        self.built = True

    def regu(self):
        if self.regularizer is not None:
            r = []
            for v in self.regulate_vars:
                r.append(tf.clip_by_value(self.regularizer(v), 1e-8, np.Inf))
            return r
        return []

    def get_input(self, batch_size, wins, chs, dims, features):
        inputs = tf.zeros((batch_size, wins, chs, dims, features), dtype=self.dtype)
        return inputs

    # inputs ------ before disassembled_tensor------(batch, wins, chs, xxx, f)
    # @tf.function
    def call(self, inputs):
        inputs = tf.cast(inputs, self.dtype)
        inp = tf.transpose(inputs, (0, 4, 2, 1, 3))
        # (batch, f, chs, wins, dims)
        key = tf.tanh(tf.matmul(inp, self.key_w) + self.key_b)
        alpha = tf.matmul(key, self.query)
        alpha = tf.nn.softmax(alpha, axis=-2)
        # shape (batch, f, wins, chs)
        focus_wins = tf.transpose(tf.squeeze(alpha, axis=-1), (0, 1, 3, 2))
        # shape (batch, chs, xxx, f)
        synthesis = tf.reduce_sum(tf.multiply(inputs, tf.transpose(alpha, (0, 3, 2, 4, 1))), axis=1)
        return focus_wins, synthesis


class SelfChannelAttention(keras.layers.Layer):
    def __init__(self, name, seed, dims, dtype=tf.float32, wd=0.001):
        super(SelfChannelAttention, self).__init__(name=name, dtype=dtype)
        self.dims = dims
        self.seed = seed
        self.wd = wd
        if self.wd:
            self.regularizer = keras.regularizers.l2(self.wd)
        else:
            self.regularizer = None
        self.regulate_vars = []

    def get_config(self):
        config = super(SelfChannelAttention, self).get_config()
        config.update({
            'seed': self.seed,
            'dims': self.dims,
            'wd': self.wd
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'seed', 'dims', 'wd']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def build(self, input_shape):
        self.query = self.add_weight(name='query',
                                     shape=(input_shape[-1], self.dims, 1),
                                     dtype=self.dtype,
                                     initializer=keras.initializers.RandomNormal(
                                         seed=self.seed, stddev=sqrt(2. / (input_shape[-2] + self.dims))),
                                     regularizer=self.regularizer,
                                     trainable=True,
                                     constraint=None)
        self.regulate_vars.append(self.query)
        self.key_w = self.add_weight(name='key_weights',
                                     shape=(input_shape[-1], input_shape[-2], self.dims),
                                     dtype=self.dtype,
                                     initializer=MyGlorotInitializer(self.seed, 'normal', -2, -1),
                                     regularizer=self.regularizer,
                                     trainable=True,
                                     constraint=None)
        self.regulate_vars.append(self.key_w)
        self.key_b = self.add_weight(name='key_bias',
                                     shape=(input_shape[-1], 1, self.dims),
                                     dtype=self.dtype,
                                     initializer=keras.initializers.RandomNormal(
                                         seed=self.seed, stddev=sqrt(2. / (input_shape[-2] + self.dims))),
                                     regularizer=self.regularizer,
                                     trainable=True,
                                     constraint=None)
        self.regulate_vars.append(self.key_b)
        self.built = True

    def regu(self):
        if self.regularizer is not None:
            r = []
            for v in self.regulate_vars:
                r.append(tf.clip_by_value(self.regularizer(v), 1e-8, np.Inf))
            return r
        return []

    def get_input(self, batch_size, chs, dims, features):
        inputs = tf.zeros((batch_size, chs, dims, features), dtype=self.dtype)
        return inputs

    # inputs ------ before disassembled_tensor------(batch, chs, xxx, f)
    # @tf.function
    def call(self, inputs):
        inputs = tf.cast(inputs, self.dtype)
        inp = tf.transpose(inputs, (0, 3, 1, 2))
        # (batch, f, chs, dims)
        key = tf.tanh(tf.matmul(inp, self.key_w) + self.key_b)
        alpha = tf.matmul(key, self.query)
        alpha = tf.nn.softmax(alpha, axis=-2)
        # shape (batch, f, chs)
        focus_chs = tf.squeeze(alpha, axis=-1)
        # shape (batch, xxx, f)
        synthesis = tf.reduce_sum(tf.multiply(inputs, tf.transpose(alpha, (0, 2, 3, 1))), axis=1)
        return focus_chs, synthesis


class SelfAttention(keras.layers.Layer):
    def __init__(self, name, seed, dims, dtype=tf.float32, wd=0.001):
        super(SelfAttention, self).__init__(name=name, dtype=dtype)
        self.seed = seed

        if isinstance(dims, int):
            self.dims = (dims,) * 2
        elif hasattr(dims, '__iter__') and len(dims) == 2:
            self.dims = tuple(dims)
        else:
            raise ValueError('parameter \'dims\' is unqualified')

        if isinstance(wd, float):
            self.wd = (wd,) * 2
        elif hasattr(wd, '__iter__') and len(wd) == 2:
            self.wd = tuple(wd)
        else:
            raise ValueError('parameter \'wd\' is unqualified')

        self.regularizer = list(keras.regularizers.l2(w) if w != 0 else None for w in self.wd)
        if self.regularizer[0] is None and self.regularizer[1] is None:
            self.regularizer = None
        self.regulate_vars = [[], []]

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({
            'seed': self.seed,
            'dims': tuple(self.dims),
            'wd': tuple(self.wd)
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'seed', 'dims', 'wd']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def build(self, input_shape):
        self.win_query = self.add_weight(name='window_query',
                                         shape=(input_shape[-1], input_shape[-3], self.dims[0], 1),
                                         dtype=self.dtype,
                                         initializer=keras.initializers.RandomNormal(
                                             seed=self.seed, stddev=sqrt(2. / (input_shape[-2] + self.dims[0]))),
                                         regularizer=
                                         self.regularizer[0] if hasattr(self.regularizer, '__iter__') else None,
                                         trainable=True,
                                         constraint=None)
        self.regulate_vars[0].append(self.win_query)
        self.win_key_w = self.add_weight(name='window_key_weights',
                                         shape=(input_shape[-1], input_shape[-3], input_shape[-2], self.dims[0]),
                                         dtype=self.dtype,
                                         initializer=MyGlorotInitializer(self.seed, 'normal', -2, -1),
                                         regularizer=
                                         self.regularizer[0] if hasattr(self.regularizer, '__iter__') else None,
                                         trainable=True,
                                         constraint=None)
        self.regulate_vars[0].append(self.win_key_w)
        self.win_key_b = self.add_weight(name='window_key_bias',
                                         shape=(input_shape[-1], input_shape[-3], 1, self.dims[0]),
                                         dtype=self.dtype,
                                         initializer=keras.initializers.RandomNormal(
                                             seed=self.seed, stddev=sqrt(2. / (input_shape[-2] + self.dims[0]))),
                                         regularizer=
                                         self.regularizer[0] if hasattr(self.regularizer, '__iter__') else None,
                                         trainable=True,
                                         constraint=None)
        self.regulate_vars[0].append(self.win_key_b)
        self.ch_query = self.add_weight(name='channel_query',
                                        shape=(input_shape[-1], self.dims[1], 1),
                                        dtype=self.dtype,
                                        initializer=keras.initializers.RandomNormal(
                                            seed=self.seed, stddev=sqrt(2. / (input_shape[-2] + self.dims[1]))),
                                        regularizer=
                                        self.regularizer[1] if hasattr(self.regularizer, '__iter__') else None,
                                        trainable=True,
                                        constraint=None)
        self.regulate_vars[1].append(self.ch_query)
        self.ch_key_w = self.add_weight(name='channel_key_weights',
                                        shape=(input_shape[-1], input_shape[-2], self.dims[1]),
                                        dtype=self.dtype,
                                        initializer=MyGlorotInitializer(self.seed, 'normal', -2, -1),
                                        regularizer=
                                        self.regularizer[1] if hasattr(self.regularizer, '__iter__') else None,
                                        trainable=True,
                                        constraint=None)
        self.regulate_vars[1].append(self.ch_key_w)
        self.ch_key_b = self.add_weight(name='channel_key_bias',
                                        shape=(input_shape[-1], 1, self.dims[1]),
                                        dtype=self.dtype,
                                        initializer=keras.initializers.RandomNormal(
                                            seed=self.seed, stddev=sqrt(2. / (input_shape[-2] + self.dims[1]))),
                                        regularizer=
                                        self.regularizer[1] if hasattr(self.regularizer, '__iter__') else None,
                                        trainable=True,
                                        constraint=None)
        self.regulate_vars[1].append(self.ch_key_b)
        self.built = True

    def regu(self):
        if self.regularizer is not None:
            r = []
            for i, reg in enumerate(self.regularizer):
                if reg is not None:
                    for v in self.regulate_vars[i]:
                        r.append(tf.clip_by_value(reg(v), 1e-8, np.Inf))
            return r
        return []

    def get_input(self, batch_size, wins, chs, dims, features):
        inputs = tf.zeros((batch_size, wins, chs, dims, features), dtype=self.dtype)
        return inputs

    # inputs ------ before disassembled_tensor------(batch, wins, chs, xxx, f)
    # @tf.function
    def call(self, inputs):
        inputs = tf.cast(inputs, self.dtype)

        inp = tf.transpose(inputs, (0, 4, 2, 1, 3))
        # (batch, f, chs, wins, dims)
        win_key = tf.tanh(tf.matmul(inp, self.win_key_w) + self.win_key_b)
        alpha = tf.matmul(win_key, self.win_query)
        alpha = tf.nn.softmax(alpha, axis=-2)
        # shape (batch, f, wins, chs)
        focus_wins = tf.transpose(tf.squeeze(alpha, axis=-1), (0, 1, 3, 2))
        # shape (batch, chs, xxx, f)
        win_synthesis = tf.reduce_sum(tf.multiply(inputs, tf.transpose(alpha, (0, 3, 2, 4, 1))), axis=1)

        win_inp = tf.transpose(win_synthesis, (0, 3, 1, 2))
        # (batch, f, chs, dims)
        ch_key = tf.tanh(tf.matmul(win_inp, self.ch_key_w) + self.ch_key_b)
        alpha = tf.matmul(ch_key, self.ch_query)
        alpha = tf.nn.softmax(alpha, axis=-2)
        # shape (batch, f, chs)
        focus_chs = tf.squeeze(alpha, axis=-1)
        # shape (batch, xxx, f)
        synthesis = tf.reduce_sum(tf.multiply(inputs, tf.transpose(alpha, (0, 2, 3, 1))), axis=1)
        return focus_chs, focus_wins, synthesis


class StepAttention(keras.layers.Layer):
    def __init__(self, name, seed, dims, refer, dtype=tf.float32, wd=0.001):
        super(StepAttention, self).__init__(name=name, dtype=dtype)
        self.dims = dims
        self.seed = seed
        self.refer = refer
        self.wd = wd
        if self.wd:
            self.regularizer = keras.regularizers.l2(self.wd)
        else:
            self.regularizer = None
        self.regulate_vars = []

    def get_config(self):
        config = super(StepAttention, self).get_config()
        config.update({
            'seed': self.seed,
            'dims': self.dims,
            'refer': self.refer,
            'wd': self.wd
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'seed', 'dims', 'refer', 'wd']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def build(self, input_shape):
        self.query = self.add_weight(name='query_weights',
                                     shape=(input_shape[-1], self.dims, 1),
                                     dtype=self.dtype,
                                     initializer=keras.initializers.RandomNormal(
                                         seed=self.seed, stddev=sqrt(2. / (input_shape[-2] + self.dims))),
                                     regularizer=self.regularizer,
                                     trainable=True,
                                     constraint=None)
        self.regulate_vars.append(self.query)
        self.key_w = self.add_weight(name='key_weights',
                                     shape=(input_shape[-1], input_shape[-2], self.dims),
                                     dtype=self.dtype,
                                     initializer=MyGlorotInitializer(self.seed, 'normal', -2, -1),
                                     regularizer=self.regularizer,
                                     trainable=True,
                                     constraint=None)
        self.regulate_vars.append(self.key_w)
        self.key_b = self.add_weight(name='key_bias',
                                     shape=(input_shape[-1], 1, self.dims),
                                     dtype=self.dtype,
                                     initializer=
                                     keras.initializers.RandomNormal(
                                         seed=self.seed, stddev=sqrt(2. / (input_shape[-2] + self.dims))),
                                     regularizer=None,
                                     trainable=True,
                                     constraint=None)
        self.regulate_vars.append(self.key_b)
        self.built = True

    def regu(self):
        if self.regularizer is not None:
            r = []
            for v in self.regulate_vars:
                r.append(tf.clip_by_value(self.regularizer(v), 1e-8, np.Inf))
            return r
        return []

    def get_input(self, steps, step_batch_size, dims, features):
        inputs = tf.zeros((steps, step_batch_size, dims, features), dtype=self.dtype)
        refer = None
        mask = tf.ones((steps, step_batch_size), dtype='bool')
        return inputs, refer, mask

    # inputs ---- (step, step_b_size, dims, f)
    # refer --- (self.refer, step_b_size, dims, f)
    # mask --- (step, step_b_size)
    #@tf.function
    def call(self, inputs, refer=None, mask=None):
        inputs = tf.cast(inputs, self.dtype)
        shape = inputs.shape
        if refer is not None:
            refer = tf.cast(refer, self.dtype)
        else:
            refer = tf.zeros((self.refer,) + inputs.shape[1:], dtype=self.dtype)
        tail = tf.zeros((self.refer,) + inputs.shape[1:], dtype=self.dtype)
        temp = tf.concat((refer, inputs, tail), axis=0)
        del inputs
        gc.collect()
        if mask is not None:
            ndims = shape.ndims - mask.shape.ndims
            exp_mask = tf.reshape(mask, mask.shape + (1,) * ndims)
            temp = tf.multiply(temp, exp_mask)
            new_refer = tf.stop_gradient(temp[-int(self.refer * 2): -self.refer])
        else:
            new_refer = tf.stop_gradient(temp[-int(self.refer * 2): -self.refer])
        ta = tf.TensorArray(self.dtype, size=shape[0])
        for i in range(inputs.shape[0]):
            ta = ta.write(i, temp[i: int(self.refer * 2 + 1 + i)])
        # shape (step, size, step_b_size, dims, f)
        temp = ta.stack()
        key = tf.tanh(tf.matmul(tf.transpose(temp, (0, 1, 4, 2, 3)), self.key_w) + self.key_b)
        # shape (step, 2 * self.refer + 1, f, step_b_size, 1)
        alpha = tf.nn.softmax(tf.matmul(key, self.query), axis=1)
        del key
        gc.collect()
        # shape (step, step_b_size, dims, f)
        temp = tf.reduce_sum(tf.multiply(temp, tf.transpose(alpha, (0, 1, 3, 4, 2))), axis=1)
        return temp, new_refer


class LN(keras.layers.Layer):
    def __init__(self, name, use_axes=None, except_axes=None, is_offset=True, is_scale=True, epislon=1e-4,
                 gamma_initializer=1, beta_initializer=0, dtype=tf.float32):
        super(LN, self).__init__(name=name, dtype=dtype)
        if use_axes is not None and except_axes is not None:
            raise Exception('Two types of axes can\'t be assigned simultaneously')
        if use_axes is None and except_axes is None:
            raise ValueError('need one of two types of axes')
        if use_axes is None:
            self.axes = except_axes     # int, list, tuple     input in ascending order
            self.axes_type = 'N'
        else:
            self.axes = use_axes
            self.axes_type = 'P'
        if isinstance(self.axes, int):
            self.axes = [self.axes]
        elif hasattr(self.axes, '__iter__'):
            self.axes = list(self.axes)
        else:
            raise TypeError('the parameter about axes must be iterative or integer')
        self.is_offset = is_offset
        self.is_scale = is_scale
        if epislon < 1.001e-5:
            self.epislon = 1.001e-5
        else:
            self.epislon = epislon
        self.gamma_initializer = gamma_initializer
        self.beta_initializer = beta_initializer

    def get_config(self):
        config = super(LN, self).get_config()
        if self.axes_type == 'P':
            config['use_axes'] = tuple(self.axes) if hasattr(self.axes, '__iter__') else self.axes
        else:
            config['except_axes'] = tuple(self.axes) if hasattr(self.axes, '__iter__') else self.axes
        config.update({
            'is_offset': self.is_offset,
            'is_scale': self.is_scale,
            'epislon': self.epislon,
            'gamma_initializer': self.gamma_initializer,
            'beta_initializer': self.beta_initializer
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'is_offset', 'is_scale', 'epislon', 'gamma_initializer', 'beta_initializer']
        new = kwargs.fromkeys(ks)
        new['use_axes'] = kwargs.get('use_axes', None)
        new['except_axes'] = kwargs.get('except_axes', None)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def build(self, input_shape):
        dims = input_shape.ndims
        if self.axes_type == 'P':
            for i, ax in enumerate(self.axes):
                if ax < 0:
                    ax = ax + dims
                if ax < 0 or ax >= dims:
                    raise ValueError('invalid ax: {}'.format(ax))
                self.axes[i] = ax
        else:
            tr = [i for i in range(dims)]
            for d in self.axes:
                if d < 0:
                    d += dims
                if d < 0 or d >= dims:
                    raise ValueError('invalid ax: {}'.format(d))
                tr.remove(d)
            self.axes = tr
            self.axes_type = 'P'
        axes_reserved = [input_shape[d] if d not in self.axes else 1 for d in range(dims)]
        self.gamma = self.add_weight(name='gamma',
                                     shape=axes_reserved,
                                     dtype=self.dtype,
                                     initializer=keras.initializers.Constant(self.gamma_initializer),
                                     regularizer=None,
                                     trainable=self.is_scale,
                                     constraint=None)
        self.beta = self.add_weight(name='beta',
                                    shape=axes_reserved,
                                    dtype=self.dtype,
                                    initializer=keras.initializers.Constant(self.beta_initializer),
                                    regularizer=None,
                                    trainable=self.is_offset,
                                    constraint=None)
        self.built = True

    @tf.function
    def call(self, inputs, mask=None):     # (batch_size, ...)
        inputs = tf.cast(inputs, self.dtype)
        # applicable situation: difference only alis in batch_size supposing
        if mask is not None:
            exp_dims = inputs.shape.ndims - mask.shape.ndims
            exp_msk = tf.reshape(mask, mask.shape + (1,) * exp_dims)
            inp = tf.multiply(inputs, tf.cast(exp_msk, self.dtype))
            mean, variable = tf.nn.moments(inp, self.axes, keepdims=True)
            result = tf.nn.batch_normalization(inp, mean, variable, self.beta, self.gamma,
                                               self.epislon) + inputs * tf.cast(tf.logical_not(exp_msk), self.dtype)
        else:
            mean, variable = tf.nn.moments(inputs, self.axes, keepdims=True)
            result = tf.nn.batch_normalization(inputs, mean, variable, self.beta, self.gamma, self.epislon)

        return result


class MyGRUCell(keras.layers.AbstractRNNCell):
    def __init__(self, name, hidden_units, seed, dtype=tf.float32, wd=0.001, input_dropout=0., hidden_dropout=0.,
                 ln_is_scale=True, ln_is_offset=False):
        super(MyGRUCell, self).__init__(name=name, dtype=dtype)
        self.units = hidden_units
        self.seed = seed
        self.wd = wd
        self.input_dropout = min(1., max(0., input_dropout))   # input use
        self.hidden_dropout = min(1., max(0., hidden_dropout))    # hidden state use
        if isinstance(ln_is_scale, bool):
            self.is_scale = (ln_is_scale,) * 3
        elif hasattr(ln_is_scale, '__iter__') and len(ln_is_scale) == 3:
            self.is_scale = tuple(ln_is_scale)
        else:
            raise TypeError('parameter \'ln_is_scale\' must be iterative of 3 elements or a bool value')
        if isinstance(ln_is_offset, bool):
            self.is_offset = (ln_is_offset,) * 3
        elif hasattr(ln_is_offset, '__iter__') and len(ln_is_offset) == 3:
            self.is_offset = tuple(ln_is_offset)
        else:
            raise TypeError('parameter \'ln_is_offset\' must be iterative of 3 elements or a bool value')
        # (step_ba, xxx, f) ---> (step_ba, f, xxx)
        self.u_ln = LN(self.name + '/update_gate_ln', except_axes=[0, 1], is_scale=self.is_scale[0],
                       is_offset=self.is_offset[0], dtype=self.dtype)
        self.r_ln = LN(self.name + '/reset_gate_ln', except_axes=[0, 1], is_scale=self.is_scale[1],
                       is_offset=self.is_offset[1], dtype=self.dtype)
        self.delta_ln = LN(self.name + '/delta_ln', except_axes=[0, 1], is_scale=self.is_scale[2],
                           is_offset=self.is_offset[2], dtype=self.dtype)
        self.sublayer_num = 3
        self.__input_dropout_mask = None
        self.__hidden_dropout_mask = None
        if self.wd:
            self.regularizer = keras.regularizers.l2(self.wd)
        else:
            self.regularizer = None
        self.regulate_vars = []

    def get_config(self):
        config = super(MyGRUCell, self).get_config()
        config.update({
            'hidden_units': self.units,
            'seed': self.seed,
            'wd': self.wd,
            'input_dropout': self.input_dropout,
            'hidden_dropout': self.hidden_dropout,
            'ln_is_scale': tuple(self.is_scale),
            'ln_is_offset': tuple(self.is_offset)
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'hidden_units', 'seed', 'wd', 'input_dropout', 'hidden_dropout', 'ln_is_scale',
              'ln_is_offset']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    @property
    def state_size(self):
        return [self.units]

    @property
    def output_size(self):
        return [self.units]

    def reset_input_dropout_mask(self):
        self.__input_dropout_mask = None

    def reset_hidden_dropout_mask(self):
        self.__hidden_dropout_mask = None

    @property
    def input_dropout_mask(self):
        return self.__input_dropout_mask

    @property
    def hidden_dropout_mask(self):
        return self.__hidden_dropout_mask

    def get_input_dropout_mask(self, inputs, training):
        if self.__input_dropout_mask is None:      # because use self.input_dropout to generate mask, can reuse
            ones = tf.ones((3,) + inputs.shape, dtype=inputs.dtype)
            self.__input_dropout_mask = tf.cond(training, lambda: tf.nn.dropout(ones, self.input_dropout,
                                                                                seed=np.random.randint(1e7)),
                                                lambda: ones)
        return self.__input_dropout_mask

    def get_hidden_dropout_mask(self, inputs, training):     # inputs is a Tensor
        if self.__hidden_dropout_mask is None:
            ones = tf.ones((3,) + inputs.shape, dtype=inputs.dtype)
            self.__hidden_dropout_mask = tf.cond(training, lambda: tf.nn.dropout(ones, self.input_dropout,
                                                                                 seed=np.random.randint(1e7)),
                                                 lambda: ones)
        return self.__hidden_dropout_mask

    def regu(self):
        if self.regularizer is not None:
            r = []
            for v in self.regulate_vars:
                r.append(tf.clip_by_value(self.regularizer(v), 1e-8, np.Inf))
            return r
        return []

    def build(self, input_shape):
        self.kernel = self.add_weight(name='input_weights',
                                      shape=(3, input_shape[-1], input_shape[-2], self.units),    # update_gate, reset_gate, delta_h
                                      dtype=self.dtype,
                                      initializer=MyGlorotInitializer(self.seed, 'normal', -2, -1),
                                      regularizer=self.regularizer,
                                      trainable=True,
                                      constraint=None)
        self.regulate_vars.append(self.kernel)
        self.recurrent_kernel = self.add_weight(name='hidden_weights',
                                                shape=(3, input_shape[-1], self.units, self.units),    # update_gate, reset_gate, delta_hidden
                                                dtype=self.dtype,
                                                initializer=MyOrthogonalInitializer(self.seed),
                                                regularizer=self.regularizer,
                                                trainable=True,
                                                constraint=None)
        self.regulate_vars.append(self.recurrent_kernel)
        self.input_bias = self.add_weight(name='input_bias',
                                          shape=(3, input_shape[-1], 1, self.units),
                                          dtype=self.dtype,
                                          initializer=keras.initializers.get('zeros'),
                                          regularizer=None,
                                          trainable=True,
                                          constraint=None)
        self.recurrent_bias = self.add_weight(name='hidden_bias',
                                              shape=(3, input_shape[-1], 1, self.units),
                                              dtype=self.dtype,
                                              initializer=keras.initializers.get('zeros'),
                                              regularizer=None,
                                              trainable=True,
                                              constraint=None)
        self.built = True

    def get_input(self, step_batch_size, features):
        inputs = tf.zeros((step_batch_size, self.units, features), dtype=self.dtype)
        states = tf.zeros((step_batch_size, self.units, features), dtype=self.dtype)
        mask = tf.constant([True] * step_batch_size, dtype='bool')
        return inputs, states, mask

    def get_initial_state(self, inputs):    # inputs, batch_size, dtype
        with tf.init_scope():
            batch_size = inputs.shape[0]
            dtype = self.dtype
            features = inputs.shape[-1]
            return tf.zeros((batch_size, self.units, features), dtype)

    def get_initial_output(self, inputs):    # inputs, batch_size, dtype
        with tf.init_scope():
            batch_size = inputs.shape[0]
            dtype = self.dtype
            features = inputs.shape[-1]
            return tf.zeros((batch_size, self.units, features), dtype)

    # one subject data remains the part which can't exact divide step_batch_size*steps, its step_batch_size is 1
    # (get previous step's end state)
    # inputs ----- cell_input---(step_batch_size, features, 1/3), states ----- last_state---(step_batch_size, hidden_units, 1/3))
    # note that first step may have no last state
    @tf.function
    def call(self, inputs, states, training=None, mask=None):
        inputs = tf.cast(inputs, self.dtype)
        states = tf.cast(states, self.dtype)
        if mask is not None and not tf.reduce_all(mask):
            inputs = tf.multiply(inputs, tf.reshape(tf.cast(mask, self.dtype), mask.shape + (1,) * 2))
            states = tf.multiply(states, tf.reshape(tf.cast(mask, self.dtype), mask.shape + (1,) * 2))
        exp_inputs = tf.stack([inputs for _ in range(3)], axis=0)
        if self.input_dropout > 0.:
            inputs_mask = tf.stop_gradient(tf.py_function(self.get_input_dropout_mask, inp=[inputs, training],
                                                          Tout=inputs.dtype))
            inputs_mask.set_shape((3,) + inputs.shape)
            exp_inputs = tf.multiply(inputs_mask, exp_inputs)
        if states is None:
            states = self.get_initial_state(inputs)
        exp_states = tf.stack([states for _ in range(3)], axis=0)
        if self.hidden_dropout > 0.:
            hidden_mask = tf.stop_gradient(tf.py_function(self.get_hidden_dropout_mask, inp=[states, training],
                                                          Tout=states.dtype))
            hidden_mask.set_shape((3,) + states.shape)
            exp_states = tf.multiply(hidden_mask, exp_states)

        # (3, 1/3, batch_size, hidden_units)
        xh = tf.add(tf.matmul(tf.transpose(exp_inputs, (0, 3, 1, 2)), self.kernel), self.input_bias)
        hh = tf.add(tf.matmul(tf.transpose(exp_states[:2], (0, 3, 1, 2)), self.recurrent_kernel[:2]),
                    self.recurrent_bias[:2])
        update_gate = tf.sigmoid(self.u_ln(tf.transpose(xh[0] + hh[0], (1, 0, 2)), mask))
        reset_gate = tf.sigmoid(self.r_ln(tf.transpose(xh[1] + hh[1], (1, 0, 2)), mask))
        exp_states = tf.transpose(exp_states, (0, 3, 1, 2))
        r_contents = tf.multiply(exp_states[2], tf.transpose(reset_gate, (1, 0, 2)))
        delta_hh = tf.add(tf.matmul(r_contents, self.recurrent_kernel[2]), self.recurrent_bias[2])
        delta_hidden = tf.tanh(self.delta_ln(tf.transpose(xh[2] + delta_hh, (1, 0, 2)), mask))
        hidden = tf.add(tf.multiply(update_gate, tf.transpose(exp_states[2], (1, 0, 2))),
                        tf.multiply(tf.subtract(tf.cast(1., self.dtype), update_gate), delta_hidden))
        hidden = tf.transpose(hidden, (0, 2, 1))
        output = tf.tanh(hidden)
        if mask is not None and not tf.reduce_all(mask):
            exp_mask = tf.cast(tf.reshape(mask, mask.shape + (1,) * 2), inputs.dtype)
            output = tf.multiply(output, exp_mask)
            hidden = tf.multiply(hidden, exp_mask)
        return output, hidden


class SingleLayerBiGRU(keras.layers.Layer):
    def __init__(self, name, prefix, maxlen, h_units, seed, merge_mode='sum', dtype=tf.float32, wd=0.001,
                 cell_i_dropout=0., cell_h_dropout=0.):
        '''

        :param name:
        :param prefix:
        :param maxlen: the maximum steps in the sequence
        :param h_units: int
        :param glorot_seed: used for initializer
        :param merge_mode: str, one in ('sum', 'ave', 'mul', 'concat', None), used for concatenating two opposite
                           directions's results
        :param dtype:
        :param wd:
        :param cell_i_dropout: float, tuple or list, if belongs to the last two, its length must be equal
                               to 2. used for input dropout in GRU cell
        :param cell_h_dropout: the same as above. used for the previous hidden state dropout in GRU cell
        '''
        # although has no weights in this layer, its children layers are trainable, so can't set 'trainable' to be False
        super(SingleLayerBiGRU, self).__init__(name=name, dtype=dtype)
        self.supports_masking = True
        self.prefix = prefix
        self.maxlen = maxlen
        self.h_units = h_units   # int
        self.seed = seed
        self.merge_mode = merge_mode   # one in ('sum', 'ave', 'mul', 'concat', None)
        if self.merge_mode not in ('sum', 'ave', 'mul', 'concat', None):
            raise ValueError('can\'t recognize \'merge_mode\' passed ')
        self.wd = wd
        self.c_i_dropout = cell_i_dropout  # float, tuple, list
        self.c_h_dropout = cell_h_dropout  # float, tuple, list
        if isinstance(self.c_i_dropout, float):
            self.c_i_dropout = [self.c_i_dropout] * 2
        if isinstance(self.c_h_dropout, float):
            self.c_h_dropout = [self.c_h_dropout] * 2
        if len(self.c_h_dropout) != 2 or len(self.c_i_dropout) != 2:
            raise ValueError('information don\'t match with the two layers with opposite directions')
        # +
        self.b_cell = MyGRUCell(prefix + 'backward_GRUCell', self.h_units, self.seed, dtype=self.dtype,
                                wd=self.wd, input_dropout=self.c_i_dropout[0], hidden_dropout=self.c_h_dropout[0])
        # reverse
        self.f_cell = MyGRUCell(prefix + 'forward_GRUCell', self.h_units, self.seed, dtype=self.dtype,
                                wd=self.wd, input_dropout=self.c_i_dropout[1], hidden_dropout=self.c_h_dropout[1])
        self.sublayer_num = 2

    def regu(self):
        return self.b_cell.regu() + self.f_cell.regu()

    def get_input(self, step_batch_size, dims, features):
        inputs = tf.zeros((self.maxlen, step_batch_size, dims, features), dtype=self.dtype)
        mask = tf.ones((self.maxlen, step_batch_size), dtype='bool')
        initial_states = None
        return inputs, initial_states, mask

    def get_config(self):
        config = super(SingleLayerBiGRU, self).get_config()
        config.update({
            'prefix': self.prefix,
            'maxlen': self.maxlen,
            'h_units': self.h_units,
            'seed': self.seed,
            'merge_mode': self.merge_mode,
            'wd': self.wd,
            'cell_i_dropout': tuple(self.c_i_dropout) if hasattr(self.c_i_dropout, '__iter__') else self.c_i_dropout,
            'cell_h_dropout': tuple(self.c_h_dropout) if hasattr(self.c_h_dropout, '__iter__') else self.c_h_dropout
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'prefix', 'maxlen', 'h_units', 'seed', 'merge_mode', 'cell_i_dropout', 'wd',
              'cell_h_dropout']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def get_initial_output(self, inputs):
        if self.merge_mode in ('sum', 'ave', 'mul'):
            return self.b_cell.get_initial_output(inputs[0])
        elif self.merge_mode == 'concat':
            return tf.concat((self.b_cell.get_initial_output(inputs[0]),
                              self.f_cell.get_initial_output(inputs[-1])), axis=1)
        else:
            b = tf.unstack(self.b_cell.get_initial_output(inputs[0]), axis=0)
            f = tf.unstack(self.f_cell.get_initial_output(inputs[-1]), axis=0)
            return tuple((_b, _f) for _b, _f in zip(b, f))

    # inputs---(steps, step_batch_size, features, 1/3)  initial_state---(b_state, f_state)
    def call(self, inputs, initial_states=None, mask=None, training=None):
        if self.maxlen < inputs.shape[0]:
            raise ValueError('the number of steps of data input is larger than max length set')
        steps = inputs.shape[0]

        tf.py_function(self.b_cell.reset_input_dropout_mask, inp=(), Tout=[])
        tf.py_function(self.b_cell.reset_hidden_dropout_mask, inp=(), Tout=[])
        tf.py_function(self.f_cell.reset_input_dropout_mask, inp=(), Tout=[])
        tf.py_function(self.f_cell.reset_hidden_dropout_mask, inp=(), Tout=[])

        inputs = tf.cast(inputs, self.dtype)
        inputs_r = inputs[::-1]
        if (hasattr(initial_states, '__iter__') and len(initial_states) != 2) or \
                (not hasattr(initial_states, '__iter__') and initial_states is not None):
            raise ValueError('not desired \'initial_states\'')
        # NoneType not be supported in graph, so must give value first
        if initial_states is None or (initial_states[0] is None and initial_states[1] is None):
            initial_states = (self.b_cell.get_initial_state(inputs[0]), self.f_cell.get_initial_state(inputs[-1]))
        elif initial_states[0] is None:
            initial_states = (self.b_cell.get_initial_state(inputs[0]), initial_states[1])
        elif initial_states[1] is None:
            initial_states = (initial_states[0], self.f_cell.get_initial_state(inputs[-1]))
        initial_states = (tf.cast(initial_states[0], self.dtype), tf.cast(initial_states[1], self.dtype))

        if mask is not None:
            m_aligning = inputs.shape.ndims - mask.shape.ndims
            exp_mask = tf.cast(tf.reshape(mask, mask.shape + (1,) * m_aligning), self.dtype)

        # in tf control flow, python collections can't change length, must be fixed structure(tensor shape, dtype)
        #@tf.function
        def _loop():
            states = tf.TensorArray(inputs.dtype, size=2)
            states = states.write(0, initial_states[0])
            states = states.write(1, initial_states[1])
            backward = tf.TensorArray(inputs.dtype, size=steps)
            forward = tf.TensorArray(inputs.dtype, size=steps)
            last_output = (self.b_cell.get_initial_output(inputs[0]), self.f_cell.get_initial_output(inputs[-1]))
            if mask is not None:
                mask_r = mask[::-1]
                exp_mask_r = exp_mask[::-1]
                aux_1 = tf.cast(1, inputs.dtype)
                for step in tf.range(steps):
                    last = (states.read(0), states.read(1))
                    o_b, s_b = self.b_cell(inputs[step], last[0], training, mask[step])
                    o_f, s_f = self.f_cell(inputs_r[step], last[1], training, mask_r[step])
                    # state/output is passed down
                    states = states.write(0, s_b + tf.multiply(last[0], aux_1 - exp_mask[step]))
                    states = states.write(1, s_f + tf.multiply(last[1], aux_1 - exp_mask_r[step]))
                    last_output = (o_b + tf.multiply(last_output[0], aux_1 - exp_mask[step]),
                                   o_f + tf.multiply(last_output[1], aux_1 - exp_mask_r[step]))
                    backward = backward.write(step, last_output[0])
                    forward = forward.write(step, last_output[1])
            else:
                for step in tf.range(steps):
                    o_b, s_b = self.b_cell(inputs[step], states.read(0), training)
                    o_f, s_f = self.f_cell(inputs_r[step], states.read(1), training)
                    states = states.write(0, s_b)
                    states = states.write(1, s_f)
                    backward = backward.write(step, o_b)
                    forward = forward.write(step, o_f)
            # TensorArray object can't be returned?
            states = (tf.stop_gradient(states.read(0)), tf.stop_gradient(states.read(1)))
            return backward.stack(), forward.stack()[::-1], states

        backward_o, forward_o, end_states = _loop()
        end_outputs = (tf.stop_gradient(backward_o[-1]), tf.stop_gradient(forward_o[0]))
        if mask is not None:
            backward_o = tf.multiply(backward_o, exp_mask)
            forward_o = tf.multiply(forward_o, exp_mask)

        def _combine():
            if self.merge_mode == 'sum':
                return tf.add(backward_o, forward_o)
            elif self.merge_mode == 'ave':
                return tf.multiply(tf.add(backward_o, forward_o), 0.5)
            elif self.merge_mode == 'mul':
                return tf.multiply(backward_o, forward_o)
            elif self.merge_mode == 'concat':
                return tf.concat((backward_o, forward_o), axis=-2)
            else:
                b = tf.unstack(backward_o, axis=0)
                f = tf.unstack(forward_o, axis=0)
                c = tuple((i, j) for i, j in zip(b, f))
                return c

        # first output has attribute '_keras_mask'， end_states(b_state, f_state)
        # when the end states are used as next sequence's initial states, must stop gradients
        return _combine(), end_states, end_outputs


class SingleBiGRUPlusAttention(keras.layers.Layer):
    def __init__(self, name, prefix, maxlen, seed, h_units, a_units, merge_mode='sum', a_wd=0.001,
                 dtype=tf.float32, h_wd=0.001, cell_i_dropout=0., cell_h_dropout=0.):
        '''

        :param name:
        :param prefix:
        :param maxlen: the maximum steps in the sequence
        :param seed: used for initializer
        :param h_units: int
        :param a_units: int, tuple/list or nested tuple/list, if the situation is the latter, its length must be 2
                        and each one is 2-element tuple/list, if the situation is the middle one, its length must
                        be 2 (for window and channel attention separately) , then that indicating the same set of
                        two direction stack rnn
        :param merge_mode: str, n ('sum', 'ave', 'mul', 'concat', None), used for concatenating two opposite
                           directions's results
        :param a_wd: whether to use weights decay in attention layer. float, tuple/list , or nested tuple/list, the
                      same as 'a_units'
        :param dtype:
        :param h_wd:
        :param cell_i_dropout: float, tuple or list, if belongs to the last two, its length must be equal  2.
                                used for input dropout in GRU cell
        :param cell_h_dropout: the same as above. used for the previous hidden state dropout in GRU cell
        '''
        super(SingleBiGRUPlusAttention, self).__init__(name=name, dtype=dtype)
        self.supports_masking = True
        self.prefix = prefix
        self.maxlen = maxlen
        self.seed = seed
        self.h_units = h_units
        self.merge_mode = merge_mode
        self.h_wd = h_wd
        self.c_i_dropout = cell_i_dropout
        self.c_h_dropout = cell_h_dropout
        if isinstance(self.c_i_dropout, float):
            self.c_i_dropout = [self.c_i_dropout] * 2
        if isinstance(self.c_h_dropout, float):
            self.c_h_dropout = [self.c_h_dropout] * 2
        if len(self.c_h_dropout) != 2 or len(self.c_i_dropout) != 2:
            raise ValueError('per cell dropout information don\'t match with the two opposite directions')
        self.b_cell = MyGRUCell(prefix + 'backward_GRUCell', self.h_units, self.seed, dtype=self.dtype,
                                wd=self.h_wd, input_dropout=self.c_i_dropout[0], hidden_dropout=self.c_h_dropout[0])
        # reverse
        self.f_cell = MyGRUCell(prefix + 'forward_GRUCell', self.h_units, self.seed, dtype=self.dtype,
                                wd=self.h_wd, input_dropout=self.c_i_dropout[1], hidden_dropout=self.c_h_dropout[1])
        # handle attention (attention receives the first BiGRU layer's states, then acts on the first BiGRU layer)
        # (as for single direction stack GRU, can receive the top GRU layer's state)
        if isinstance(a_units, int):
            self.a_units = [(a_units,) * 2] * 2
        elif hasattr(a_units, '__iter__') and len(a_units) == 2:   # supposing each element has the same form
            if a_units[0].__class__ == a_units[1].__class__:
                if isinstance(a_units[0], int):
                    self.a_units = [tuple(a_units)] * 2
                elif hasattr(a_units[0], '__iter__') and len(a_units[0]) == 2:
                    self.a_units = list(tuple(k) for k in a_units)
                else:
                    raise ValueError('can\'t recognized \'a_units\' passed')
            else:
                raise TypeError('the two elements must have the same type')
        else:
            raise ValueError('can\'t recognized \'a_units\' passed')

        if isinstance(a_wd, float):
            self.a_wd = [(a_wd,) * 2] * 2
        elif hasattr(a_wd, '__iter__') and len(a_wd) == 2:
            if isinstance(a_wd[0], type(a_wd[1])):
                if isinstance(a_wd[0], float):
                    self.a_wd = [tuple(a_wd)] * 2
                elif hasattr(a_wd[0], '__iter__') and len(a_wd[0]) == 2:
                    self.a_wd = list(tuple(k) for k in a_wd)
                else:
                    raise ValueError('can\'t recognized \'a_wd\' passed')
            else:
                raise TypeError('the two elements must have the same type')
        else:
            raise ValueError('can\'t recognized \'a_wd\' passed')

        self.attention_b = Attention(prefix + 'attention_forward', self.seed, self.a_units[0], self.dtype, self.a_wd[0])
        self.attention_f = Attention(prefix + 'attention_reverse', self.seed, self.a_units[1], self.dtype, self.a_wd[1])
        self.sublayer_num = 4

    def regu(self):
        return self.b_cell.regu() + self.f_cell.regu() + self.attention_b.regu() + self.attention_f.regu()

    def get_input(self, step_batch_size, wins, chs, dims, features):
        inputs = tf.zeros((self.maxlen, step_batch_size, wins, chs, dims, features), dtype=self.dtype)
        mask = tf.ones((self.maxlen, step_batch_size), dtype='bool')
        initial_states = None
        last_outputs = None  # for attention
        no_begin = tf.ones((step_batch_size,), dtype='bool')   # for attention
        return inputs, initial_states, last_outputs, mask, no_begin

    def get_config(self):
        config = super(SingleBiGRUPlusAttention, self).get_config()
        config.update({
            'prefix': self.prefix,
            'maxlen': self.maxlen,
            'seed': self.seed,
            'h_units': self.h_units,
            'a_units': tuple(self.a_units) if hasattr(self.a_units, '__iter__') else self.a_units,
            'merge_mode': self.merge_mode,
            'a_wd': tuple(self.a_wd) if hasattr(self.a_wd, '__iter__') else self.a_wd,
            'h_wd': self.h_wd,
            'cell_i_dropout': tuple(self.c_i_dropout) if hasattr(self.c_i_dropout, '__iter__') else self.c_i_dropout,
            'cell_h_dropout': tuple(self.c_h_dropout) if hasattr(self.c_h_dropout, '__iter__') else self.c_h_dropout
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'prefix', 'maxlen', 'h_units', 'a_units', 'seed', 'merge_mode', 'a_wd', 'h_wd',
              'cell_i_dropout', 'cell_h_dropout']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def get_initial_output(self, inputs):
        if self.merge_mode in ('sum', 'ave', 'mul'):
            return self.b_cell.get_initial_output(inputs[0])
        elif self.merge_mode == 'concat':
            return tf.concat((self.b_cell.get_initial_output(inputs[0]),
                              self.f_cell.get_initial_output(inputs[-1])), axis=1)
        else:
            b = tf.unstack(self.b_cell.get_initial_output(inputs[0]), axis=0)
            f = tf.unstack(self.f_cell.get_initial_output(inputs[-1]), axis=0)
            return tuple((_b, _f) for _b, _f in zip(b, f))

    # inputs---(steps, step_batch_size, ...)  initial_state---(b_state, f_state)
    # in fact, initial_states[1] and last_outputs[1] are None forever
    def call(self, inputs, initial_states=None, last_outputs=None, mask=None, no_begin=None, training=None):
        if self.maxlen < inputs.shape[0]:
            raise ValueError('the number of steps of data input is larger than max length set')
        steps = inputs.shape[0]

        tf.py_function(self.b_cell.reset_input_dropout_mask, inp=(), Tout=[])
        tf.py_function(self.b_cell.reset_hidden_dropout_mask, inp=(), Tout=[])
        tf.py_function(self.f_cell.reset_input_dropout_mask, inp=(), Tout=[])
        tf.py_function(self.f_cell.reset_hidden_dropout_mask, inp=(), Tout=[])

        inputs = tf.cast(inputs, self.dtype)
        inputs_r = inputs[::-1]
        if (hasattr(initial_states, '__iter__') and len(initial_states) != 2) or \
                (not hasattr(initial_states, '__iter__') and initial_states is not None):
            raise ValueError('not desired \'initial_states\'')
        if (hasattr(last_outputs, '__iter__') and len(last_outputs) != 2) or \
                (not hasattr(last_outputs, '__iter__') and last_outputs is not None):
            raise ValueError('not desired \'last_outputs\'')
        # NoneType not be supported in graph, so must give value first
        if initial_states is None or (initial_states[0] is None and initial_states[1] is None):
            initial_states = (self.b_cell.get_initial_state(inputs[0]), self.f_cell.get_initial_state(inputs[-1]))
        elif initial_states[0] is None:
            initial_states = (self.b_cell.get_initial_state(inputs[0]), initial_states[1])
        elif initial_states[1] is None:
            initial_states = (initial_states[0], self.f_cell.get_initial_state(inputs[-1]))
        # not only for settling NoneType problem, but also for banning vector passed in attention from being None
        if last_outputs is None or (last_outputs[0] is None and last_outputs[1] is None):
            last_outputs = (self.b_cell.get_initial_output(inputs[0]), self.f_cell.get_initial_output(inputs[-1]))
        elif last_outputs[0] is None:
            last_outputs = (self.b_cell.get_initial_output(inputs[0]), last_outputs[1])
        elif last_outputs[1] is None:
            last_outputs = (last_outputs[0], self.f_cell.get_initial_output(inputs[-1]))
        initial_states = (tf.cast(initial_states[0], self.dtype), tf.cast(initial_states[1], self.dtype))
        last_outputs = (tf.cast(last_outputs[0], self.dtype), tf.cast(last_outputs[1], self.dtype))

        if mask is not None:
            m_aligning = inputs.shape.ndims - mask.shape.ndims
            exp_mask = tf.cast(tf.reshape(mask, mask.shape + (1,) * m_aligning), self.dtype)

        # in tf control flow, python collections can't change length, must be fixed structure(tensor shape, dtype)
        #@tf.function
        def _loop():
            states = initial_states
            last_o = last_outputs
            backward = tf.TensorArray(inputs.dtype, size=steps)
            forward = tf.TensorArray(inputs.dtype, size=steps)
            ch_focus_b = tf.TensorArray(inputs.dtype, size=steps)
            ch_focus_f = tf.TensorArray(inputs.dtype, size=steps)
            win_focus_b = tf.TensorArray(inputs.dtype, size=steps)
            win_focus_f = tf.TensorArray(inputs.dtype, size=steps)
            if mask is not None:
                mask_r = mask[::-1]
                exp_r_mask = tf.cast(1, inputs.dtype) - exp_mask
                exp_r_mask_r = exp_r_mask[::-1]
                for step in tf.range(steps):
                    if step == 0:
                        c_b, w_b, i_b = self.attention_b((last_o[0], inputs[step]), no_begin, mask[step])
                        # because the first step for the reverse must use self-attention, the foreign query is invalid
                        c_f, w_f, i_f = self.attention_f((last_o[1], inputs_r[step]),
                                                         tf.ones_like(mask_r[step], dtype='bool'), mask_r[step])
                    else:
                        c_b, w_b, i_b = self.attention_b((last_o[0], inputs[step]), None, mask[step])
                        if tf.reduce_all(mask_r[step]):
                            c_f, w_f, i_f = self.attention_f((last_o[1], inputs_r[step]), None, mask_r[step])
                        else:
                            c_f, w_f, i_f = self.attention_f((last_o[1], inputs_r[step]),
                                                             tf.logical_not(mask_r[step]), mask_r[step])
                    o_b, s_b = self.b_cell(i_b, states[0], training, mask[step])
                    s_b = s_b + tf.multiply(states[0], exp_r_mask[step])
                    o_b = o_b + tf.multiply(last_o[0], exp_r_mask[step])
                    ch_focus_b = ch_focus_b.write(step, c_b)
                    win_focus_b = win_focus_b.write(step, w_b)
                    backward = backward.write(step, o_b)
                    o_f, s_f = self.f_cell(i_f, states[1], training, mask_r[step])
                    s_f = s_f + tf.multiply(states[1], exp_r_mask_r[step])
                    o_f = o_f + tf.multiply(last_o[1], exp_r_mask_r[step])
                    forward = forward.write(step, o_f)
                    ch_focus_f = ch_focus_f.write(step, c_f)
                    win_focus_f = win_focus_f.write(step, w_f)
                    last_o = (o_b, o_f)
                    states = (s_b, s_f)
            else:
                for step in tf.range(steps):
                    if step == 0:
                        c_b, w_b, i_b = self.attention_b((last_o[0], inputs[step]), no_begin)
                        # need to know which axis/axes is/are about step_batch_size
                        c_f, w_f, i_f = self.attention_f((last_o[1], inputs_r[step]),
                                                         tf.ones((inputs_r.shape[1],), dtype='bool'))
                    else:
                        c_b, w_b, i_b = self.attention_b((last_o[0], inputs[step]), None)
                        c_f, w_f, i_f = self.attention_f((last_o[1], inputs_r[step]), None)
                    o_b, s_b = self.b_cell(i_b, states[0], training)
                    ch_focus_b = ch_focus_b.write(step, c_b)
                    win_focus_b = win_focus_b.write(step, w_b)
                    backward = backward.write(step, o_b)
                    o_f, s_f = self.f_cell(i_f, states[1], training)
                    forward = forward.write(step, o_f)
                    ch_focus_f = ch_focus_f.write(step, c_f)
                    win_focus_f = win_focus_f.write(step, w_f)
                    last_o = (o_b, o_f)
                    states = (s_b, s_f)
            states = (tf.stop_gradient(states[0]), tf.stop_gradient(states[1]))
            ch_focus = tf.stack((ch_focus_b.stack(), ch_focus_f.stack()[::-1]), axis=0)
            win_focus = tf.stack((win_focus_b.stack(), win_focus_f.stack()[::-1]), axis=0)
            return backward.stack(), forward.stack()[::-1], states, ch_focus, win_focus

        backward_o, forward_o, end_states, ch_attention, win_attention = _loop()
        end_outputs = (tf.stop_gradient(backward_o[-1]), tf.stop_gradient(forward_o[0]))
        if mask is not None:
            backward_o = tf.multiply(backward_o, exp_mask)
            forward_o = tf.multiply(forward_o, exp_mask)

        def _combine():
            if self.merge_mode == 'sum':
                return tf.add(backward_o, forward_o)
            elif self.merge_mode == 'ave':
                return tf.multiply(tf.add(backward_o, forward_o), 0.5)
            elif self.merge_mode == 'mul':
                return tf.multiply(backward_o, forward_o)
            elif self.merge_mode == 'concat':
                return tf.concat((backward_o, forward_o), axis=-2)
            else:
                b = tf.unstack(backward_o, axis=0)
                f = tf.unstack(forward_o, axis=0)
                c = tuple((i, j) for i, j in zip(b, f))
                return c

        # first output has attribute '_keras_mask'， end_states(b_state, f_state)
        # when the end states are used as next sequence's initial states, must stop gradients
        # ch_attention --- (2, steps, batches, chs, 1/3)    win_attention --- (2, steps, batches, wins, chs, 1/3)
        return _combine(), end_states, end_outputs, ch_attention, win_attention


class StackBiGRUPlusAttention(keras.layers.Layer):
    def __init__(self, name, maxlen, seed, h_units, a_units, merge_mode='sum', layers=1,
                 a_wd=0.001, dtype=tf.float32, h_wd=0.001, cell_i_dropout=0., cell_h_dropout=0.):
        '''

        :param name:
        :param maxlen: the maximum steps in the sequence
        :param seed: used for initializer
        :param h_units: int, tuple or list, if belongs to the last two, its length must be equal to 'layers'
        :param a_units: int, tuple/list or nested tuple/list, if the situation is the latter, its length must be 2
                         and each one is 2-element tuple/list, if the situation is the middle one, its length must
                         be 2 (for window and channel attention separately) , then that indicating the same set of
                         two direction stack rnn
        :param merge_mode: str, tuple or list, if belongs to the last two, its length must be equal to 'layers'.
                           per element in ('sum', 'ave', 'mul', 'concat', None), used for concatenating two opposite
                           directions's results
        :param layers: the number of stacked layers
        :param a_wd: whether to use weights decay in attention layer. float, tuple/list , or nested tuple/list, the
                      same as 'a_units'
        :param dtype:
        :param h_wd: float, tuple or list, if belongs to the last two, its length must be equal to layers
        :param cell_i_dropout: float, nested tuple or nested list, if belongs to the last two, its length must be equal
                               to 'layers' and its per element's length must be 2. used for input dropout in GRU cell
        :param cell_h_dropout: the same as above. used for the previous hidden state dropout in GRU cell
        '''
        super(StackBiGRUPlusAttention, self).__init__(name=name, dtype=dtype)
        self.supports_masking = True
        self.maxlen = maxlen
        self.seed = seed
        self.sublayer_num = layers
        if layers < 1:
            raise ValueError('must have at least one rnn layer')
        self.h_units = h_units
        self.merge_mode = merge_mode
        self.h_wd = h_wd
        self.c_i_dropout = cell_i_dropout
        self.c_h_dropout = cell_h_dropout
        if isinstance(self.h_units, int):
            self.h_units = [self.h_units] * layers
        if isinstance(self.merge_mode, str):
            self.merge_mode = [self.merge_mode] * layers
        if isinstance(self.h_wd, float):
            self.h_wd = [self.h_wd] * layers
        if isinstance(self.c_i_dropout, float):
            self.c_i_dropout = [(self.c_i_dropout,) * 2] * layers
        if isinstance(self.c_i_dropout, (tuple, list)) and isinstance(self.c_i_dropout[0], float):
            self.c_i_dropout = [(dropout,) * 2 for dropout in self.c_i_dropout]
        if isinstance(self.c_h_dropout, float):
            self.c_h_dropout = [(self.c_h_dropout,) * 2] * layers
        if isinstance(self.c_h_dropout, (tuple, list)) and isinstance(self.c_h_dropout[0], float):
            self.c_h_dropout = [(dropout,) * 2 for dropout in self.c_h_dropout]
        if len(self.c_h_dropout[0]) != 2 or len(self.c_i_dropout[0]) != 2:
            raise ValueError('per cell dropout information don\'t match with the two opposite directions')
        if layers != len(self.h_units) or layers != len(self.c_h_dropout) or layers != len(
                self.c_i_dropout) or layers != len(self.merge_mode) or layers != len(self.h_wd):
            raise ValueError('information don\'t match with desired layers')

        if isinstance(a_units, int):
            self.a_units = [(a_units,) * 2] * 2
        elif hasattr(a_units, '__iter__') and len(a_units) == 2:   # supposing each element has the same form
            if a_units[0].__class__ == a_units[1].__class__:
                if isinstance(a_units[0], int):
                    self.a_units = [tuple(a_units)] * 2
                elif hasattr(a_units[0], '__iter__') and len(a_units[0]) == 2:
                    self.a_units = list(tuple(k) for k in a_units)
                else:
                    raise ValueError('can\'t recognized \'a_units\' passed')
            else:
                raise TypeError('the two elements must have the same type')
        else:
            raise ValueError('can\'t recognized \'a_units\' passed')

        if isinstance(a_wd, float):
            self.a_wd = [(a_wd,) * 2] * 2
        elif hasattr(a_wd, '__iter__') and len(a_wd) == 2:
            if isinstance(a_wd[0], type(a_wd[1])):
                if isinstance(a_wd[0], float):
                    self.a_wd = [tuple(a_wd)] * 2
                elif hasattr(a_wd[0], '__iter__') and len(a_wd[0]) == 2:
                    self.a_wd = list(tuple(k) for k in a_wd)
                else:
                    raise ValueError('can\'t recognized \'a_wd\' passed')
            else:
                raise TypeError('the two elements must have the same type')
        else:
            raise ValueError('can\'t recognized \'a_wd\' passed')

        self.plus_attention_layer = SingleBiGRUPlusAttention('No.1_layer_receiving_attention', 'No.1_',
                                                             self.maxlen, self.seed, self.h_units[0], self.a_units,
                                                             self.merge_mode[0], self.a_wd, self.dtype,
                                                             self.h_wd[0], self.c_i_dropout[0], self.c_h_dropout[0])
        self.stack_layers = list(SingleLayerBiGRU('No.{}_layer'.format(i + 1), 'No.{}_'.format(i + 1), self.maxlen,
                                                  self.h_units[i], self.seed, self.merge_mode[i], self.dtype,
                                                  self.h_wd[i], self.c_i_dropout[i], self.c_h_dropout[i]) for i in
                                 range(1, layers))

    def regu(self):
        r = []
        r += self.plus_attention_layer.regu()
        for l in self.stack_layers:
            r += l.regu()
        return r

    def get_input(self, step_batch_size, wins, chs, dims, f):
        inputs = tf.zeros((self.maxlen, step_batch_size, wins, chs, dims, f), dtype=self.dtype)
        mask = tf.ones((self.maxlen, step_batch_size), dtype='bool')
        initial_states = None
        last_outputs = None
        no_begin = tf.ones((step_batch_size,), dtype='bool')
        return inputs, initial_states, last_outputs, mask, no_begin

    def get_config(self):
        config = super(StackBiGRUPlusAttention, self).get_config()
        config.update({
            'maxlen': self.maxlen,
            'seed': self.seed,
            'h_units': tuple(self.h_units) if hasattr(self.h_units, '__iter__') else self.h_units,
            'a_units': tuple(self.a_units) if hasattr(self.a_units, '__iter__') else self.a_units,
            'merge_mode': tuple(self.merge_mode) if hasattr(self.merge_mode, '__iter__') else self.merge_mode,
            'layers': self.sublayer_num,
            'a_wd': tuple(self.a_wd) if hasattr(self.a_wd, '__iter__') else self.a_wd,
            'h_wd': tuple(self.h_wd) if hasattr(self.h_wd, '__iter__') else self.h_wd,
            'cell_i_dropout': tuple(self.c_i_dropout) if hasattr(self.c_i_dropout, '__iter__') else self.c_i_dropout,
            'cell_h_dropout': tuple(self.c_h_dropout) if hasattr(self.c_h_dropout, '__iter__') else self.c_h_dropout
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'maxlen', 'h_units', 'a_units', 'seed', 'merge_mode', 'layers',
              'a_wd', 'h_wd', 'cell_i_dropout', 'cell_h_dropout']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    # inputs---(steps, step_batch_size, wins, chs, i_dims, 1/3)
    # initial_states/last_outputs --- nested_tuple
    def call(self, inputs, initial_states=None, last_outputs=None, mask=None, no_begin=None, training=None,
             leaky=0.01, threshold=0.):
        if hasattr(initial_states, '__iter__'):
            if len(initial_states) != self.sublayer_num:
                raise ValueError('initial_states must match with rnn layers')
        elif initial_states is None:
            initial_states = [None] * self.sublayer_num
        else:
            raise ValueError('can\'t recognize the param \'initial_states\' passed in')
        all_outputs_0, end_states_0, end_outputs_0, ch_attention, win_attention = \
            self.plus_attention_layer(inputs, initial_states[0], last_outputs, mask, no_begin, training)

        rnn_inputs = all_outputs_0
        end_states = tf.TensorArray(all_outputs_0.dtype, size=self.sublayer_num, infer_shape=False)
        end_states = end_states.write(0, end_states_0)
        # start to loop for normal stack layers, idx in loop is tensor if use tf.range, can't be used as tuple/list indices
        # and None value can't appear in tf loop   (the second one of initial_states' per tuple is None)
        for idx in range(1, self.sublayer_num):
            all_outputs_i, end_states_i, _ = self.stack_layers[idx - 1](rnn_inputs, initial_states[idx],
                                                                                    mask, training)
            rnn_inputs = all_outputs_i
            end_states = end_states.write(idx, end_states_i)

        top_outputs = rnn_inputs   # local variable problem, can't use 'all_outputs_i'
        states = tuple(end_states.read(i) for i in range(self.sublayer_num))

        # states/outputs returned ((2, step_batch_size, h_units),...)_layer_num
        return top_outputs, states, end_outputs_0, ch_attention, win_attention


class StackGRUPlusAttention(keras.layers.Layer):
    def __init__(self, name, maxlen, seed, h_units, a_units, reverse, layers, a_wd=0.001,
                 dtype=tf.float32, h_wd=0.001, cell_i_dropout=0., cell_h_dropout=0.):
        '''

        :param name:
        :param maxlen: the maximum steps in the sequence
        :param seed: used for initializer
        :param h_units: int, tuple or list, if belongs to the last two, its length must be equal to 'layers'
        :param a_units: int, tuple or list, if belongs to the last two, its length must be equal to 2,
                        for channel and window attention separately
        :param reverse: bool, used to indicate that the direction of the GRU
        :param layers: the number of stacked layers
        :param a_wd: whether to use weights decay in attention layer. float, tuple or list, if belongs to the last two,
                     its length must be equal to 2
        :param dtype:
        :param h_wd: float, tuple or list, if belongs to the last two, its length must be equal to layers
        :param cell_i_dropout: float, tuple or list, if belongs to the last two, its length must be equal to 'layers' .
                               used for input dropout in GRU cell
        :param cell_h_dropout: the same as the last term. used for hidden dropout in GRU cell
        '''
        super(StackGRUPlusAttention, self).__init__(name=name, dtype=dtype)
        self.supports_masking = True
        self.maxlen = maxlen
        self.reverse = reverse  # bool
        self.seed = seed
        self.sublayer_num = layers + 1
        self.h_units = h_units
        self.h_wd = h_wd
        self.c_i_dropout = cell_i_dropout
        self.c_h_dropout = cell_h_dropout
        if isinstance(self.h_units, int):
            self.h_units = [self.h_units] * layers
        if isinstance(self.h_wd, float):
            self.h_wd = [self.h_wd] * layers
        if isinstance(self.c_i_dropout, float):
            self.c_i_dropout = [self.c_i_dropout] * layers
        if isinstance(self.c_h_dropout, float):
            self.c_h_dropout = [self.c_h_dropout] * layers
        if layers != len(self.h_units) or layers != len(self.c_h_dropout) or layers != len(self.c_i_dropout)\
                or layers != len(self.h_wd):
            raise ValueError('information don\'t match with desired layers')
        self.a_units = a_units
        if isinstance(self.a_units, int):
            self.a_units = [self.a_units] * 2
        self.a_wd = a_wd
        if isinstance(self.a_wd, float):
            self.a_wd = [self.a_wd] * 2
        if len(self.a_units) != 2 or len(self.a_wd) != 2:
            raise ValueError('attention information must be two-element for channel attention and window attention')
        if reverse:
            prefix = 'reverse_'
        else:
            prefix = 'forward_'
        self.cells = [MyGRUCell(prefix + 'No.{}_rnn_layer'.format(i + 1),
                                self.h_units[i], self.seed, self.dtype, self.h_wd[i], self.c_i_dropout[i],
                                self.c_h_dropout[i]) for i in range(layers)]
        self.attention_layer = Attention(prefix + 'attention_layer', self.seed, self.a_units, self.dtype,
                                         self.a_wd)

    def regu(self):
        r = []
        r += self.attention_layer.regu()
        for l in self.cells:
            r += l.regu()
        return r

    def get_input(self, step_batch_size, wins, chs, dims, f):
        inputs = tf.zeros((self.maxlen, step_batch_size, wins, chs, dims, f), dtype=self.dtype)
        mask = tf.ones((self.maxlen, step_batch_size), dtype='bool')
        initial_states = None
        last_outputs = None
        no_begin = tf.ones((step_batch_size,), dtype='bool')
        return inputs, initial_states, last_outputs, mask, no_begin

    def get_config(self):
        config = super(StackGRUPlusAttention, self).get_config()
        config.update({
            'maxlen': self.maxlen,
            'seed': self.seed,
            'h_units': tuple(self.h_units) if hasattr(self.h_units, '__iter__') else self.h_units,
            'a_units': tuple(self.a_units) if hasattr(self.a_units, '__iter__') else self.a_units,
            'reverse': self.reverse,
            'layers': self.sublayer_num - 1,
            'a_wd': tuple(self.a_wd) if hasattr(self.a_wd, '__iter__') else self.a_wd,
            'h_wd': tuple(self.h_wd) if hasattr(self.h_wd, '__iter__') else self.h_wd,
            'cell_i_dropout': tuple(self.c_i_dropout) if hasattr(self.c_i_dropout, '__iter__') else self.c_i_dropout,
            'cell_h_dropout': tuple(self.c_h_dropout) if hasattr(self.c_h_dropout, '__iter__') else self.c_h_dropout
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'maxlen', 'h_units', 'a_units', 'seed', 'reverse', 'layers',
              'a_wd', 'h_wd', 'cell_i_dropout', 'cell_h_dropout']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    # inputs --- (steps, step_batch_size, wins, chs, features, 1/3)
    # initial_states --- ((step_batch_size, h_units, 1/3),...)_rnn_layers
    # last_outputs --- (step_batch_size, top_h_units, 1/3)
    # mask --- (steps, step_batch_size)
    def call(self, inputs, initial_states=None, last_outputs=None, mask=None, no_begin=None, training=None,
             leaky=0.01, threshold=0.):
        if self.maxlen < inputs.shape[0]:
            raise ValueError('the number of steps of data input is larger than max length set')
        steps = inputs.shape[0]
        if hasattr(initial_states, '__iter__'):
            if len(initial_states) != len(self.cells):
                raise ValueError('initial_states must match with rnn layers')
        elif initial_states is None:
            initial_states = [None] * len(self.cells)
        else:
            raise ValueError('can\'t recognize the param \'initial_states\' passed in')
        for cell in self.cells:
            tf.py_function(cell.reset_input_dropout_mask, inp=(), Tout=[])
            tf.py_function(cell.reset_hidden_dropout_mask, inp=(), Tout=[])

        inputs = tf.cast(inputs, self.dtype)
        if self.reverse:
            if no_begin is not None:
                no_begin = None
            inputs = inputs[::-1]
            if mask is not None:
                mask = mask[::-1]
        if mask is not None:
            m_aligning = inputs.shape.ndims - mask.shape.ndims
            exp_mask = tf.cast(tf.reshape(mask, mask.shape + (1,) * m_aligning), self.dtype)
        batch = inputs.shape[1]
        f = inputs.shape[-1]

        #@tf.function
        def _loop_with_attention():
            # the tensor returned by 'read' has unknown shape because of no infer_shape in tf.function
            # but in eager executing mode, still have specific shape
            last_s = tf.TensorArray(self.dtype, size=len(initial_states), infer_shape=False)
            # python loop
            for i in range(len(initial_states)):     # TensorArray.size() return tensor
                if initial_states[i] is None:
                    last_s = last_s.write(i, self.cells[i].get_initial_state(inputs[0]))
                else:
                    last_s = last_s.write(i, tf.cast(initial_states[i], self.dtype))
            if last_outputs is None:
                last_o_for_a = self.cells[-1].get_initial_output(inputs[0])
            else:
                last_o_for_a = tf.cast(last_outputs, self.dtype)
            last_o = self.cells[-1].get_initial_output(inputs[0])     # serve for current batch, in despite of last batch
            top_outputs = tf.TensorArray(self.dtype, size=steps)
            ch_focus = tf.TensorArray(self.dtype, size=steps)
            win_focus = tf.TensorArray(self.dtype, size=steps)
            # has mask and use dynamics subnets
            if mask is not None:
                exp_r_m = tf.cast(1, self.dtype) - exp_mask
                # tf loop
                for step in tf.range(steps):
                    if step == 0:
                        ch_f, win_f, o = self.attention_layer((last_o_for_a, inputs[step]), no_begin, mask[step])
                    else:
                        if not self.reverse or tf.reduce_all(mask[step - 1]):
                            ch_f, win_f, o = self.attention_layer((last_o_for_a, inputs[step]), None, mask[step])
                        else:
                            ch_f, win_f, o = self.attention_layer((last_o_for_a, inputs[step]),
                                                                  tf.logical_not(mask[step]), mask[step])
                    ch_focus = ch_focus.write(step, ch_f)
                    win_focus = win_focus.write(step, win_f)
                    rnn_i = o
                    for i in range(len(self.cells)):
                        aux_s = last_s.read(i)    # unknown shape
                        aux_s.set_shape(tf.TensorShape([batch] + self.cells[i].state_size + [f]).dims)
                        o, s = self.cells[i](rnn_i, aux_s, training, mask[step])
                        last_s = last_s.write(i, s + tf.multiply(aux_s, exp_r_m[step]))
                        rnn_i = o
                    last_o = rnn_i + tf.multiply(last_o, exp_r_m[step])
                    top_outputs = top_outputs.write(step, last_o)
                    last_o_for_a = rnn_i
            else:
                for step in tf.range(steps):
                    if step == 0:
                        ch_f, win_f, rnn_i = self.attention_layer((last_o_for_a, inputs[step]), no_begin)
                    else:
                        ch_f, win_f, rnn_i = self.attention_layer((last_o_for_a, inputs[step]), None)
                    ch_focus = ch_focus.write(step, ch_f)
                    win_focus = win_focus.write(step, win_f)
                    for i in range(len(self.cells)):
                        aux_s = last_s.read(i)
                        aux_s.set_shape(tf.TensorShape([batch] + self.cells[i].state_size + [f]).dims)
                        o, s = self.cells[i](rnn_i, aux_s, training)
                        last_s = last_s.write(i, s)
                        rnn_i = o
                    top_outputs = top_outputs.write(step, rnn_i)
                    last_o_for_a = rnn_i
            ch_focus = ch_focus.stack()
            win_focus = win_focus.stack()
            states = list(tf.stop_gradient(last_s.read(i)) for i in range(len(self.cells)))
            return top_outputs.stack(), states, ch_focus, win_focus

        outputs, end_states, ch_attention, win_attention = _loop_with_attention()
        end_outputs = tf.stop_gradient(outputs[-1])
        if mask is not None:
            outputs = tf.multiply(outputs, exp_mask)
        for i, c in enumerate(self.cells):
            s = end_states[i]
            s.set_shape([batch] + c.state_size + [f])
            end_states[i] = s

        return outputs, tuple(end_states), end_outputs, ch_attention, win_attention


class StackGRUCells(keras.layers.Layer):
    def __init__(self, name, seed, h_units, reverse, layers,
                 dtype=tf.float32, h_wd=0.001, cell_i_dropout=0., cell_h_dropout=0.):
        '''

        :param name:
        :param seed: used for initializer
        :param h_units: int, tuple or list, if belongs to the last two, its length must be equal to 'layers'
        :param reverse: bool, used to indicate that the direction of the GRU
        :param layers: the number of stacked layers
        :param dtype:
        :param h_wd: float, tuple or list, if belongs to the last two, its length must be equal to 'layers'
        :param cell_i_dropout: float, tuple or list, if belongs to the last two, its length must be equal to 'layers' .
                               used for input dropout in GRU cell
        :param cell_h_dropout: the same as the last term. used for hidden dropout in GRU cell
        '''
        super(StackGRUCells, self).__init__(name=name, dtype=dtype)
        self.supports_masking = True
        self.reverse = reverse  # bool
        self.seed = seed
        self.sublayer_num = layers
        self.h_units = h_units
        self.h_wd = h_wd
        self.c_i_dropout = cell_i_dropout
        self.c_h_dropout = cell_h_dropout
        if isinstance(self.h_units, int):
            self.h_units = [self.h_units] * layers
        if isinstance(self.h_wd, float):
            self.h_wd = [self.h_wd] * layers
        if isinstance(self.c_i_dropout, float):
            self.c_i_dropout = [self.c_i_dropout] * layers
        if isinstance(self.c_h_dropout, float):
            self.c_h_dropout = [self.c_h_dropout] * layers
        if layers != len(self.h_units) or layers != len(self.c_h_dropout) or layers != len(self.c_i_dropout)\
                or layers != len(self.h_wd):
            raise ValueError('information don\'t match with desired layers')
        if reverse:
            prefix = 'reverse_'
        else:
            prefix = 'forward_'
        self.cells = [MyGRUCell(prefix + 'No.{}_rnn_layer'.format(i + 1),
                                self.h_units[i], self.seed, self.dtype, self.h_wd[i], self.c_i_dropout[i],
                                self.c_h_dropout[i]) for i in range(layers)]

    def regu(self):
        r = []
        for l in self.cells:
            r += l.regu()
        return r

    def get_input(self, step_batch_size, dims, f):
        inputs = tf.zeros((step_batch_size, dims, f), dtype=self.dtype)
        states = tuple(cell.get_initial_state(inputs) for cell in self.cells)
        mask = tf.constant([True] * step_batch_size, dtype='bool')
        return inputs, states, mask

    def get_config(self):
        config = super(StackGRUCells, self).get_config()
        config.update({
            'seed': self.seed,
            'h_units': tuple(self.h_units) if hasattr(self.h_units, '__iter__') else self.h_units,
            'reverse': self.reverse,
            'layers': self.sublayer_num,
            'h_wd': tuple(self.h_wd) if hasattr(self.h_wd, '__iter__') else self.h_wd,
            'cell_i_dropout': tuple(self.c_i_dropout) if hasattr(self.c_i_dropout, '__iter__') else self.c_i_dropout,
            'cell_h_dropout': tuple(self.c_h_dropout) if hasattr(self.c_h_dropout, '__iter__') else self.c_h_dropout
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'h_units', 'seed', 'reverse', 'layers', 'h_wd',
              'cell_i_dropout', 'cell_h_dropout']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def get_initial_output(self, inp):
        with tf.init_scope():
            return self.cells[-1].get_initial_output(inp)

    def get_initial_states(self, inp):
        with tf.init_scope():
            ta = tf.TensorArray(self.dtype, size=self.sublayer_num, infer_shape=False)
            for i, cell in zip(range(self.sublayer_num), self.cells):
                ta = ta.write(i, cell.get_initial_state(inp))
            return tuple(ta.read(i) for i in range(self.sublayer_num))

    # inputs --- (step_batch_size, input_dims, 1/3)
    # initial_states --- ((step_batch_size, h_units, 1/3),...)_rnn_layers
    # mask --- (step_batch_size,)
    @tf.function
    def call(self, inputs, states=None, training=None, mask=None):
        if hasattr(states, '__iter__'):
            if len(states) != len(self.cells):
                raise ValueError('initial_states must match with rnn layers')
        elif states is None:
            states = [None] * len(self.cells)
        else:
            raise ValueError('params states isn\'t desired')
        new_states = tf.TensorArray(self.dtype, size=self.sublayer_num, infer_shape=False)
        batch = [inputs.shape[0]]
        f = [inputs.shape[-1]]
        inp = inputs
        for i, cell, s in zip(range(self.sublayer_num), self.cells, states):
            tf.py_function(cell.reset_input_dropout_mask, inp=(), Tout=[])
            tf.py_function(cell.reset_hidden_dropout_mask, inp=(), Tout=[])
            if s is None:
                s = cell.get_initial_state(inp)
            o, h = cell(inp, s, training, mask)
            inp = o
            new_states = new_states.write(i, h)
        output = inp
        new_hidden = tuple(new_states.read(i) for i in range(self.sublayer_num))
        for i in range(self.sublayer_num):
            new_hidden[i].set_shape(batch + self.cells[i].state_size + f)
        return output, new_hidden


class BidirectionalWrapper(keras.layers.Layer):
    def __init__(self, name, maxlen, seed, h_units, a_units, layers, merge_mode='sum',
                 a_wd=0.001, dtype=tf.float32, h_wd=0.001, cell_i_dropout=0., cell_h_dropout=0.):
        '''

        :param name:
        :param maxlen:
        :param seed:
        :param h_units: int, tuple/list or nested tuple/list, because of aiming to settle the outputs of top layers,
                        so the h_units in top layers must be equal, type tuple/list only for the same set of two
                        direction stack rnn
        :param a_units: int, tuple/list or nested tuple/list, if the situation is the latter,
                        its length must be 2 and each one is 2-element tuple/list, if the situation is the middle one,
                        its length must be 2 (for window and channel attention separately) , then that indicating the
                        same set of two direction stack rnn
        :param layers: int, tuple/list, if be the latter, its length must be 2
        :param merge_mode: the same as StackBiGRUPlusAttention
        :param a_wd: float, tuple/list , or nested tuple/list, the same as 'a_units'
        :param dtype:
        :param h_wd: float, tuple/list or nested tuple/list. type tuple/list only for
                     the same set of two direction stack rnn. if is the last one, its length must be 2
        :param cell_i_dropout: float, tuple/list or nested tuple/list, if is the last one,
                               its length must be 2, and type tuple/list only for the same set of two
                               direction stack rnn
        :param cell_h_dropout: the same as above
        '''
        super(BidirectionalWrapper, self).__init__(name=name, dtype=dtype)
        self.supports_masking = True
        self.maxlen = maxlen
        self.seed = seed
        self.sublayer_num = 2
        if merge_mode not in ('sum', 'ave', 'mul', 'concat', None):
            raise ValueError('can\'t recognized \'merge_mode\' passed')
        self.merge_mode = merge_mode

        if isinstance(layers, int):
            l = [layers] * 2
        elif hasattr(layers, '__iter__'):
            if len(layers) != 2:
                raise ValueError('layers can indicate normal and reverse rnn at most')
            else:
                l = list(layers)
        else:
            raise ValueError('can\'t recognized \'layers\' passed')

        if isinstance(h_units, int):
            self.h_units = [(h_units,) * l[0], (h_units,) * l[1]]
        elif hasattr(h_units, '__iter__') and len(h_units) == l[0] and l[0] == l[1] \
                and not hasattr(h_units[0], '__iter__'):
            self.h_units = [tuple(h_units)] * 2       # the middle situation for 'h_units'
        elif hasattr(h_units, '__iter__') and len(h_units) == 2 and hasattr(h_units[0], '__iter__'):
            if len(h_units[0]) != l[0] or len(h_units[1]) != l[1]:
                raise ValueError('information concerned should coordinate with layers')
            elif h_units[0][-1] != h_units[1][-1]:
                raise ValueError('the h_units of the two top layers must be equal')
            else:
                self.h_units = list(tuple(k) for k in h_units)
        else:
            raise ValueError('can\'t recognized \'h_units\' passed')

        if isinstance(a_units, int):
            self.a_units = [(a_units,) * 2] * 2
        elif hasattr(a_units, '__iter__') and len(a_units) == 2:   # supposing each element has the same form
            if a_units[0].__class__ == a_units[1].__class__:
                if isinstance(a_units[0], int):
                    self.a_units = [tuple(a_units)] * 2
                elif hasattr(a_units[0], '__iter__') and len(a_units[0]) == 2:
                    self.a_units = list(tuple(k) for k in a_units)
                else:
                    raise ValueError('can\'t recognized \'a_units\' passed')
            else:
                raise TypeError('the two elements must have the same type')
        else:
            raise ValueError('can\'t recognized \'a_units\' passed')

        if isinstance(a_wd, float):
            self.a_wd = [(a_wd,) * 2] * 2
        elif hasattr(a_wd, '__iter__') and len(a_wd) == 2:
            if isinstance(a_wd[0], type(a_wd[1])):
                if isinstance(a_wd[0], float):
                    self.a_wd = [tuple(a_wd)] * 2
                elif hasattr(a_wd[0], '__iter__') and len(a_wd[0]) == 2:
                    self.a_wd = list(tuple(k) for k in a_wd)
                else:
                    raise ValueError('can\'t recognized \'a_wd\' passed')
            else:
                raise TypeError('the two elements must have the same type')
        else:
            raise ValueError('can\'t recognized \'a_wd\' passed')

        if isinstance(h_wd, float):
            self.h_wd = [(h_wd,) * l[0], (h_wd,) * l[1]]
        elif hasattr(h_wd, '__iter__') and len(h_wd) == l[0] and l[0] == l[1] \
                and not hasattr(h_wd[0], '__iter__'):
            self.h_wd = [tuple(h_wd)] * 2
        elif hasattr(h_wd, '__iter__') and len(h_wd) == 2 and hasattr(h_wd[0], '__iter__'):
            if len(h_wd[0]) != l[0] or len(h_wd[1]) != l[1]:
                raise ValueError('information concerned should coordinate with layers')
            else:
                self.h_wd = list(tuple(k) for k in h_wd)
        else:
            raise ValueError('can\'t recognized \'h_units\' passed')

        if isinstance(cell_i_dropout, float):
            self.c_i_dropout = [(cell_i_dropout,) * l[0], (cell_i_dropout,) * l[1]]
        elif hasattr(cell_i_dropout, '__iter__') and len(cell_i_dropout) == l[0] and \
                isinstance(cell_i_dropout[0], float) and l[0] == l[1]:
            self.c_i_dropout = [tuple(cell_i_dropout)] * l[0]
        elif hasattr(cell_i_dropout, '__iter__') and len(cell_i_dropout) == 2 \
                and hasattr(cell_i_dropout[0], '__iter__'):
            if len(cell_i_dropout[0]) != l[0] or len(cell_i_dropout[1]) != l[1]:
                raise ValueError('information concerned should coordinate with layers')
            else:
                self.c_i_dropout = list(tuple(k) for k in cell_i_dropout)
        else:
            raise ValueError('can\'t recognized \'cell_i_dropout\' passed')

        if isinstance(cell_h_dropout, float):
            self.c_h_dropout = [(cell_h_dropout,) * l[0], (cell_h_dropout,) * l[1]]
        elif hasattr(cell_h_dropout, '__iter__') and len(cell_h_dropout) == l[0] and \
                isinstance(cell_h_dropout[0], float) and l[0] == l[1]:
            self.c_h_dropout = [tuple(cell_h_dropout)] * l[0]
        elif hasattr(cell_h_dropout, '__iter__') and len(cell_h_dropout) == 2 \
                and hasattr(cell_h_dropout[0], '__iter__'):
            if len(cell_h_dropout[0]) != l[0] or len(cell_h_dropout[1]) != l[1]:
                raise ValueError('information concerned should coordinate with layers')
            else:
                self.c_h_dropout = list(tuple(k) for k in cell_h_dropout)
        else:
            raise ValueError('can\'t recognized \'cell_h_dropout\' passed')

        self.rnn_attention_b = StackGRUPlusAttention('forward_direction', self.maxlen, self.seed, self.h_units[0],
                                                     self.a_units[0], False, l[0],  self.a_wd[0],
                                                     self.dtype, self.h_wd[0], self.c_i_dropout[0], self.c_h_dropout[0])
        self.rnn_attention_f = StackGRUPlusAttention('reverse_direction', self.maxlen, self.seed, self.h_units[1],
                                                     self.a_units[1], True, l[1], self.a_wd[1],
                                                     self.dtype, self.h_wd[1], self.c_i_dropout[1], self.c_h_dropout[1])

    def regu(self):
        return self.rnn_attention_b.regu() + self.rnn_attention_f.regu()

    def get_input(self, step_batch_size, wins, chs, dims, f):
        inputs = tf.zeros((self.maxlen, step_batch_size, wins, chs, dims, f), dtype=self.dtype)
        mask = tf.ones((self.maxlen, step_batch_size), dtype='bool')
        initial_states = None
        last_outputs = None
        no_begin = tf.ones((step_batch_size,), dtype='bool')
        return inputs, initial_states, last_outputs, mask, no_begin

    def get_config(self):
        config = super(BidirectionalWrapper, self).get_config()
        config.update({
            'maxlen': self.maxlen,
            'seed': self.seed,
            'h_units': tuple(self.h_units) if hasattr(self.h_units, '__iter__') else self.h_units,
            'a_units': tuple(self.a_units) if hasattr(self.a_units, '__iter__') else self.a_units,
            'layers': (len(self.rnn_stack_b.cells), len(self.rnn_stack_f.cells)),
            'merge_mode': self.merge_mode,
            'a_wd': tuple(self.a_wd) if hasattr(self.a_wd, '__iter__') else self.a_wd,
            'h_wd': tuple(self.h_wd) if hasattr(self.h_wd, '__iter__') else self.h_wd,
            'cell_i_dropout': tuple(self.c_i_dropout) if hasattr(self.c_i_dropout, '__iter__') else self.c_i_dropout,
            'cell_h_dropout': tuple(self.c_h_dropout) if hasattr(self.c_h_dropout, '__iter__') else self.c_h_dropout
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'maxlen', 'seed', 'h_units', 'a_units', 'layers', 'merge_mode',
              'a_wd', 'h_wd', 'cell_i_dropout', 'cell_h_dropout']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    # the same inputs\mask\no_begin(only providing normal direction's)\training\leaky\threshold
    # initial_states --- ((0_layer_state, ...), None)
    # last_outputs --- (top_layer_output, None)
    def call(self, inputs, initial_states=None, last_outputs=None, mask=None, no_begin=None, training=None,
             leaky=0.01, threshold=0.):
        if last_outputs is None:
            last_outputs = [None] * 2
        elif hasattr(last_outputs, '__iter__'):
            if len(last_outputs) != 2:
                raise ValueError('\'last_outputs\' passed should be in form of for backward stack rnn and  for '
                                 'the reverse')
            if last_outputs[0] is not None and last_outputs[1] is not None \
                    and last_outputs[0].shape[-2] != last_outputs[1].shape[-2]:
                raise ValueError('the last step\'s outputs of two direction must have equal dimension ')
        else:
            raise ValueError('can\'t recognize the passed \'last_outputs\'')
        if initial_states is None:
            initial_states = [None] * 2
        elif hasattr(initial_states, '__iter__'):
            if len(initial_states) != 2:
                raise ValueError('\'initial_states\' passed should be in form of for backward stack rnn and  for '
                                 'the reverse')
            # though also need to judge the top layers from both have the same dimensions whether or not,
            # the same operation has been completed at the 'last_output' step
        else:
            raise ValueError('can\'t recognize the passed \'initial_states\'')
        no_begin = [no_begin, tf.ones((inputs.shape[1],), dtype='bool')]

        b_outputs, b_end_states, b_end_output, b_chs_focus, b_wins_focus = \
            self.rnn_attention_b(inputs, initial_states[0], last_outputs[0], mask, no_begin[0], training, leaky,
                                 threshold)
        if mask is not None:
            f_outputs, f_end_states, f_end_output, f_chs_focus, f_wins_focus = \
                self.rnn_attention_b(inputs[::-1], initial_states[1], last_outputs[1], mask[::-1], no_begin[1],
                                     training, leaky, threshold)
        else:
            f_outputs, f_end_states, f_end_output, f_chs_focus, f_wins_focus = \
                self.rnn_attention_f(inputs[::-1], initial_states[1], last_outputs[1], None, no_begin[1],
                                     training, leaky, threshold)
        f_outputs = f_outputs[::-1]

        def _combine():
            if self.merge_mode == 'sum':
                return tf.add(b_outputs, f_outputs)
            elif self.merge_mode == 'ave':
                return tf.multiply(tf.add(b_outputs, f_outputs), 0.5)
            elif self.merge_mode == 'mul':
                return tf.multiply(b_outputs, f_outputs)
            elif self.merge_mode == 'concat':
                return tf.concat((b_outputs, f_outputs), axis=-2)
            else:
                b = tf.unstack(b_outputs, axis=0)
                f = tf.unstack(f_outputs, axis=0)
                c = tuple((i, j) for i, j in zip(b, f))
                return c

        return _combine(), (b_end_states, f_end_states), (b_end_output, f_end_output), tf.stack(
            (b_chs_focus, f_chs_focus), axis=2), tf.stack((b_wins_focus, f_wins_focus), axis=2)


class Classification(keras.layers.Layer):
    def __init__(self, name, n_class, seed, dr_rate=0., dtype=tf.float32, bias=False, wd=0.001):
        super(Classification, self).__init__(name=name, dtype=dtype)
        self.supports_masking = True
        self.n_class = n_class
        self.seed = seed
        self.dr_rate = dr_rate
        self.use_bias = bias
        self.wd = wd
        if self.wd:
            self.regularizer = keras.regularizers.l2(self.wd)
        else:
            self.regularizer = None
        self.regulate_vars = []

    def get_config(self):
        config = super(Classification, self).get_config()
        config.update({
            'n_class': self.n_class,
            'seed': self.seed,
            'dr_rate': self.dr_rate,
            'bias': self.use_bias,
            'wd': self.wd
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        new = {'name': kwargs['name'], 'dtype': kwargs['dtype'], 'n_class': kwargs['n_class'], 'seed': kwargs['seed'],
               'dr_rate': kwargs['dr_rate'], 'bias': kwargs['bias'], 'wd': kwargs['wd']}
        return cls(**new)

    def build(self, input_shape):
        self.w = self.add_weight(name='weights',
                                 shape=(input_shape[-1], self.n_class),
                                 dtype=self.dtype,
                                 initializer=MyGlorotInitializer(self.seed, 'normal', 0, 1),
                                 regularizer=self.regularizer,
                                 trainable=True,
                                 constraint=None)
        self.regulate_vars.append(self.w)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.n_class,),
                                    dtype=self.dtype,
                                    initializer=keras.initializers.get('zeros'),
                                    regularizer=None,
                                    trainable=self.use_bias,
                                    constraint=None)
        self.built = True

    def regu(self):
        if self.regularizer is not None:
            r = []
            for v in self.regulate_vars:
                r.append(tf.clip_by_value(self.regularizer(v), 1e-8, np.Inf))
            return r
        return []

    def get_input(self, step, step_batch_size, dims):
        inputs = tf.zeros((step, step_batch_size, dims), dtype=self.dtype)
        mask = tf.ones((step, step_batch_size), dtype='bool')
        return inputs, mask

    # (steps, step_batch_size, integrated_top_h_units)
    # output --- (steps, step_batch_size, n_class)
    @tf.function
    def call(self, inputs, mask=None, training=None):
        inputs = tf.cast(inputs, self.dtype)
        if mask is not None:
            exp_mask = tf.cast(tf.expand_dims(mask, axis=-1), inputs.dtype)
            inputs = tf.multiply(inputs, exp_mask)
        if self.dr_rate and training:
            inputs = tf.nn.dropout(inputs, rate=self.dr_rate, noise_shape=(1, 1, None))
        logits = tf.add(tf.matmul(inputs, self.w), self.bias)
        result = tf.nn.softmax(logits, axis=-1)
        if mask is not None:
            result = tf.multiply(result, exp_mask)
        return result


class RectifyClassification(keras.layers.Layer):
    def __init__(self, name, n_class, dr_rate=0., seed=None, fc_bias=True, dtype=tf.float32, wd=0.001):
        super(RectifyClassification, self).__init__(name=name, dtype=dtype)
        self.supports_masking = True
        self.n_class = n_class     
        self.dr_rate = dr_rate
        self.seed = seed
        self.fc_bias = fc_bias
        self.wd = wd
        if self.wd:
            self.regularizer = keras.regularizers.l2(self.wd)
        else:
            self.regularizer = None
        self.regulate_vars = []

    def get_config(self):
        config = super(RectifyClassification, self).get_config()
        config.update({
            'n_class': self.n_class,
            'dr_rate': self.dr_rate,
            'seed': self.seed,
            'fc_bias': self.fc_bias,
            'wd': self.wd
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        ks = ['name', 'dtype', 'n_class', 'dr_rate', 'seed', 'fc_bias', 'wd']
        new = kwargs.fromkeys(ks)
        for k in ks:
            new[k] = kwargs[k]
        return cls(**new)

    def build(self, input_shape):
        self.w_n = self.add_weight(name='n_classification_fc_weights',
                                   shape=(input_shape[-1], self.n_class),
                                   dtype=self.dtype,
                                   initializer=MyGlorotInitializer(self.seed, 'normal', 0, 1),
                                   regularizer=self.regularizer,
                                   trainable=True,
                                   constraint=None)
        self.regulate_vars.append(self.w_n)
        self.w_2 = self.add_weight(name='2_classification_fc_weights',
                                   shape=(input_shape[-1], 1),
                                   dtype=self.dtype,
                                   initializer=MyGlorotInitializer(self.seed, 'normal', 0, 1),
                                   regularizer=self.regularizer,
                                   trainable=True,
                                   constraint=None)
        self.regulate_vars.append(self.w_2)
        self.fc_b_n = self.add_weight(name='n_classification_fc_bias', 
                                      shape=(self.n_class,), 
                                      dtype=self.dtype, 
                                      initializer=keras.initializers.get('zeros'),
                                      regularizer=None,
                                      trainable=self.fc_bias,
                                      constraint=None)
        self.fc_b_2 = self.add_weight(name='2_classification_fc_bias',
                                      shape=(1,),
                                      dtype=self.dtype,
                                      initializer=keras.initializers.get('zeros'),
                                      regularizer=None,
                                      trainable=self.fc_bias,
                                      constraint=None)
        self.built = True

    def regu(self):
        if self.regularizer is not None:
            r = []
            for v in self.regulate_vars:
                r.append(tf.clip_by_value(self.regularizer(v), 1e-8, np.Inf))
            return r
        return []

    def get_input(self, step, step_batch_size, dims):
        inputs = tf.zeros((step, step_batch_size, dims), dtype=self.dtype)
        references = tf.zeros((step_batch_size, dims), dtype=self.dtype)
        mask = tf.ones((step, step_batch_size), dtype='bool')
        return inputs, references, mask

    # inputs --- (steps, step_batch_size, f_integrated_top_h_units)
    # output --- (steps, step_batch_size, n_classes) + (steps, step_batch_size, 2) + new_references
    # references --- (step_batch_size, f_integrated_top_h_units) --- before fc
    @tf.function
    def call(self, inputs, references=None, mask=None, training=None):
        inputs = tf.cast(inputs, self.dtype)
        new_references = tf.stop_gradient(inputs[-1])
        if references is None:
            references = tf.zeros_like(inputs[-1], dtype=self.dtype)
        else:
            references = tf.cast(references, self.dtype)
        references = tf.stop_gradient(tf.concat([tf.expand_dims(references, axis=0), inputs[:-1]], axis=0))
        if self.dr_rate and training:
            inputs1 = tf.nn.dropout(inputs, rate=self.dr_rate, noise_shape=(1, 1, None))
            inputs2 = tf.nn.dropout(inputs - references, rate=self.dr_rate, noise_shape=(1, 1, None))
        else:
            inputs1, inputs2 = inputs, inputs - references
        if mask is not None:
            exp_mask = tf.cast(tf.expand_dims(mask, axis=-1), self.dtype)
            inputs1 = tf.multiply(inputs1, exp_mask)
            inputs2 = tf.multiply(inputs2, exp_mask)
        logits1 = tf.add(tf.matmul(inputs1, self.w_n), self.fc_b_n)
        logits2 = tf.add(tf.matmul(inputs2, self.w_2), self.fc_b_2)
        result1 = tf.nn.softmax(logits1, axis=-1)
        result2 = tf.sigmoid(logits2)
        result2 = tf.concat([result2, tf.cast(1., self.dtype) - result2], axis=-1)
        if mask is not None:
            result1 = tf.multiply(result1, exp_mask)
            result2 = tf.multiply(result2, exp_mask)
            new_references = tf.multiply(new_references, exp_mask[-1])
        return result1, result2, new_references


# back to the original batch form
# need changed: outputs, attention result, mask(if has at the beginning)
class RevertBatch(keras.layers.Layer):
    def __init__(self, name, mode):
        super(RevertBatch, self).__init__(name=name, trainable=False)
        self.support_masking = True
        self.mode = mode   # in ('seq', 'batch')
        if self.mode not in ('seq', 'batch'):
            raise ValueError('parameter \'mode\' isn\'t the desired')

    def get_config(self):
        config = super(RevertBatch, self).get_config()
        config.update({
            'mode': self.mode,
        })
        return config

    @classmethod
    def from_config(cls, kwargs):
        new = {'name': kwargs['name'], 'mode': kwargs['mode']}
        return cls(**new)

    # have no weights, so have no need to get input

    # inputs --- rnn_result ---- 'seq': (seqs, steps, 1, n_class)    'batch': (steps, files, n_class)
    #            ch_attention --- 'seq': (seqs, 2, steps, 1, chs, 1/3)  'batch': (2, steps, files, chs, 1/3)
    #            win_attention --- 'seq': (seqs, 2, steps, 1, wins, chs, 1/3) 'batch': (2, steps, files, wins, chs, 1/3)
    # inputs --- list/tuple
    # mask --- 'seq': (seqs, steps, 1) ---> (epochs + padding,)     'batch': (steps, files) ---> (files, steps)
    # outputs --- rnn_result ---- 'seq': (epochs, n_class)    'batch': (files, steps, n_class)
    #            ch_attention --- 'seq': (epochs, 2, 1/3, chs)  'batch': (files, steps, 2, 1/3, chs)
    #            win_attention --- 'seq': (epochs, 2, 1/3, wins, chs) 'batch': (files, steps, 2, 1/3, wins, chs)
    def call(self, inputs, mask=None):
        if not isinstance(inputs, (list, tuple)) or len(inputs) not in (1, 2, 3, 4):
            raise ValueError('parameters passed don\'t meet the requirements')
        outputs = []
        if self.mode == 'seq':
            if mask is not None:
                mask = tf.reshape(mask, (-1,))
            for idx, inp in enumerate(inputs):
                if inp.shape.ndims == 4:   # rnn_result
                    inp = tf.reshape(inp, (-1, inp.shape[-1]))
                else:
                    aux = list(range(inp.shape.ndims))
                    aux.insert(3, aux.pop(1))
                    aux.insert(4, aux.pop())
                    inp = tf.transpose(inp, aux)
                    # (-1,) + inp.shape[3:] cast error  Dimension -1 must be >= 0
                    aux = tf.concat((tf.constant([-1], dtype='int32'), tf.constant(inp.shape[3:])), axis=0)
                    inp = tf.reshape(inp, aux)
                if mask is not None:
                    inp = tf.gather(inp, tf.reshape(tf.where(mask), (-1,)), axis=0)
                outputs.append(inp)
        else:
            if mask is not None:
                mask = tf.transpose(mask, (1, 0))
            for idx, inp in enumerate(inputs):
                if inp.shape.ndims == 3:    # rnn_result
                    inp = tf.transpose(inp, (1, 0, 2))
                else:
                    aux = list(range(inp.shape.ndims))
                    aux.insert(2, aux.pop(0))
                    aux.insert(3, aux.pop())
                    inp = tf.transpose(inp, aux)
                if mask is not None:
                    exp_dims = inp.shape.ndims - mask.shape.ndims
                    inp = tf.multiply(inp, tf.cast(tf.reshape(mask, mask.shape + (1,) * exp_dims), inp.dtype))
                outputs.append(inp)
        return tuple(outputs), mask


# =============================================== experiment use =======================================================


