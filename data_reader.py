# -*- coding: utf-8 -*-
'''
to read the head of EDF+/EDF files and to read the annotations of the EDF+ files
'''
import math
import numpy as np
import re


ANN = 'EDF Annotations'   # sleep-EDF expanded dataset's annotation channel label


# b is stream of ASCII characters
def byte2int(b):
    value = 0
    length = len(b)
    for i in range(length - 1, 0, -1):
        value += int((b[i] - 48) * math.pow(10, length - 1 - i))  # 48 --- '0'
    if b[0] == 45:  # '-'
        value = -value
    elif b[0] == 43:  # '+'
        pass
    else:
        value += int((b[0] - 48) * math.pow(10, length - 1))
    return value


class NRUnknownException(Exception):
    def __init__(self, msg):
        super().__init__()
        self.msg = msg

    def __str__(self):
        return str(self.msg)


# read sleep-EDF expanded dataset
class EDFReader(object):
    def __init__(self, file):
        self.file = file
        self.__head = None

    def get_head(self):
        return self.__head

    # return head information
    def read_head(self):
        with open(self.file, 'rb') as f:
            head = {}
            assert f.tell() == 0
            version = b'0       '
            assert f.read(8) == version    # read 8 bytes, check version
            subject = f.read(80)
            record_info = f.read(80)

            # start date
            startdate = f.read(8)    # b'dd.mm.yy'
            D, M, Y = startdate.split(b'.')
            D, M, Y = int(D), int(M), int(Y)
            if Y >= 85:
                Y += 1900
            else:
                Y += 2000
            head['date'] = '{}-{}-{}'.format(Y, M, D)

            # start time
            starttime = f.read(8)   # b'hh.mm.ss'
            h, m, s = starttime.split(b'.')
            h, m, s = int(h), int(m), int(s)
            head['time'] = '{}:{}:{}'.format(h, m, s)

            #head['date_time'] = '{}-{}-{} {}:{}:{}'.format(Y, M, D, h, m, s)

            head['bytes'] = int(f.read(8).rstrip())
            reserved = f.read(44).rstrip()
            if reserved != b'':
                head['file_type'] = 'EDF+'
                if reserved.split(b'+')[1] == b'D':   # read 'reserved'field
                    head['is_continued'] = False
                else:
                    head['is_continued'] = True
            else:
                head['file_type'] = 'EDF'    # EDF file is continued
            head['sum_records'] = int(f.read(8).split()[0])  # -1 implies it's unknown
            duration = float(f.read(8).strip())    # each record duration, of EDF+D or Annotation .etc is 0
            if duration > 0.:
                head['duration'] = duration
            head['sum_channels'] = int(f.read(4).split()[0])

            labels = []
            have_annotation = False
            for i in range(head['sum_channels']):
                label = f.read(16).rstrip().decode('utf-8')
                labels.append(label)
                if label == ANN:
                    have_annotation = True
            head['labels'] = labels

            transducer_types = []
            for i in range(head['sum_channels']):
                transducer_types.append(f.read(80).rstrip().decode('utf-8'))

            physical_dimensions = []
            for i in range(head['sum_channels']):
                physical_dimensions.append(f.read(8).rstrip().decode('utf-8'))

            physical_minimums = []
            for i in range(head['sum_channels']):
                physical_minimums.append(float(f.read(8).rstrip()))

            physical_maximums = []
            for i in range(head['sum_channels']):
                physical_maximums.append(float(f.read(8).rstrip()))

            digital_minimums = []
            for i in range(head['sum_channels']):
                digital_minimums.append(float(f.read(8).rstrip()))

            digital_maximums = []
            for i in range(head['sum_channels']):
                digital_maximums.append(float(f.read(8).rstrip()))

            prefilterings = []
            for i in range(head['sum_channels']):
                prefilterings.append(f.read(80).rstrip().decode('utf-8'))

            if (have_annotation is False) or (len(labels) > 1):
                head['transducer_type'] = transducer_types
                head['physical_dimension'] = physical_dimensions
                head['physical_minimum'] = np.asarray(physical_minimums)
                head['physical_maximum'] = np.asarray(physical_maximums)
                head['digital_minimum'] = np.asarray(digital_minimums)
                head['digital_maximum'] = np.asarray(digital_maximums)
                head['gains'] = (head['physical_maximum'] - head['physical_minimum']) / \
                               (head['digital_maximum'] - head['digital_minimum'])
                head['prefiltering'] = prefilterings

            samples = []   # sampled points per record
            for i in range(head['sum_channels']):
                samples.append(int(f.read(8).rstrip()))
            head['samples_per_record'] = np.asarray(samples, dtype=np.int32)
            head['sum_samples_per_record'] = np.sum(head['samples_per_record'])
            duration = head.get('duration', 0)
            if duration:
                head['sampling_rates'] = head['samples_per_record'] / duration

            f.read(32 * head['sum_channels'])
            assert f.tell() == head['bytes'], 'errors have occurred during reading the head ' \
                                              'information of {} '.format(self.file)

        self.__head = head
        return head

    def tals(self, string):
        '''
        parse string stream into TALs' pattern information
        :param string:
        :return:
        [ [float,],
        (
            Onset: float,
            Duration: float,
            Annotation: list or None
        ),...]
        '''
        pattern = r'(?P<Onset>[\+|-]\d+\.?\d*)(?P<Duration>\x15\d+\.?\d*)*(?P<is_Start>\x14)?' \
                  r'(?P<Annotation>\x14[^\x00]+)*(?:\x14\x00)'
        labels = [[]]
        for tal in re.finditer(pattern, string):
            di = tal.groupdict()
            di['Onset'] = float(di['Onset'])
            if di['is_Start'] is not None:
                labels[0].append(di['Onset'])
            labels.append((di['Onset'], float(di['Duration'].split('\x15')[1]) if di['Duration'] is not None else 0.,
                           di['Annotation'].split('\x14')[1:] if di['Annotation'] is not None else None))
        return labels

    # f is a instance of _io.BufferedReader type
    def _read_ann(self, string):
        result = self.tals(string)
        time_bias = result[0][0]  # in theory, one record has only one time bias related to file start time/date
        labels = result[1:]
        return time_bias, labels

    # ordinary signals are in units of 1D arrays, annotation signal is a list
    def _read_per_record(self, r):
        signals = []
        events = []
        time_bias = 0.
        with open(self.file, 'rb') as f:
            start = self.__head['bytes'] + self.__head['sum_samples_per_record'] * r * 2
            f.seek(start)
            for i, channel in enumerate(self.__head['labels']):
                if channel == ANN:
                    time_bias, events = self._read_ann(f.read(self.__head['samples_per_record'][i] * 2).decode('utf-8'))
                else:
                    # arr is 1D
                    arr = np.frombuffer(f.read(self.__head['samples_per_record'][i] * 2),
                                        dtype='<i2').astype(np.float32)
                    digital_minimum = self.__head['digital_minimum'][i]
                    gain = self.__head['gains'][i]
                    physical_minimum = self.__head['physical_minimum'][i]
                    signals.append((arr - digital_minimum) * gain + physical_minimum)
        return time_bias, events, signals

    # raw per record generator
    def read_data(self):
        try:
            if self.__head is None:
                self.read_head()
            if self.__head['sum_records'] == -1:
                raise NRUnknownException('the head of \'{}\' has no information about sum of records '.
                                         format(self.file))
            n = 0
            while n < self.__head['sum_records']:
                yield self._read_per_record(n)
                n += 1
        except Exception as e:
            raise e
            return










