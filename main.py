# -*- coding: utf-8 -*-
import config
from callbacks import EarlyStoppingCallback, NFoldsTrainCallback, PredictCallback
from train import CustomTrainer, MultitaskTrainer
import model
import utils
from predict import Predictor
import numpy as np
import os
import shutil
import glob
import ntpath
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# 检查tf可用的设备
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())　


PREPRO_DIR = 'data'
DS_INFO = 'dataset_info'       # path of information of dataset
CHOICE_ARCH = 'choose_data_archive'     # save the record of data for experiment
SETS = ('train', 'val', 'test')


# 20折交叉验证则choice=(19，1, 0), repeat=20, repeat_mode='cross'
# 其他示例：eg: for 'personal' --- 103 files, choice=(71, 16, 16), repeat=5, repeat_mode='again'
def random_choice(dataset_id, info_path, choice, repeat, repeat_mode, age_mode='whole', subject=False,
                  arch_dir=CHOICE_ARCH):
    '''

    :param dataset_id:
    :param info_path: the path of 'xxx_info.npz' which about the statistic information of the dataset
    :param choice: is iterative and can not exceed three items --- must comply with the order of train, val, test
    :param repeat: the number of times for choosing
    :param repeat_mode: whether choose sets from all data again, qualified options are 'cross' and 'again'
    :param age_mode: only valid for the scene that dataset_id is 1 --- qualified options are 'whole', 'young' and None
    :param subject: the unit of choice whether is subject, if not is file
    :return:
    '''
    info = utils.get_info(dataset_id, info_path)
    if not hasattr(choice, '__iter__') or len(choice) > 3:
        raise Exception('not right parameter \'choice\'')
    sum_up = sum(choice)
    if not os.path.exists(arch_dir):
        os.makedirs(arch_dir)
    else:
        shutil.rmtree(arch_dir)
        os.makedirs(arch_dir)

    # union --- 1D array
    def _save(union, extend=0):
        accu = 0
        if not extend:
            for se, na in zip(choice, SETS):
                if se > 0:
                    aux = union[accu: accu + se]
                    aux = np.random.permutation(aux)
                    np.savetxt(os.sep.join([arch_dir, na + '.txt']), np.reshape(aux, (1, -1)),
                               fmt='%d', delimiter=',', newline='\n')
                    accu += se
        else:
            for se, na in zip(choice, SETS):
                if se > 0:
                    aux = union[accu: accu + se]
                    aux = np.reshape(np.vstack((aux, aux + extend)).T, (-1,))
                    aux = np.random.permutation(aux)
                    np.savetxt(os.sep.join([arch_dir, na + '.txt']), np.reshape(aux, (1, -1)),
                               fmt='%d', delimiter=',', newline='\n')
                    accu += se

    # union --- 1D array
    def _extra_save(union, extend=0):
        accu = 0
        if not extend:
            for se, na in zip(choice, SETS):
                if se > 0:
                    with open(os.sep.join([arch_dir, na + '.txt']), 'a+') as f:
                        aux = union[accu: accu + se]
                        aux = np.random.permutation(aux)
                        np.savetxt(f, np.reshape(aux, (1, -1)), fmt='%d', delimiter=',', newline='\n')
                    accu += se
        else:
            for se, na in zip(choice, SETS):
                if se > 0:
                    with open(os.sep.join([arch_dir, na + '.txt']), 'a+') as f:
                        aux = union[accu: accu + se]
                        aux = np.reshape(np.vstack((aux, aux + extend)).T, (-1,))
                        aux = np.random.permutation(aux)
                        np.savetxt(f, np.reshape(aux, (1, -1)), fmt='%d', delimiter=',', newline='\n')
                    accu += se

    if dataset_id == 1 and subject:     # 本人实验仅使用SC公共数据库，ST公共数据库也是可以选择个体的
        subject_indices = utils.prepare_subjects(1, info['inventory'])     # Int64Index of DataFrame
        if repeat_mode == 'cross':
            fetch = utils.choose_subjects(dataset_id, subject_indices, sum_up, age_mode, info['inventory'])   # 1D array
            _save(fetch, 1)
            temp = fetch[: sum(choice[: 2])]
            n = len(temp) - choice[1]
            for i in range(1, repeat):
                start = n - i * choice[1]
                val = temp[start: start + choice[1]]
                with open(os.sep.join([arch_dir, SETS[1] + '.txt']), 'a+') as f:
                    assist = np.reshape(np.vstack((val, val + 1)).T, (-1,))
                    assist = np.random.permutation(assist)
                    np.savetxt(f, np.reshape(assist, (1, -1)), fmt='%d', delimiter=',', newline='\n')
                with open(os.sep.join([arch_dir, SETS[0] + '.txt']), 'a+') as f:
                    tr = np.setdiff1d(temp, val)
                    tr = np.reshape(np.vstack((tr, tr + 1)).T, (-1,))
                    np.random.shuffle(tr)
                    np.savetxt(f, np.reshape(tr, (1, -1)), fmt='%d', delimiter=',', newline='\n')
        elif repeat_mode == 'again':
            fetch = utils.choose_subjects(dataset_id, subject_indices, sum_up, age_mode, info['inventory'])
            _save(fetch, 1)
            for i in range(1, repeat):
                fetch = utils.choose_subjects(dataset_id, subject_indices, sum_up, age_mode, info['inventory'])
                _extra_save(fetch, 1)
    else:
        if repeat_mode == 'cross':
            fetch = utils.choose_files(dataset_id, sum_up, age_mode, info['inventory'])    # 1D array
            _save(fetch)
            temp = fetch[: sum(choice[: 2])]
            n = len(temp) - choice[1]
            for i in range(1, repeat):
                start = n - i * choice[1]
                val = temp[start: start + choice[1]]
                with open(os.sep.join([arch_dir, SETS[1] + '.txt']), 'a+') as f:
                    np.savetxt(f, np.reshape(np.random.permutation(val), (1, -1)),
                               fmt='%d', delimiter=',', newline='\n')
                with open(os.sep.join([arch_dir, SETS[0] + '.txt']), 'a+') as f:
                    tr = np.setdiff1d(temp, val)
                    np.random.shuffle(tr)
                    np.savetxt(f, np.reshape(tr, (1, -1)), fmt='%d', delimiter=',', newline='\n')
        elif repeat_mode == 'again':
            fetch = utils.choose_files(dataset_id, sum_up, age_mode, info['inventory'])
            _save(fetch)
            for i in range(1, repeat):
                fetch = utils.choose_files(dataset_id, sum_up, age_mode, info['inventory'])
                _extra_save(fetch)


# please make sure that have operated 'random_choice' method first
def get_sets(ori_dir=CHOICE_ARCH):
    li = glob.glob(os.sep.join([ori_dir, '']) + '*.txt')

    def _fetch(file):
        with open(file, 'r') as f:
            info = f.readlines()
            info = list(line.rstrip().split(',') for line in info)
            info = list(map(lambda aa: list(int(item) for item in aa), info))
            info = np.asarray(info)
        return info

    set_dict = {}
    for l in li:
        name = ntpath.basename(l).replace('.txt', '')
        if name == SETS[0]:
            set_dict[SETS[0]] = _fetch(l)
        elif name == SETS[1]:
            set_dict[SETS[1]] = _fetch(l)
        elif name == SETS[2]:
            set_dict[SETS[2]] = _fetch(l)

    return set_dict
                

def train_normal():
    configurer = config.RunConfig()
    # （首先是小一点的数据集，全集范围内，使用提前结束；然后使用有独立测试集的young模式的稍大的数据集，进行交叉验证）
    # （使用提前结束的参数，直到有满意的值或超过一个大的epoch数）
    id_sets = get_sets()     # 得到是文件索引信息
    manager = config.ExperControl(1, id_sets, 0, DS_INFO, PREPRO_DIR, True, False)    # 初始化时会将文件索引变成文件路径

    base_model = model.ABiGRUBaseForStackBi('two_layer_stack_bi', configurer)

    # ====================================================================================================
    # callback = EarlyStoppingCallback('sample', configurer.patience, configurer.callback_monitor)    # 不知道该设置patience为多少
    callback = EarlyStoppingCallback(patience=configurer.patience, monitor=configurer.callback_monitor)
    trainer = CustomTrainer(configurer, manager, callback, 'seq')  # config.seq_len, config.file_num
    trainer.train(base_model, configurer.multitask)
    # note that setting such as file_num must remain the same in the case, otherwise model will errors
    if id_sets.get('test', None) is not None:
        if len(id_sets['test']) == len(id_sets['train']):
            predict_callback = PredictCallback()
            predictor = Predictor(configurer, manager, predict_callback, 'seq', 60)
            predictor.predict(base_model, configurer.multitask, True)


# 继续训练分两种 1.本次epoch没跑完    2.repeat
def train_continue(model_path, reset_opt=False):
    configurer = config.RunConfig()   # 记得按照层数修改相应的rnn参数
    # 2 bi_layer_gru_wrapper_explorer; 1 bi_layer_stack_gru_explorer
    base_model = model.ABiGRUBaseForStackBi('', configurer)
    # 生成模型weights
    input = base_model.get_input()
    base_model(*input[:-1], True, input[-1])   # the last one is 'training'   if multitask --- *input[:-1], True, input[-1]
    # 替换成之前训练的weights
    opt_weights = utils.load_same_model_weights(base_model, model_path)
    if reset_opt:
        opt_weights = None
    #utils.from_file_set_model_weights(base_model, path)
    id_sets = get_sets()
    manager = config.ExperControl(configurer.dataset, id_sets, configurer.repeat_id, DS_INFO, PREPRO_DIR, True, True)
    #callback = EarlyStoppingCallback('sample', configurer.patience, configurer.callback_monitor)
    # =========================================================================================================
    callback = EarlyStoppingCallback(patience=configurer.patience, monitor=configurer.callback_monitor)
    trainer = CustomTrainer(configurer, manager, callback, 'seq')
    trainer.train(base_model, configurer.multitask, opt_weights)
    # note that setting such as file_num must remain the same in the case, otherwise model will errors
    if id_sets.get('test', None) is not None:
        if len(id_sets['test']) == len(id_sets['train']):
            predict_callback = PredictCallback()
            predictor = Predictor(configurer, manager, predict_callback, 'seq', 60)
            predictor.predict(base_model, configurer.multitask, True)


def successive_nfolds(model_path):       # 属于除第一折外的精调
    configurer = config.RunConfig()
    # 2 bi_layer_gru_wrapper_explorer; 1 bi_layer_stack_gru_explorer
    base_model = model.ABiGRUBaseForStackBi('', configurer)
    # 生成模型weights
    input = base_model.get_input()
    base_model(*input[:-1], True, input[-1])   # 第一次才要，要先build
    current_repeat = configurer.repeat_id
    id_sets = get_sets()
    manager = config.ExperControl(configurer.dataset, id_sets, current_repeat, DS_INFO, PREPRO_DIR, True, True)
    callback = NFoldsTrainCallback(patience=configurer.patience, monitor=configurer.callback_monitor)
    trainer = CustomTrainer(configurer, manager, callback, 'seq')
    print('=*' * 18 + ' start {} folds training '.format(manager.repeats) + '*=' * 18 + '\n')
    while current_repeat < manager.repeats:
        # 替换成初折训练的weights
        _ = utils.load_same_model_weights(base_model, model_path)
        print('\n' + '-*' * 19 + ' No.{}_fold '.format(current_repeat + 1) + '*-' * 19)
        trainer.train_successive_nfolds(current_repeat, base_model, configurer.multitask, fully_test_end=False)
        # note that setting such as file_num must remain the same in the case, otherwise model will errors
        if id_sets.get('test', None) is not None:
            if len(id_sets['test']) == len(id_sets['train']):
                predict_callback = PredictCallback()
                predictor = Predictor(configurer, manager, predict_callback, 'seq', 60)
                predictor.predict(base_model, configurer.multitask, True)
        current_repeat += 1
        manager.set_repeat_id(current_repeat)
        manager.set_new_sets()
    print('=*' * 18 + ' stop {} folds training '.format(manager.repeats) + '*=' * 18 + '\n')


def parallel_nfolds(model_paths, repeat_ids):
    configurer = config.RunConfig()
    # 2 bi_layer_gru_wrapper_explorer; 1 bi_layer_stack_gru_explorer
    base_model = model.ABiGRUBaseForStackBi('', configurer)
    # 生成模型weights
    input = base_model.get_input()
    base_model(*input[:-1], True, input[-1])  # 第一次才要，要先build
    id_sets = get_sets()
    manager = config.ExperControl(configurer.dataset, id_sets, 0, DS_INFO, PREPRO_DIR, True, True)
    callback = NFoldsTrainCallback(patience=configurer.patience, monitor=configurer.callback_monitor)
    trainer = CustomTrainer(configurer, manager, callback, 'seq')
    print('=*' * 18 + ' start {} folds training '.format(manager.repeats) + '*=' * 18 + '\n')
    for r, p in zip(repeat_ids, model_paths):
        manager.set_repeat_id(r)
        manager.set_new_sets()
        # 替换成初折训练的weights
        _ = utils.load_same_model_weights(base_model, p)
        print('\n' + '-*' * 19 + ' No.{}_fold '.format(r + 1) + '*-' * 19)
        trainer.train_successive_nfolds(r, base_model, configurer.multitask, fully_test_end=False)
        # note that setting such as file_num must remain the same in the case, otherwise model will errors
        if id_sets.get('test', None) is not None:
            if len(id_sets['test']) == len(id_sets['train']):
                predict_callback = PredictCallback()
                predictor = Predictor(configurer, manager, predict_callback, 'seq', 60)
                predictor.predict(base_model, configurer.multitask, True)
    print('=*' * 18 + ' stop {} folds training '.format(manager.repeats) + '*=' * 18 + '\n')


def predict(model_path, output_dir, set_dir):
    d = get_sets(set_dir)
    id_set = np.concatenate([d['train'][0], d['val'][0]], axis=0)  # 根据需要拿测试数据，输入测试manager的id_set是一维iterable object
    configurer = config.RunConfig()  # 模型参数设置
    manager = config.PredictControl(configurer.dataset, output_dir, id_set, DS_INFO, PREPRO_DIR,
                                    True)  # 初始化时会转换成路径名构成的列表

    used_model = model.ABiGRUBaseForStackBi('two_layer_stack_bi', configurer)  # 'single_bi_gru_explorer'
    # 生成模型weights
    inps = used_model.get_input()
    used_model(*inps[:-1], False, inps[-1], True)
    # 替换模型weights为已训练好的
    _ = utils.load_same_model_weights(used_model, model_path)
    # ===================================================================================================
    predict_callback = PredictCallback()
    predictor = Predictor(configurer, manager, predict_callback, 'seq', 60)
    predictor.predict(used_model, configurer.multitask, True)


if __name__ == '__main__':
    #random_choice(1, DS_INFO, (19, 1, 0), 1, 'again', 'young', True, CHOICE_ARCH)       # 先选一个repeat用于探索实验
    #train_normal()
    #train_continue(os.sep.join(['output', 'personal', 'archive', 'No.1_repeat', 'trained_model', '17.npz']))
    # predict(os.sep.join(['output', 'personal', 'archive', 'No.1_repeat', 'trained_model', '17.npz']),
    # 'predict_from_17_npz', CHOICE_ARCH)

    '''
    # 可能需要先从文件中获取学习率
    path = os.sep.join([''])
    with np.load(path, allow_pickle=True) as f:
        lr = dict(f).get('lr', None)
    if lr is not None:
        print(lr)
    train_continue(path)
    '''

'''
if __name__ == '__main__':
    path = os.sep.join(['MyFiles', 'output', 'archive', 'subject_mode', '5repeats_20folds', 'trained_model', 'bi_layer_stack_gru_explorer', '4.npz'])
    out = os.sep.join(['MyFiles', 'predict'])
    _, val, _, _ = utils.fetch_split_set(os.sep.join(['MyFiles', 'auto_upload_20201031182301', 'inventory.csv']),
                                         os.sep.join(['MyFiles', 'auto_upload_20201031182301']),
                                         os.sep.join(['MyFiles', 'output', 'selected_info', 'subject_mode',
                                       '5repeats_20folds']), 0, 0)
    predict(path, out, val)
'''






