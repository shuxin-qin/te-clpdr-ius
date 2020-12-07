# -*- coding: utf-8 -*-

import numpy as np
import logging
import os
import time

CHARS1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 
          'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
          'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','_']

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z','_'
         ]
DICT = {'A01':'京','A02':'津','A03':'沪','B02':'蒙',
        'S01':'皖','S02':'闽','S03':'粤','S04':'甘',
        'S05': '贵', 'S06': '鄂', 'S07': '冀', 'S08': '黑', 'S09': '湘',
        'S10': '豫', 'S12': '吉', 'S13': '苏', 'S14': '赣', 'S15': '辽',
        'S17': '川', 'S18': '鲁', 'S22': '浙',
        'S30':'渝', 'S31':'晋', 'S32':'桂', 'S33':'琼', 'S34':'云', 'S35':'藏',
        'S36':'陕','S37':'青', 'S38':'宁', 'S39':'新'}

CHARS_PRC = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
             '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
             '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
             '新']

CHARS_PRA = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
             'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z']

CHARS_NUM = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
             'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z']

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

NUM_CHARS = len(CHARS)

#将车牌号转为字典中的值
def encode_label(s):
    label = np.zeros([len(s)])
    for i, c in enumerate(s):
        label[i] = CHARS_DICT[c]
    return label

#将车牌号转为字典中的值
def encode_label_multi(s, idx=2):
    label = np.zeros([len(s)])
    label_up = np.zeros([idx])
    label_down = np.zeros([len(s)-idx])
    for i, c in enumerate(s):
        label[i] = CHARS_DICT[c]
        if i < idx:
            label_up[i] = CHARS_DICT[c]
        else:
            label_down[i-idx] = CHARS_DICT[c]
    return label, label_up, label_down

#判断序列是否符合车牌定义规则
def is_legal_lpnumber(detect_list, length=[7, 8]):
    flag = False
    length_d = len(detect_list)
    if length_d in length:
        if (detect_list[0] in CHARS_PRC) and (detect_list[1] in CHARS_PRA):
            flag = True
            for i in range(2, length_d):
                if detect_list[i] not in CHARS_NUM:
                    flag = False
                    break
    return flag

def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def decode_sparse_tensor(sparse_tensor, flag=0):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    dict_x = {}
    num_x = sparse_tensor[2][0]
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
        dict_x[i] = 1
    decoded_indexes.append(current_seq)
    result = []
    j = 0
    for i in range(num_x):
        if i in dict_x.keys():
            result.append(decode_a_seq(decoded_indexes[j], sparse_tensor, flag=flag))
            j += 1
        else:
            result.append([])
    return result


def decode_a_seq(indexes, spars_tensor, flag=0):
    decoded = []
    for m in indexes:
        if flag == 0:
            strr = CHARS[spars_tensor[1][m]]
        elif flag == 1:
            strr = CHARS1[spars_tensor[1][m]]
        decoded.append(strr)
    return decoded

#打印日志到控制台和log_path下的txt文件
def get_logger(log_path='log_path'):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    timer=time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime())
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s]   %(asctime)s    %(message)s')
    txthandle = logging.FileHandler((log_path+'/'+timer+'log.txt'))
    txthandle.setFormatter(formatter)
    logger.addHandler(txthandle)
    return logger
