import os, argparse, datetime, time, re, collections, random, copy
import pickle, json
import tensorflow as tf
from transformers import XLMTokenizer
from tqdm import tqdm
import numpy as np


def load_mscoco(type='train'):
    '''
    wiki_{type}_ic.json
    [{"inp": ["Homarus", "gammarus", "known", ...], "ref": "Homarus gammarus , known as the .... "}, ... ]

    max : rate까지 최대 선택
    fix : rate 고정하여 선택
    isMask False : masking 하지 않음
    '''

    print(f'\nload MSCOCO dataset ({type})........\n')
    with open(f'data/MSCOCO/k_split/{type}.json', 'r') as f:
        json_ = json.load(f)
    img_name = []
    cap = []
    for ele in json_:
        img_name.append(ele['filename'])
        cap.append(ele['sentence'])
    return img_name, cap


def build_data_loader(args, tokenizer, type='train'):

    def map_func(img_name, cap):
        file_path = 'data/MSCOCO/preprocessing/feature/' + img_name.decode('utf-8') + '.npy'
        img_tensor = np.load(file_path)
        img_tensor = img_tensor.reshape((-1))
        cap_ = cap.decode('utf-8')
        label = tokenizer.encode(cap_)
        return img_tensor, label

    def encode(item1, item2):
        result_input, result_label = tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32])
        return result_input, result_label

    img_name, cap = load_mscoco(type)
    length = len(img_name)
    BUFFER_SIZE = args.dataset_bufferSize
    BATCH_SIZE = args.dataset_batchSize
    dataset = tf.data.Dataset.from_tensor_slices((img_name, cap))
    # train_dataset <TensorSliceDataset shapes: (2,), types: tf.string>
    dataset = dataset.map(encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # cache the dataset to memory to get a speedup while reading from it.
    dataset = dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, length

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wiki_shuffle', type=str2bool, default=True, required=False,
                        help="determine inp shuffle")
    parser.add_argument('--wiki_isStopword', type=str2bool, default=True, required=False,
                        help="determine inp shuffle")
    parser.add_argument('--wiki_isMask', type=str2bool, default=False, required=False,
                        help="determine inp masking")
    parser.add_argument('--wiki_maskRate', type=float, default=0.3, required=False,
                        help="inp mask rate")
    parser.add_argument('--enc_word_max_length', type=int, default=20, required=False,
                        help="encoder input max length")
    parser.add_argument('--dec_word_max_length', type=int, default=39, required=False,
                        help="decoder input max length")
    parser.add_argument('--dec_max_length', type=int, default=100, required=False,
                        help="decoder input max length")
    parser.add_argument('--enc_max_first', default=False, type=str2bool, required=False,
                        help="determine encoder length limit")
    parser.add_argument("--isMax", default=False, type=str2bool, required=False,
                        help="keyword max or fix select")
    parser.add_argument("--file_path", default="../data/preprocessing/wiki/", type=str, required=False,
                        help="wiki data path")
    parser.add_argument('--wiki_bufferSize', type=int, default=20000, required=False,
                        help="dataset buffer size")
    parser.add_argument('--wiki_batchSize', type=int, default=60, required=False,
                        help="dataset batch size")
    parser.add_argument("--positional_encoding", default=False, type=str2bool, required=False,
                        help="create or not encoder positional encoding")
    parser.add_argument("--wiki_isMax", default=False, type=str2bool, required=False,
                        help="keyword max or fix select")
    args = parser.parse_args()
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')

    data_loader = build_data_loader(args, tokenizer, 'valid')
    for (inp, ref) in data_loader:
        # print(inp)
        # print(ref)
        print(f'#inp : {inp.shape[1]}\t #ref : {ref.shape[1]}', end='\n\n')
        # exit()