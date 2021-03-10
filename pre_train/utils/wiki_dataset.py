import os, argparse, datetime, time, re, collections, random, copy
import pickle, json
import tensorflow as tf
from transformers import XLMTokenizer
from tqdm import tqdm


def load_wiki_dataset(args, tokenizer, type='train'):
    '''
    wiki_{type}_ic.json
    [{"inp": ["Homarus", "gammarus", "known", ...], "ref": "Homarus gammarus , known as the .... "}, ... ]

    max : rate까지 최대 선택
    fix : rate 고정하여 선택
    isMask False : masking 하지 않음
    '''
    file_path = args.file_path
    shuffle = args.wiki_shuffle
    isStopword = args.wiki_isStopword
    isMask = args.wiki_isMask
    maskRate = args.wiki_maskRate
    pe = args.positional_encoding
    isMax = args.wiki_isMax
    enc_word_max_length = args.enc_word_max_length
    dec_max_length = args.dec_max_length
    enc_max_first = args.enc_max_first
    dec_word_max_length = args.dec_word_max_length

    print(f'dataset mode : shuffle({shuffle}), isStopword({isStopword}), isMask({isMask}), maskRate({maskRate}), pe({pe})')


    def keword_to_sentence(keywords):
        output = ''
        for ele in keywords:
            output += f'{ele} '
        return output[:-1]

    def masking_keyword_shuffle_max(keywords):
        rng = random.Random()
        rate = rng.uniform(0.0, maskRate + 0.01)
        if rate >= maskRate:
            rate = maskRate
        cnt = int(len(keywords) * rate)
        result = rng.sample(keywords, len(keywords) - cnt)
        return result

    def masking_keyword_wo_shuffle_max(keywords_orig):
        rng = random.Random()
        keywords = copy.deepcopy(keywords_orig)
        rate = rng.uniform(0.0, maskRate + 0.01)
        if rate >= maskRate:
            rate = maskRate
        cnt = int(len(keywords) * rate)
        del_list_ = rng.sample(keywords, cnt)
        for ele in del_list_:
            del keywords[keywords.index(ele)]
        return keywords

    def masking_keyword_shuffle_fix(keywords):
        rng = random.Random()
        cnt = int(len(keywords) * maskRate)
        result = rng.sample(keywords, len(keywords) - cnt)
        return result

    def masking_keyword_wo_shuffle_fix(keywords_orig):
        rng = random.Random()
        keywords = copy.deepcopy(keywords_orig)
        cnt = int(len(keywords) * maskRate)
        del_list_ = rng.sample(keywords, cnt)
        for ele in del_list_:
            del keywords[keywords.index(ele)]
        return keywords

    def _processing_inputs(inp):
        if not isMask and not shuffle:
            return keword_to_sentence(inp)
        elif not isMask and shuffle:
            random.shuffle(inp)
            return keword_to_sentence(inp)
        elif isMax and shuffle:
            return keword_to_sentence(masking_keyword_shuffle_max(inp))
        elif isMax and not shuffle:
            return keword_to_sentence(masking_keyword_wo_shuffle_max(inp))
        elif not isMax and shuffle:
            return keword_to_sentence(masking_keyword_shuffle_fix(inp))
        elif not isMax and not shuffle:
            return keword_to_sentence(masking_keyword_wo_shuffle_fix(inp))

    def _processing_data(data):
        inputs = []
        refs = []
        for ele in tqdm(data):
            if enc_max_first:
                if len(ele['inp']) <= enc_word_max_length:
                    inputs.append(_processing_inputs(ele['inp']))
                    refs.append(ele['ref'])
            else:
                if len(ele['ref']) <= dec_max_length:
                    inputs.append(_processing_inputs(ele['inp']))
                    refs.append(ele['ref'])
        return inputs, refs

    output = {}
    with open(os.path.join(file_path, f'wiki_{type}_ic.json'), 'r') as f:
        data = json.load(f)
    inputs, refs = _processing_data(data)

    output['inp'] = inputs
    output['ref'] = refs
    return dict_to_list(output)


def dict_to_list(data):
    output_list = []
    temp = []
    input = data['inp']
    label = data['ref']
    for i in range(len(input)):
        temp.append(input[i])
        temp.append(label[i])
        output_list.append(temp)
        temp = []
    return output_list


def build_data_loader(args, tokenizer, type='train'):
    def filter_max_length(x, y, enc_max_length=args.enc_word_max_length, dec_max_length=args.dec_word_max_length):
        return tf.logical_and(tf.size(x) <= enc_max_length,
                              tf.size(y) <= dec_max_length)

    def encode(inp, ref):
        '''
        input : tensor text
        input.numpy() : byte word. b'text
        input.numpy().decode('utf-8') : text
        '''
        input_tokens = tokenizer.encode(inp.numpy().decode('utf-8'))
        label_tokens = tokenizer.encode(ref.numpy().decode('utf-8'))

        '''
            0 = sos, 1 = eos
            input = offensive Army continue would Allied
            [0, 4530, 489, 1862, 78, 4956, 1]
            -> [4530, 489, 1862, 78, 4956]
        '''
        source = input_tokens[1:-1]

        return source, label_tokens

    def tf_encode(data):
        result_input, result_label = tf.py_function(encode, [data[0], data[1]], [tf.int64, tf.int64])
        return result_input, result_label


    data = load_wiki_dataset(args, tokenizer, type)
    length = len(data)
    BUFFER_SIZE = args.wiki_bufferSize
    BATCH_SIZE = args.wiki_batchSize
    data = tf.Variable(data)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    # train_dataset <TensorSliceDataset shapes: (2,), types: tf.string>
    dataset = dataset.map(tf_encode)
    dataset = dataset.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    dataset = dataset.cache()
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