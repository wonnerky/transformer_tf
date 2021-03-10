import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import random, copy
import pickle, os
from model_keyword_transformer import Transformer as Tr_keyword
from model_original_transformer import Transformer as Tr_orig
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


#tf.enable_eager_execution()

# 소스: Keyword
# 타겟: Text

MASK_PROB = 0.15
MAX_PRED_PER_SEQ = 20
rng = random.Random(12345)
MAX_LENGTH = 180


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    # return tf.reduce_mean(loss_)
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def encode(keyword, text):
    input_tokens = tokenizer_en.encode(keyword.numpy())
    label_tokens = tokenizer_en.encode(text.numpy())
    source = [tokenizer_en.vocab_size] + input_tokens + [tokenizer_en.vocab_size + 1]
    target = [tokenizer_en.vocab_size] + label_tokens + [tokenizer_en.vocab_size + 1]
    return source, target


def tf_encode(data):
    result_input, result_label = tf.py_function(encode, [data[0], data[1]], [tf.int64, tf.int64])
    return result_input, result_label


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def mask_sequence(token_ids):
    cand_indexes = []
    for (i, token) in enumerate(token_ids):
        cand_indexes.append([i])

    rng.shuffle(cand_indexes)
    tokens_after_masking = token_ids[:]
    num_to_predict = min(MAX_PRED_PER_SEQ, max(1, int(round(len(token_ids) * MASK_PROB))))
    masked_tokens = []
    covered_indexes = set()

    for index_set in cand_indexes:
        if len(masked_tokens) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_tokens) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = 1
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = index
                # 10% of the time, replace with random word
                else:
                    masked_token = rng.randint(1, tokenizer_en.vocab_size)

            tokens_after_masking[index] = masked_token
            masked_tokens.append(masked_token)

    assert len(masked_tokens) <= num_to_predict
    assert len(tokens_after_masking) == len(token_ids)

    return tokens_after_masking


def filter_max_length(x, y, max_length = MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


def evaluate(inp_sentence, model):
    start_token = [tokenizer_en.vocab_size]
    end_token = [tokenizer_en.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer_en.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = model(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_en.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights



'''
train, test, valid
ourput: train = {'input': [keyword1, keyword2, ... ], 'label': [text1, text2, ... ]}
'''
def load_wiki_dataset(file_path, flag=1, shuffle=True, isStopword=False, isNoun=False, isMask=False, maskRate=0.0, metric_path=None):
    test = {}
    test_shu = {}
    if isStopword:
        if flag == 0:
            with open(f'{file_path}test.txt', 'rb') as f:
                test_labels = pickle.load(f)
            with open(f'{file_path}test_keyword_stopword.txt', 'rb') as f:
                test_data = pickle.load(f)
        elif flag == 1:
            with open(f'{file_path}test_one_sentence.txt', 'rb') as f:
                test_labels = pickle.load(f)
            with open(f'{file_path}test_one_sentence_keyword_stopword.txt', 'rb') as f:
                test_data = pickle.load(f)
        else:
            raise ValueError('wrong load data flag!!!')
    elif isNoun:
        if flag == 0:
            with open(f'{file_path}test.txt', 'rb') as f:
                test_labels = pickle.load(f)
            with open(f'{file_path}test_keyword_noun.txt', 'rb') as f:
                test_data = pickle.load(f)
        elif flag == 1:
            with open(f'{file_path}test_one_sentence.txt', 'rb') as f:
                test_labels = pickle.load(f)
            with open(f'{file_path}test_one_sentence_keyword_noun.txt', 'rb') as f:
                test_data = pickle.load(f)
        else:
            raise ValueError('wrong load data flag!!!')
    else:
        if flag == 0:
            with open(f'{file_path}test.txt', 'rb') as f:
                test_labels = pickle.load(f)
            with open(f'{file_path}test_keyword.txt', 'rb') as f:
                test_data = pickle.load(f)
        elif flag == 1:
            with open(f'{file_path}test_one_sentence.txt', 'rb') as f:
                test_labels = pickle.load(f)
            with open(f'{file_path}test_one_sentence_keyword.txt', 'rb') as f:
                test_data = pickle.load(f)
        else:
            raise ValueError('wrong load data flag!!!')
    if isMask:
        # if shuffle:
        #     test_data = masking_keyword_pe(test_data, maskRate)
        # else:
        #     test_data = masking_keyword_wo_pe(test_data, maskRate)
        test_data = masking_keyword_wo_pe(test_data, maskRate)
        # non shuffle keyword
    test['input'] = keyword_to_text(test_data, False)
    test_shu['input'] = keyword_to_text(test_data, True)
    test['label'] = copy.deepcopy(test_labels)
    test_shu['label'] = copy.deepcopy(test_labels)
    with open(f'{metric_path}/keyword.list', 'wb') as f:
        pickle.dump(test_data, f)
    with open(f'{metric_path}/test.dict', 'wb') as f:
        pickle.dump(test, f)
    with open(f'{metric_path}/test_shuffle.dict', 'wb') as f:
        pickle.dump(test_shu, f)
    print('test dataset save complete!!')
    return test, test_shu


def keyword_to_text(keywords, shuffle=True):
    if shuffle:
        for line in keywords:
            random.shuffle(line)
    new_lines = []
    for line in keywords:
        text = ''
        for li in line:
            text += f'{li} '
        new_lines.append(text)
    input_lines = new_lines
    return input_lines


def masking_keyword_pe(keywords, maskRate):
    rng = random.Random()
    result = []
    for keywords_ in keywords:
        rate = rng.uniform(0.0, maskRate+0.01)
        if rate >= maskRate:
            rate = maskRate
        cnt = int(len(keywords_) * rate)
        result_ = rng.sample(keywords_, len(keywords_) - cnt)
        result.append(result_)
    return result


def masking_keyword_wo_pe(keywords_orig, maskRate):
    rng = random.Random()
    keywords = copy.deepcopy(keywords_orig)
    result = []
    for keywords_ in keywords:
        rate = rng.uniform(0.0, maskRate+0.01)
        if rate >= maskRate:
            rate = maskRate
        cnt = int(len(keywords_) * rate)
        del_list_ = rng.sample(keywords_, cnt)
        for ele in del_list_:
            del keywords_[keywords_.index(ele)]
        result.append(keywords_)
    return result


def dict_to_list(data):
    output_list = []
    temp = []
    input = data['input']
    label = data['label']
    for i in range(len(input)):
        temp.append(input[i])
        temp.append(label[i])
        output_list.append(temp)
        temp = []
    return output_list


def wiki_dataset(file_path, flag=1, shuffle=True, isStopword=False, isNoun=False, isMask=False, maskRate=0.0, metric_path=None):
    print(f'dataset mode : flag({flag}), shuffle({shuffle}), isStopword({isStopword}), isNoun({isNoun}), isMask({isMask}), maskRate({maskRate}), matric_path({metric_path}')
    _test, _test_shu = load_wiki_dataset(file_path, flag, shuffle, isStopword, isNoun, isMask, maskRate, metric_path)
    test = dict_to_list(_test)
    test_shu = dict_to_list(_test_shu)
    return test, test_shu


def translate(sentence, model_list):
    predict_list = []
    for _model in model_list:
        result, _ = evaluate(sentence, _model)
        predict = tokenizer_en.decode(([i for i in result if i < tokenizer_en.vocab_size]))
        predict_list.append(predict)

    return predict_list


if __name__ == '__main__':
    print("===============  making sentence  ===============")
    isStopword = True
    isNoun = False
    if isStopword:
        file_path = './data/preprocessing/wiki103_stopword/'
    elif isNoun:
        file_path = './data/preprocessing/wiki103_noun/'
    else:
        file_path = './data/preprocessing/wiki103/'
    check_data = 'test'
    shuffle = False
    isMask = True
    maskRate = 0.8

    # load wiki dataset and make tensor
    metric_path = './metric/compare/0813'
    if not os.path.isdir(metric_path):
        os.makedirs(metric_path)
    test, test_shu = wiki_dataset(file_path, shuffle=shuffle, isStopword=isStopword, isNoun=isNoun, isMask=isMask, maskRate=maskRate, metric_path=metric_path)
    test = random.Random(100).sample(test, 4000)
    test_shu = random.Random(100).sample(test_shu, 4000)
    total_data_len = len(test)
    total_data_len_shu = len(test_shu)

    print(f'test dataset length : {len(test)}')

    test = tf.Variable(test, tf.string)
    test_shu = tf.Variable(test_shu, tf.string)

    check_dataset = tf.data.Dataset.from_tensor_slices(test)
    check_dataset_shu = tf.data.Dataset.from_tensor_slices(test_shu)
    print('wiki data load complete!')

    tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file('en_tfds_wmt8K.MASK')
    print("corpus loading finished!")

    BUFFER_SIZE = 20000
    BATCH_SIZE = 20
    # train_dataset <TensorSliceDataset shapes: (2,), types: tf.string>
    check_dataset = check_dataset.map(tf_encode)
    check_dataset_shu = check_dataset_shu.map(tf_encode)
    check_dataset = check_dataset.filter(filter_max_length)
    check_dataset_shu = check_dataset_shu.filter(filter_max_length)
    check_dataset = check_dataset.shuffle(total_data_len)
    check_dataset_shu = check_dataset_shu.shuffle(total_data_len_shu)

    # cache the dataset to memory to get a speedup while reading from it.
    print("dataset encoding finished")

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8

    input_vocab_size = tokenizer_en.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2
    dropout_rate = 0.1
    print("hyperparameters confirmed")

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # model_dict = {'pe_15': 'orig', 'pe_30': 'orig', 'pe_50': 'orig', 'wo_pe_15': 'keyword', 'wo_pe_30': 'keyword', 'wo_pe_50': 'keyword', 'wo_pe_30_init': 'keyword'}
    # for key, value in model_dict.items():
    #     if value == 'orig':
    #         key =

    tr_15 = Tr_orig(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)

    tr_30 = Tr_orig(num_layers, d_model, num_heads, dff,
                       input_vocab_size, target_vocab_size,
                       pe_input=input_vocab_size,
                       pe_target=target_vocab_size,
                       rate=dropout_rate)

    tr_50 = Tr_orig(num_layers, d_model, num_heads, dff,
                       input_vocab_size, target_vocab_size,
                       pe_input=input_vocab_size,
                       pe_target=target_vocab_size,
                       rate=dropout_rate)

    tr_wo_15 = Tr_keyword(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)

    tr_wo_30 = Tr_keyword(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

    tr_wo_50 = Tr_keyword(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

    tr_wo_30_init = Tr_keyword(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

    tr_wo_30_shu = Tr_keyword(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

    tr_wo_30_shu_init = Tr_keyword(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)

    tr_wo_80_shu = Tr_keyword(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)

    directory = "./ckpt/check_gen/"
    checkpoint_path = directory + 'stop_wo_pe_mask_shuffle_30'
    ckpt_tr_wo_30_shu = tf.train.Checkpoint(transformer=tr_wo_30_shu, optimizer=optimizer)
    ckpt_manager_tr_wo_30_shu = tf.train.CheckpointManager(ckpt_tr_wo_30_shu, checkpoint_path, max_to_keep=10)
    if ckpt_manager_tr_wo_30_shu.latest_checkpoint:
        ckpt_tr_wo_30_shu.restore(ckpt_manager_tr_wo_30_shu.latest_checkpoint)
        print('         !!!!! Latest checkpoint restored!!!!!')
        print('              ', ckpt_manager_tr_wo_30_shu.latest_checkpoint)
    else:
        print('         !!!!! NO checkpoint restored - rank keyword shuffle!!!!!')
        exit()

    checkpoint_path = directory + 'stop_wo_pe_mask_shuffle_30_init'
    ckpt_tr_wo_30_shu_init = tf.train.Checkpoint(transformer=tr_wo_30_shu_init, optimizer=optimizer)
    ckpt_manager_tr_wo_30_shu_init = tf.train.CheckpointManager(ckpt_tr_wo_30_shu_init, checkpoint_path, max_to_keep=10)
    if ckpt_manager_tr_wo_30_shu_init.latest_checkpoint:
        ckpt_tr_wo_30_shu_init.restore(ckpt_manager_tr_wo_30_shu_init.latest_checkpoint)
        print('         !!!!! Latest checkpoint restored!!!!!')
        print('              ', ckpt_manager_tr_wo_30_shu_init.latest_checkpoint)
    else:
        print('         !!!!! NO checkpoint restored - rank keyword shuffle!!!!!')
        exit()

    checkpoint_path = directory + 'stop_wo_pe_mask_shuffle_80'
    ckpt_tr_wo_80_shu = tf.train.Checkpoint(transformer=tr_wo_80_shu, optimizer=optimizer)
    ckpt_manager_tr_wo_80_shu = tf.train.CheckpointManager(ckpt_tr_wo_80_shu, checkpoint_path, max_to_keep=10)
    if ckpt_manager_tr_wo_80_shu.latest_checkpoint:
        ckpt_tr_wo_80_shu.restore(ckpt_manager_tr_wo_80_shu.latest_checkpoint)
        print('         !!!!! Latest checkpoint restored!!!!!')
        print('              ', ckpt_manager_tr_wo_80_shu.latest_checkpoint)
    else:
        print('         !!!!! NO checkpoint restored - rank keyword shuffle!!!!!')
        exit()

    directory = "./ckpt/check_gen/"
    checkpoint_path = directory + 'stop_pe_mask_shuffle_15'
    ckpt_tr_15 = tf.train.Checkpoint(transformer=tr_15, optimizer=optimizer)
    ckpt_manager_tr_15 = tf.train.CheckpointManager(ckpt_tr_15, checkpoint_path, max_to_keep=10)
    if ckpt_manager_tr_15.latest_checkpoint:
        ckpt_tr_15.restore(ckpt_manager_tr_15.latest_checkpoint)
        print('         !!!!! Latest checkpoint restored!!!!!')
        print('              ', ckpt_manager_tr_15.latest_checkpoint)
    else:
        print('         !!!!! NO checkpoint restored - rank keyword shuffle!!!!!')
        exit()

    directory = "./ckpt/check_gen/"
    checkpoint_path = directory + 'stop_pe_mask_shuffle_30'
    ckpt_tr_30 = tf.train.Checkpoint(transformer=tr_30, optimizer=optimizer)
    ckpt_manager_tr_30 = tf.train.CheckpointManager(ckpt_tr_30, checkpoint_path, max_to_keep=10)
    if ckpt_manager_tr_30.latest_checkpoint:
        ckpt_tr_30.restore(ckpt_manager_tr_30.latest_checkpoint)
        print('         !!!!! Latest checkpoint restored!!!!!')
        print('              ', ckpt_manager_tr_30.latest_checkpoint)
    else:
        print('         !!!!! NO checkpoint restored - rank keyword shuffle!!!!!')
        exit()

    directory = "./ckpt/check_gen/"
    checkpoint_path = directory + 'stop_pe_mask_shuffle_50'
    ckpt_tr_50 = tf.train.Checkpoint(transformer=tr_50, optimizer=optimizer)
    ckpt_manager_tr_50 = tf.train.CheckpointManager(ckpt_tr_50, checkpoint_path, max_to_keep=10)
    if ckpt_manager_tr_50.latest_checkpoint:
        ckpt_tr_50.restore(ckpt_manager_tr_50.latest_checkpoint)
        print('         !!!!! Latest checkpoint restored!!!!!')
        print('              ', ckpt_manager_tr_50.latest_checkpoint)
    else:
        print('         !!!!! NO checkpoint restored - rank keyword shuffle!!!!!')
        exit()

    directory = "./ckpt/check_gen/"
    checkpoint_path = directory + 'stop_wo_pe_mask_wo_shuffle_15'
    ckpt_tr_wo_15 = tf.train.Checkpoint(transformer=tr_wo_15, optimizer=optimizer)
    ckpt_manager_tr_wo_15 = tf.train.CheckpointManager(ckpt_tr_wo_15, checkpoint_path, max_to_keep=10)
    if ckpt_manager_tr_wo_15.latest_checkpoint:
        ckpt_tr_wo_15.restore(ckpt_manager_tr_wo_15.latest_checkpoint)
        print('         !!!!! Latest checkpoint restored!!!!!')
        print('              ', ckpt_manager_tr_wo_15.latest_checkpoint)
    else:
        print('         !!!!! NO checkpoint restored - rank keyword shuffle!!!!!')
        exit()

    directory = "./ckpt/check_gen/"
    checkpoint_path = directory + 'stop_wo_pe_mask_wo_shuffle_30'
    ckpt_tr_wo_30 = tf.train.Checkpoint(transformer=tr_wo_30, optimizer=optimizer)
    ckpt_manager_tr_wo_30 = tf.train.CheckpointManager(ckpt_tr_wo_30, checkpoint_path, max_to_keep=10)
    if ckpt_manager_tr_wo_30.latest_checkpoint:
        ckpt_tr_wo_30.restore(ckpt_manager_tr_wo_30.latest_checkpoint)
        print('         !!!!! Latest checkpoint restored!!!!!')
        print('              ', ckpt_manager_tr_wo_30.latest_checkpoint)
    else:
        print('         !!!!! NO checkpoint restored - rank keyword shuffle!!!!!')
        exit()

    directory = "./ckpt/check_gen/"
    checkpoint_path = directory + 'stop_wo_pe_mask_wo_shuffle_50'
    ckpt_tr_wo_50 = tf.train.Checkpoint(transformer=tr_wo_50, optimizer=optimizer)
    ckpt_manager_tr_wo_50 = tf.train.CheckpointManager(ckpt_tr_wo_50, checkpoint_path, max_to_keep=10)
    if ckpt_manager_tr_wo_50.latest_checkpoint:
        ckpt_tr_wo_50.restore(ckpt_manager_tr_wo_50.latest_checkpoint)
        print('         !!!!! Latest checkpoint restored!!!!!')
        print('              ', ckpt_manager_tr_wo_50.latest_checkpoint)
    else:
        print('         !!!!! NO checkpoint restored - rank keyword shuffle!!!!!')
        exit()

    directory = "./ckpt/check_gen/"
    checkpoint_path = directory + 'stop_wo_pe_mask_wo_shuffle_30_all_init'
    ckpt_tr_wo_30_init = tf.train.Checkpoint(transformer=tr_wo_30_init, optimizer=optimizer)
    ckpt_manager_tr_wo_30_init = tf.train.CheckpointManager(ckpt_tr_wo_30_init, checkpoint_path, max_to_keep=10)
    if ckpt_manager_tr_wo_30_init.latest_checkpoint:
        ckpt_tr_wo_30_init.restore(ckpt_manager_tr_wo_30_init.latest_checkpoint)
        print('         !!!!! Latest checkpoint restored!!!!!')
        print('              ', ckpt_manager_tr_wo_30_init.latest_checkpoint)
    else:
        print('         !!!!! NO checkpoint restored - rank keyword shuffle!!!!!')
        exit()

    print('=' * 40)
    print("just before text generation!!")
    print(f'dataset status: dataset={check_data}, isStopword={isStopword}, shuffle={shuffle}')
    print('=' * 40)
    j = 0
    delimiter = '=' * 50 + '\n'
    # model_dict = {'tr_15': tr_15, 'tr_30': tr_30, 'tr_50': tr_50, 'tr_wo_15': tr_wo_15, 'tr_wo_30': tr_wo_30, 'tr_wo_50': tr_wo_50, 'tr_wo_30_init': tr_wo_30_init}
    model_list = [tr_15, tr_30, tr_50, tr_wo_15, tr_wo_30, tr_wo_50, tr_wo_30_shu]
    model_name_list = ['tr_15', 'tr_30', 'tr_50', 'tr_wo_15', 'tr_wo_30', 'tr_wo_50', 'tr_wo_30_shu']

    # test data로 score 계산
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    shu_bleu = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}
    shu_bleu_1 = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}
    shu_bleu_2 = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}
    shu_bleu_3 = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}
    shu_bleu_4 = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}
    shu_meteor_f = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}
    shu_meteor_a = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}
    shu_rouge = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}

    bleu = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}
    bleu_1 = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}
    bleu_2 = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}
    bleu_3 = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}
    bleu_4 = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}
    meteor_f = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}
    meteor_a = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}
    rouge = {'tr_15': [], 'tr_30': [], 'tr_50': [], 'tr_wo_15': [], 'tr_wo_30': [], 'tr_wo_50': [], 'tr_wo_30_shu': []}

    # for i, (inp, ref) in enumerate(tqdm(check_dataset_shu)):
    #     inp = tokenizer_en.decode(inp[1:-1])
    #     ref = tokenizer_en.decode(ref[1:-1])
    #     predict_list = translate(inp, model_list)
    #     for j in range(len(model_list)):
    #         bleu_ = sentence_bleu([ref.split()], predict_list[j].split())
    #         bleu_1_ = sentence_bleu([ref.split()], predict_list[j].split(), [1])
    #         bleu_2_ = sentence_bleu([ref.split()], predict_list[j].split(), [0, 1])
    #         bleu_3_ = sentence_bleu([ref.split()], predict_list[j].split(), [0, 0, 1])
    #         bleu_4_ = sentence_bleu([ref.split()], predict_list[j].split(), [0, 0, 0, 1])
    #         meteor_f_ = meteor_score([ref], predict_list[j], alpha=0.78, beta=0.75, gamma=0.38)
    #         meteor_a_ = meteor_score([ref], predict_list[j], alpha=0.82, beta=1.0, gamma=0.21)
    #         rouge_ = scorer.score(ref, predict_list[j])
    #         shu_bleu[model_name_list[j]].append(bleu_)
    #         shu_bleu_1[model_name_list[j]].append(bleu_1_)
    #         shu_bleu_2[model_name_list[j]].append(bleu_2_)
    #         shu_bleu_3[model_name_list[j]].append(bleu_3_)
    #         shu_bleu_4[model_name_list[j]].append(bleu_4_)
    #         shu_meteor_f[model_name_list[j]].append(meteor_f_)
    #         shu_meteor_a[model_name_list[j]].append(meteor_a_)
    #         shu_rouge[model_name_list[j]].append(rouge_)

    for i, (inp, ref) in enumerate(tqdm(check_dataset)):
        inp = tokenizer_en.decode(inp[1:-1])
        ref = tokenizer_en.decode(ref[1:-1])
        predict_list = translate(inp, model_list)
        for j in range(len(model_list)):
            bleu_ = sentence_bleu([ref.split()], predict_list[j].split())
            bleu_1_ = sentence_bleu([ref.split()], predict_list[j].split(), [1])
            bleu_2_ = sentence_bleu([ref.split()], predict_list[j].split(), [0, 1])
            bleu_3_ = sentence_bleu([ref.split()], predict_list[j].split(), [0, 0, 1])
            bleu_4_ = sentence_bleu([ref.split()], predict_list[j].split(), [0, 0, 0, 1])
            meteor_f_ = meteor_score([ref], predict_list[j], alpha=0.78, beta=0.75, gamma=0.38)
            meteor_a_ = meteor_score([ref], predict_list[j], alpha=0.82, beta=1.0, gamma=0.21)
            rouge_ = scorer.score(ref, predict_list[j])
            bleu[model_name_list[j]].append(bleu_)
            bleu_1[model_name_list[j]].append(bleu_1_)
            bleu_2[model_name_list[j]].append(bleu_2_)
            bleu_3[model_name_list[j]].append(bleu_3_)
            bleu_4[model_name_list[j]].append(bleu_4_)
            meteor_f[model_name_list[j]].append(meteor_f_)
            meteor_a[model_name_list[j]].append(meteor_a_)
            rouge[model_name_list[j]].append(rouge_)

    os.makedirs(f'{metric_path}/non_shuffle')
    os.makedirs(f'{metric_path}/shuffle')

################ shuffle ################
    # with open(f'{metric_path}/shuffle/model_compare_bleu.txt', 'wb') as f:
    #     pickle.dump(shu_bleu, f)
    # with open(f'{metric_path}/shuffle/model_compare_bleu_1.txt', 'wb') as f:
    #     pickle.dump(shu_bleu_1, f)
    # with open(f'{metric_path}/shuffle/model_compare_bleu_2.txt', 'wb') as f:
    #     pickle.dump(shu_bleu_2, f)
    # with open(f'{metric_path}/shuffle/model_compare_bleu_3.txt', 'wb') as f:
    #     pickle.dump(shu_bleu_3, f)
    # with open(f'{metric_path}/shuffle/model_compare_bleu_4.txt', 'wb') as f:
    #     pickle.dump(shu_bleu_4, f)
    # with open(f'{metric_path}/shuffle/model_compare_meteor_fluency.txt', 'wb') as f:
    #     pickle.dump(shu_meteor_f, f)
    # with open(f'{metric_path}/shuffle/model_compare_meteor_adequacy.txt', 'wb') as f:
    #     pickle.dump(shu_meteor_a, f)
    # with open(f'{metric_path}/shuffle/model_compare_rougeL.txt', 'wb') as f:
    #     pickle.dump(shu_rouge, f)

############### non shuffle ##############
    with open(f'{metric_path}/non_shuffle/model_compare_bleu.txt', 'wb') as f:
        pickle.dump(bleu, f)
    with open(f'{metric_path}/non_shuffle/model_compare_bleu_1.txt', 'wb') as f:
        pickle.dump(bleu_1, f)
    with open(f'{metric_path}/non_shuffle/model_compare_bleu_2.txt', 'wb') as f:
        pickle.dump(bleu_2, f)
    with open(f'{metric_path}/non_shuffle/model_compare_bleu_3.txt', 'wb') as f:
        pickle.dump(bleu_3, f)
    with open(f'{metric_path}/non_shuffle/model_compare_bleu_4.txt', 'wb') as f:
        pickle.dump(bleu_4, f)
    with open(f'{metric_path}/non_shuffle/model_compare_meteor_fluency.txt', 'wb') as f:
        pickle.dump(meteor_f, f)
    with open(f'{metric_path}/non_shuffle/model_compare_meteor_adequacy.txt', 'wb') as f:
        pickle.dump(meteor_a, f)
    with open(f'{metric_path}/non_shuffle/model_compare_rougeL.txt', 'wb') as f:
        pickle.dump(rouge, f)


