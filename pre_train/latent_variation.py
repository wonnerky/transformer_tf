import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import random
import pickle
from model_original_transformer import Transformer as Tr_orig
from model_latent_transformer import Transformer as Tr_latent

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


def evaluate_latent(inp_sentence1, inp_sentence2, model, oper='add'):
    start_token = [tokenizer_en.vocab_size]
    end_token = [tokenizer_en.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    input_tokens1 = start_token + tokenizer_en.encode(inp_sentence1)
    input_tokens2 = start_token + tokenizer_en.encode(inp_sentence2)
    len_text1 = len(input_tokens1)
    len_text2 = len(input_tokens2)
    mask_input = 1

    if len_text1 > len_text2:
        for i in range(len_text1 - len_text2):
            input_tokens2.append(0)
            mask_input = 1
    elif len_text1 < len_text2:
        for i in range(len_text2 - len_text1):
            input_tokens1.append(0)
            mask_input = 2

    input_tokens1 = input_tokens1 + end_token
    input_tokens2 = input_tokens2 + end_token

    encoder_input1 = tf.expand_dims(input_tokens1, 0)
    encoder_input2 = tf.expand_dims(input_tokens2, 0)

    if mask_input == 1:
        mask_input_txt = encoder_input1
    else:
        mask_input_txt = encoder_input2

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(mask_input_txt, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = model(encoder_input1,
                                               encoder_input2,
                                               output,
                                               False,
                                               enc_padding_mask,
                                               combined_mask,
                                               dec_padding_mask,
                                               oper)

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
def load_wiki_dataset(file_path, flag=1, shuffle=True):
    train = {}
    test = {}
    valid = {}
    if flag == 0:
        with open(f'{file_path}train.txt', 'rb') as f:
            train_labels = pickle.load(f)
        with open(f'{file_path}valid.txt', 'rb') as f:
            valid_labels = pickle.load(f)
        with open(f'{file_path}test.txt', 'rb') as f:
            test_labels = pickle.load(f)
        with open(f'{file_path}train_keyword.txt', 'rb') as f:
            train_data = pickle.load(f)
        with open(f'{file_path}valid_keyword.txt', 'rb') as f:
            valid_data = pickle.load(f)
        with open(f'{file_path}test_keyword.txt', 'rb') as f:
            test_data = pickle.load(f)
    elif flag == 1:
        with open(f'{file_path}train_one_sentence.txt', 'rb') as f:
            train_labels = pickle.load(f)
        with open(f'{file_path}valid_one_sentence.txt', 'rb') as f:
            valid_labels = pickle.load(f)
        with open(f'{file_path}test_one_sentence.txt', 'rb') as f:
            test_labels = pickle.load(f)
        with open(f'{file_path}train_one_sentence_keyword.txt', 'rb') as f:
            train_data = pickle.load(f)
        with open(f'{file_path}valid_one_sentence_keyword.txt', 'rb') as f:
            valid_data = pickle.load(f)
        with open(f'{file_path}test_one_sentence_keyword.txt', 'rb') as f:
            test_data = pickle.load(f)
    else:
        raise ValueError('wrong load data flag!!!')
    train['input'] = keyword_to_text(train_data)
    train['label'] = train_labels
    test['input'] = keyword_to_text(test_data)
    test['label'] = test_labels
    valid['input'] = keyword_to_text(valid_data)
    valid['label'] = valid_labels
    return train, test, valid


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


def wiki_dataset(file_path, flag=1, shuffule=True):
    _train, _test, _valid = load_wiki_dataset(file_path, flag, shuffule)
    train = dict_to_list(_train)
    test = dict_to_list(_test)
    valid = dict_to_list(_valid)
    return train, test, valid


def translate(sentence1, sentence2, oper='add'):
    result1, _ = evaluate(sentence1, tr_orig)
    result2, _ = evaluate(sentence2, tr_orig)
    result_latent, _ = evaluate_latent(sentence1, sentence2, tr_latnet, oper)
    print('Input1:      {}'.format(sentence1))
    print('Input2:      {}'.format(sentence2))

    predicted_sentence1 = tokenizer_en.decode([i for i in result1 if i < tokenizer_en.vocab_size])
    predicted_sentence2 = tokenizer_en.decode([i for i in result2 if i < tokenizer_en.vocab_size])
    predicted_sentence_latent = tokenizer_en.decode([i for i in result_latent if i < tokenizer_en.vocab_size])
    print('orig_1:      {}'.format(predicted_sentence1))
    print('orig_2:      {}'.format(predicted_sentence2))
    print('latent:      {}'.format(predicted_sentence_latent))
    return predicted_sentence1, predicted_sentence2, predicted_sentence_latent


def keyword_to_text_oneline(keywords, shuffle=False):
    if shuffle:
        random.shuffle(keywords)
    text = ''
    for keyword in keywords:
        text += f'{keyword} '
    return text


if __name__ == '__main__':
    print("===============  making sentence  ===============")
    file_path = './data/preprocessing/wiki103/'
    check_data = 'test'

    # load wiki dataset and make tensor
    train, test, valid = wiki_dataset(file_path)
    if check_data == 'train':
        total_data_len = len(train)
    elif check_data == 'test':
        total_data_len = len(test)
    elif check_data == 'valid':
        total_data_len = len(valid)
    '''
    train = tf.Variable(train, tf.string)
    test = tf.Variable(test, tf.string)
    valid = tf.Variable(valid, tf.string)

    if check_data == 'train':
        check_dataset = tf.data.Dataset.from_tensor_slices(train)
    elif check_data == 'test':
        check_dataset = tf.data.Dataset.from_tensor_slices(test)
    elif check_data == 'valid':
        check_dataset = tf.data.Dataset.from_tensor_slices(valid)
    print('wiki data load complete!')
    '''
    tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file('en_tfds_wmt8K.MASK')
    print("corpus loading finished!")
    '''
    BUFFER_SIZE = 20000
    BATCH_SIZE = 20
    # train_dataset <TensorSliceDataset shapes: (2,), types: tf.string>
    check_dataset = check_dataset.map(tf_encode)
    check_dataset = check_dataset.filter(filter_max_length)
    check_dataset = check_dataset.shuffle(total_data_len)

    # cache the dataset to memory to get a speedup while reading from it.
    print("dataset encoding finished")
    '''
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

    tr_orig = Tr_orig(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)

    tr_latnet = Tr_latent(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)

    ############################# pre-trained
    directory = "./ckpt/"
    checkpoint_path = directory + 'pretrained'
    ckpt_orig = tf.train.Checkpoint(transformer=tr_orig, optimizer=optimizer)
    ckpt_manager_pre = tf.train.CheckpointManager(ckpt_orig, checkpoint_path, max_to_keep=10)
    enc = tr_orig.encoder
    dec = tr_orig.decoder
    fn = tr_orig.final_layer

    encoder_ckpt_obj_pre = tf.train.Checkpoint(encoder=enc, optimizer=optimizer)
    decoder_ckpt_obj_pre = tf.train.Checkpoint(decoder=dec, optimizer=optimizer)
    fn_ckpt_obj_pre = tf.train.Checkpoint(final_layer=fn, optimizer=optimizer)

    en_enc_ckpt_path = checkpoint_path + '/en_encoder-1'
    en_dec_ckpt_path = checkpoint_path + '/en_decoder-1'
    en_fn_ckpt_path = checkpoint_path + '/en_fn-1'

    encoder_ckpt_obj_pre.restore(en_enc_ckpt_path).expect_partial()
    print('################# enc ckpt restored   ', en_enc_ckpt_path)
    decoder_ckpt_obj_pre.restore(en_dec_ckpt_path).expect_partial()
    print('################# dec ckpt restored   ', en_dec_ckpt_path)
    fn_ckpt_obj_pre.restore(en_fn_ckpt_path).expect_partial()
    print('################# fn ckpt restored    ', en_fn_ckpt_path)

    # edit directory and path
    directory = "./ckpt/"
    checkpoint_path = directory + 'pretrained'
    ckpt_latent = tf.train.Checkpoint(transformer=tr_latnet, optimizer=optimizer)
    ckpt_manager_dec = tf.train.CheckpointManager(ckpt_latent, checkpoint_path, max_to_keep=10)
    enc1 = tr_latnet.encoder1
    enc2 = tr_latnet.encoder2
    dec = tr_latnet.decoder
    fn = tr_latnet.final_layer

    encoder_ckpt_obj_1 = tf.train.Checkpoint(encoder=enc1, optimizer=optimizer)
    encoder_ckpt_obj_2 = tf.train.Checkpoint(encoder=enc2, optimizer=optimizer)
    decoder_ckpt_obj = tf.train.Checkpoint(decoder=dec, optimizer=optimizer)
    fn_ckpt_obj = tf.train.Checkpoint(final_layer=fn, optimizer=optimizer)

    en_enc_ckpt_path = checkpoint_path + '/en_encoder-1'
    en_dec_ckpt_path = checkpoint_path + '/en_decoder-1'
    en_fn_ckpt_path = checkpoint_path + '/en_fn-1'

    encoder_ckpt_obj_1.restore(en_enc_ckpt_path).expect_partial()
    encoder_ckpt_obj_2.restore(en_enc_ckpt_path).expect_partial()
    print('################# enc1,2 ckpt restored   ', en_enc_ckpt_path)
    decoder_ckpt_obj.restore(en_dec_ckpt_path).expect_partial()
    print('################# dec ckpt restored   ', en_dec_ckpt_path)
    fn_ckpt_obj.restore(en_fn_ckpt_path).expect_partial()
    print('################# fn ckpt restored    ', en_fn_ckpt_path)


    print('=' * 40)
    print("just before text generation!!")
    print('=' * 40)

    with open('./data/preprocessing/wiki103_stopword/test_one_sentence_keyword_stopword.txt', 'rb') as f:
        data = pickle.load(f)

    delimiter = '=' * 50 + '\n'

    with open('./data/latent_variation/orig_latent_variation_using_stop_keyword.txt', 'w', encoding='utf-8') as f:
        for idx in range(0, len(data), 2):
            if idx == len(data)-1:
                break
            a = len(data[idx])
            b = len(data[idx+1])
            c = min(a, b)
            text1 = keyword_to_text_oneline(data[idx][:c])
            text2 = keyword_to_text_oneline(data[idx+1][:c])
            result1, result2, result_latent = translate(text1, text2, oper='add')
            f.write('Input1:      {}\n'.format(text1))
            f.write('Input2:      {}\n'.format(text2))
            f.write('orig_1:      {}\n'.format(result1))
            f.write('orig_2:      {}\n'.format(result2))
            f.write('latent:      {}\n'.format(result_latent))
            f.write(delimiter)


