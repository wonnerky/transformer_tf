import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import random, pickle, time, json, copy
from model_original_transformer import Transformer as Tr_orig
from model_keyword_transformer import Transformer as Tr_keyword
import os

# tf.enable_eager_execution()

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


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


def evaluate(inp_sentence):
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
        predictions, attention_weights = transformer(encoder_input,
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


def load_wiki_dataset(file_path, flag=1, shuffle=True, isStopword=False, isMask=False, maskRate=0.0, pe=False):
    train = {}
    test = {}
    valid = {}
    if isStopword:
        if flag == 0:
            with open(f'{file_path}train.txt', 'rb') as f:
                train_labels = pickle.load(f)
            with open(f'{file_path}valid.txt', 'rb') as f:
                valid_labels = pickle.load(f)
            with open(f'{file_path}test.txt', 'rb') as f:
                test_labels = pickle.load(f)
            with open(f'{file_path}train_keyword_stopword.txt', 'rb') as f:
                train_data = pickle.load(f)
            with open(f'{file_path}valid_keyword_stopword.txt', 'rb') as f:
                valid_data = pickle.load(f)
            with open(f'{file_path}test_keyword_stopword.txt', 'rb') as f:
                test_data = pickle.load(f)
        elif flag == 1:
            with open(f'{file_path}train_one_sentence.txt', 'rb') as f:
                train_labels = pickle.load(f)
            with open(f'{file_path}valid_one_sentence.txt', 'rb') as f:
                valid_labels = pickle.load(f)
            with open(f'{file_path}test_one_sentence.txt', 'rb') as f:
                test_labels = pickle.load(f)
            with open(f'{file_path}train_one_sentence_keyword_stopword.txt', 'rb') as f:
                train_data = pickle.load(f)
            with open(f'{file_path}valid_one_sentence_keyword_stopword.txt', 'rb') as f:
                valid_data = pickle.load(f)
            with open(f'{file_path}test_one_sentence_keyword_stopword.txt', 'rb') as f:
                test_data = pickle.load(f)
        else:
            raise ValueError('wrong load data flag!!!')
    else:
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
    if isMask:
        if pe:
            train_data = masking_keyword_pe(train_data, maskRate)
            test_data = masking_keyword_pe(test_data, maskRate)
            valid_data = masking_keyword_pe(valid_data, maskRate)
            shuffle = False
        else:
            train_data = masking_keyword_wo_pe(train_data, maskRate)
            test_data = masking_keyword_wo_pe(test_data, maskRate)
            valid_data = masking_keyword_wo_pe(valid_data, maskRate)
            # shuffle = False
    train['input'] = keyword_to_text(train_data, shuffle)
    train['label'] = train_labels
    test['input'] = keyword_to_text(test_data, shuffle)
    test['label'] = test_labels
    valid['input'] = keyword_to_text(valid_data, shuffle)
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


def wiki_dataset(file_path, flag=1, shuffle=True, isStopword=False, isMask=False, maskRate=0.0, pe=False):
    print(f'dataset mode : flag({flag}), shuffle({shuffle}), isStopword({isStopword}), isMask({isMask}), maskRate({maskRate}), pe({pe})')
    _train, _test, _valid = load_wiki_dataset(file_path, flag, shuffle, isStopword, isMask, maskRate, pe)
    train = dict_to_list(_train)
    test = dict_to_list(_test)
    valid = dict_to_list(_valid)
    return train, test, valid


if __name__ == '__main__':
    print("==== text gen training start ====")
    ### file path, stopword or not
    isStopword = True
    if isStopword:
        file_path = './data/preprocessing/wiki103_stopword/'
    else:
        file_path = './data/preprocessing/wiki103/'
    ###
    isMask = True
    maskRate = 0.3
    enc_init = True
    dec_init = False
    shuffle = True
    pe = True

    tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file('en_tfds_wmt8K.MASK')
    print("corpus loading finished!")

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8

    input_vocab_size = tokenizer_en.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2
    dropout_rate = 0.1
    print("hyperparameters confirmed")

    EPOCHS = 7
    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    if pe:
        transformer = Tr_orig(num_layers, d_model, num_heads, dff,
                                input_vocab_size, target_vocab_size,
                                pe_input=input_vocab_size,
                                pe_target=target_vocab_size,
                                rate=dropout_rate)
        print('Tr_orig create complete!!')
                        
    else:    
        transformer = Tr_keyword(num_layers, d_model, num_heads, dff,
                                input_vocab_size, target_vocab_size,
                                pe_target=target_vocab_size,
                                rate=dropout_rate)
        print('Tr_keyword create complete!!')

    # edit directory and path
    directory = "./ckpt/"
    checkpoint_path = directory + '0810_stop_mask_pe_shuffle_30_pre'
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=100)
    print(f'==== ckpt path : {checkpoint_path} ====')

    if enc_init:
        if dec_init:
            print('initialize all model weight')
        else:
            decoder_ckpt_obj = tf.train.Checkpoint(decoder=transformer.decoder, optimizer=optimizer)
            fn_ckpt_obj = tf.train.Checkpoint(final_layer=transformer.final_layer, optimizer=optimizer)
            dec_ckpt_path = f'{directory}pretrained/en_decoder-1'
            fn_ckpt_path = f'{directory}pretrained/en_fn-1'
            decoder_ckpt_obj.restore(dec_ckpt_path)
            fn_ckpt_obj.restore(fn_ckpt_path)
            print('decoder & final layer ckpt restored!!')
    else:
        if dec_init:
            print(f'error occur!!!\nenc_init : {enc_init}, dec_init : {dec_init}')
            exit()
        else:
            if ckpt_manager.latest_checkpoint:
                ckpt.restore(ckpt_manager.latest_checkpoint)
                print('         !!!!! Latest checkpoint restored!!!!!')
                print('              ', ckpt_manager.latest_checkpoint)
            else:
                print('         !!!!! NO checkpoint restored!!!!!')

    print('=' * 40)
    print("just before training loop")
    print('=' * 40)
    print(f'checkpoint path : {checkpoint_path}')
    step = 0
    ckpt_num = 0
    log_ = {'epoch': [], 'batch': [], 'loss': [], 'accuracy': []}

    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()

        # shuffle keyword
        start_time = time.time()
        # load wiki dataset and make tensor
        train, _, _ = wiki_dataset(file_path, shuffle=shuffle, isStopword=isStopword, isMask=isMask, maskRate=maskRate, pe=pe)
        total_data_len = len(train)
        train = tf.Variable(train, tf.string)
        train_dataset = tf.data.Dataset.from_tensor_slices(train)
        BUFFER_SIZE = 20000
        BATCH_SIZE = 60
        # train_dataset <TensorSliceDataset shapes: (2,), types: tf.string>
        train_dataset = train_dataset.map(tf_encode)
        train_dataset = train_dataset.filter(filter_max_length)
        # cache the dataset to memory to get a speedup while reading from it.
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        print('keyword shuffle complete: {:.2f}s'.format(time.time() - start_time))

        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            step = step + 1

            if batch % 300 == 0:
                print('Epoch {} Batch {}/{} Loss {:.12f} Accuracy {:.12f}'.format(epoch + 1, batch,
                                                                                  int(total_data_len / BATCH_SIZE),
                                                                                  train_loss.result(),
                                                                                  train_accuracy.result()))
                log_['epoch'].append(epoch + 1)
                log_['batch'].append(batch)
                log_['loss'].append(train_loss.result())
                log_['accuracy'].append(train_accuracy.result())

            if batch % 6000 == 0 and batch != 0:
                ckpt_num = ckpt_num + 1
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} batch {}/{} at {}'.format(epoch + 1, batch,
                                                                                int(total_data_len / BATCH_SIZE),
                                                                                ckpt_save_path))

        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        print("current step : ", step)
        print('Epoch {} ENDED --- Loss {:.12f} Accuracy {:.12f}\n\n'.format(epoch + 1, train_loss.result(),
                                                                            train_accuracy.result()))

    with open('./log/0810_stop_mask_pe_shuffle_30_pre_log.json.txt', 'wb') as f:
        pickle.dump(log_, f)