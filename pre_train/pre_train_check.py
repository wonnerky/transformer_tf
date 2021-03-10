import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import random, pickle, time, json, copy
from model_original_transformer import Transformer as Tr_orig
from model_keyword_transformer import Transformer as Tr_keyword
import os
from tqdm import tqdm
# from nltk.translate.meteor_score import meteor_score, single_meteor_score
# from nltk.translate.bleu_score import sentence_bleu
# from rouge_score import rouge_scorer

# tf.enable_eager_execution()

# 소스: Keyword
# 타겟: Text
MAX_LENGTH = 256

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model=1024, warmup_steps=10000):
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


def evaluate(encoder_input, batch_size=10):
    decoder_input = [tokenizer_en.vocab_size] * batch_size
    output = tf.expand_dims(decoder_input, 1)
    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        output = tf.concat([output, predicted_id], axis=-1)
    return output


def keywordList_to_txt(keys_lists, shuffle=False, isMask=False, maskRate=0.0):
    rng = random.Random()
    print('keyword list to text converting....')
    if isMask:
        keywords = copy.deepcopy(keys_lists)
        result = []
        print('keyword masking processing...')
        for keywords_ in tqdm(keywords):
            rate = rng.uniform(0.0, maskRate + 0.01)
            if rate >= maskRate:
                rate = maskRate
            cnt = int(len(keywords_) * rate)
            del_list_ = rng.sample(keywords_, cnt)
            for ele in del_list_:
                del keywords_[keywords_.index(ele)]
            result.append(keywords_)
        del keywords
        del keys_lists
        keys_lists = result
        del result
    if shuffle:
        print('keyword shuffling processing...')
        for keys in tqdm(keys_lists):
            random.shuffle(keys)
    output = []
    print('keyword to text processing...')
    for keys in tqdm(keys_lists):
        text = ''
        for i in keys:
            text += i + ' '
        output.append(text[:-1])
    return output


def wmt_test_dataset(file_path, shuffle=False, isMask=False, maskRate=0.0):
    print(f'test dataset mode : shuffle({shuffle}), isMask({isMask}), maskRate({maskRate})')
    output_test = []
    output_valid = []
    temp = []
    print('test dataset processing...')
    with open(f'{file_path}test_keyword_stopword.txt', 'rb') as f:
        test_keyword = pickle.load(f)
    test_keyword = keywordList_to_txt(test_keyword, shuffle=shuffle, isMask=isMask, maskRate=maskRate)
    with open(f'{file_path}test.txt', 'rb') as f:
        test_label = pickle.load(f)
    print('valid dataset processing...')
    with open(f'{file_path}valid_keyword_stopword.txt', 'rb') as f:
        valid_keyword = pickle.load(f)
    valid_keyword = keywordList_to_txt(valid_keyword, shuffle=shuffle, isMask=isMask, maskRate=maskRate)
    with open(f'{file_path}valid.txt', 'rb') as f:
        valid_label = pickle.load(f)
    for i in range(len(test_keyword)):
        temp.append(test_keyword[i])
        temp.append(test_label[i])
        output_test.append(temp)
        temp = []
    for i in range(len(valid_keyword)):
        temp.append(valid_keyword[i])
        temp.append(valid_label[i])
        output_valid.append(temp)
        temp = []
    return output_test, output_valid


def wmt_train_dataset(file_path, shuffle=False, isMask=False, maskRate=0.0):
    print(f'train dataset mode : shuffle({shuffle}), isMask({isMask}), maskRate({maskRate})')
    labels = []
    keywords = []
    output = []
    temp = []
    for i in range(2):
        with open(f'{file_path}train_shard_4_keyword_stopword_{i}.txt', 'rb') as f:
            keyword_ = pickle.load(f)
        print(f'{i}th keyword loading complete!!')
        with open(f'{file_path}train_shard_4_{i}.txt', 'rb') as f:
            label_ = pickle.load(f)
        print(f'{i}th label loading complete!!')
        labels = labels + label_
        keywords = keywords + keywordList_to_txt(keyword_, shuffle=shuffle, isMask=isMask, maskRate=maskRate)
        del label_
        del keyword_
        print(f'{i}th processing complete!!')
    for i in range(len(keywords)):
        temp.append(keywords[i])
        temp.append(labels[i])
        output.append(temp)
        temp = []
    return output



if __name__ == '__main__':
    program_start_time = time.time()
    print("==== check pre-trained effect ====")
    file_path = './data/preprocessing/wmt/'
    isMask = True
    maskRate = 0.3
    pre_trained = True
    shuffle = False
    pe = False
    from_scratch = True

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

    EPOCHS = 3
    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    learning_rate = CustomSchedule(d_model=d_model)
    optimizer = tf.keras.optimizers.Adam(0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-6)
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
    directory = "./ckpt/pre_trained_check/0923_pre_wo_shu_30_2/"
    epoch_checkpoint_path = directory + 'epoch_ckpt/'
    early_checkpoint_path = directory + 'early_ckpt/'
    log_path = directory + 'log/'
    log_file_name = f'{log_path}/0923_pre_wo_shu_30_2_log.json.txt'
    early_log_file_name = f'{log_path}/0923_pre_wo_shu_30_2_log_early.json.txt'
    if not os.path.isdir(epoch_checkpoint_path):
        os.makedirs(epoch_checkpoint_path)
    if not os.path.isdir(early_checkpoint_path):
        os.makedirs(early_checkpoint_path)
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    epoch_ckpt_manager = tf.train.CheckpointManager(ckpt, epoch_checkpoint_path, max_to_keep=10000)
    early_ckpt_manager = tf.train.CheckpointManager(ckpt, early_checkpoint_path, max_to_keep=10000)
    print(f'epoch ckpt path : {epoch_checkpoint_path}')
    print(f'early ckpt path : {early_checkpoint_path}')

    if from_scratch:
        if early_ckpt_manager.latest_checkpoint:
            print('!!!! There exist ckpt files !!!!')
            exit()
        if pre_trained:
            decoder_ckpt_obj = tf.train.Checkpoint(decoder=transformer.decoder, optimizer=optimizer)
            fn_ckpt_obj = tf.train.Checkpoint(final_layer=transformer.final_layer, optimizer=optimizer)
            dec_ckpt_path = 'ckpt/pretrained/en_decoder-1'
            fn_ckpt_path = 'ckpt/pretrained/en_fn-1'
            decoder_ckpt_obj.restore(dec_ckpt_path)
            fn_ckpt_obj.restore(fn_ckpt_path)
            log_ = {'epoch': [], 'batch': [], 'loss': [], 'accuracy': []}
            early_log_ = {'epoch': [], 'batch': [], 'loss': [], 'accuracy': []}
            print('=' * 50)
            print('decoder & final layer ckpt restored!!')
            print('=' * 50)
        else:
            log_ = {'epoch': [], 'batch': [], 'loss': [], 'accuracy': []}
            early_log_ = {'epoch': [], 'batch': [], 'loss': [], 'accuracy': []}
            print('=' * 50)
            print('initialize all model weight')
            print('=' * 50)
    else:
        if not epoch_ckpt_manager.latest_checkpoint and not early_ckpt_manager.latest_checkpoint:
            print('!!!! There not exist ckpt files !!!!')
            exit()
        if epoch_ckpt_manager.latest_checkpoint:
            ckpt.restore(epoch_ckpt_manager.latest_checkpoint)
            print('=' * 50)
            print('Latest checkpoint restored!!!!!')
            print('latest file name (epoch): ', epoch_ckpt_manager.latest_checkpoint)
        else:
            ckpt.restore(early_ckpt_manager.latest_checkpoint)
            print('=' * 50)
            print('Latest checkpoint restored!!!!!')
            print('latest file name (early): ', early_ckpt_manager.latest_checkpoint)
        if os.path.isdir(log_path):
            with open(log_file_name, 'rb') as f:
                log_ = pickle.load(f)
            with open(early_log_file_name, 'rb') as f:
                early_log_ = pickle.load(f)
            print('log data load complete!!')
        else:
            print('No log files!!!')
            exit()
        print('=' * 50)

    print('=' * 40)
    print("just before training loop")
    print('=' * 40)
    step = 0
    ckpt_num = 0

    # loading test, valid dataset
    BUFFER_SIZE = 20000
    BATCH_SIZE = 60
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()

        # shuffle keyword
        start_time = time.time()
        # load wiki dataset and make tensor
        train_dataset = wmt_train_dataset(file_path, shuffle=shuffle, isMask=isMask, maskRate=maskRate)
        total_data_len = len(train_dataset)
        train_dataset = tf.Variable(train_dataset, tf.string)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)

        # train_dataset <TensorSliceDataset shapes: (2,), types: tf.string>
        train_dataset = train_dataset.map(tf_encode)
        train_dataset = train_dataset.filter(filter_max_length)
        # cache the dataset to memory to get a speedup while reading from it.
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        print('keyword shuffle complete: {:.2f}s'.format(time.time() - start_time))
        div_num = 150
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            step = step + 1

            if epoch == 0 and batch % div_num == 0:
                print('Epoch {} Batch {}/{} Loss {:.12f} Accuracy {:.12f}'.format(epoch + 1, batch,
                                                                                  int(total_data_len / BATCH_SIZE),
                                                                                  train_loss.result(),
                                                                                  train_accuracy.result()))
                ckpt_save_path = early_ckpt_manager.save()
                print('Saving early checkpoint for epoch {} batch {}/{} at {}'.format(epoch + 1, batch,
                                                                                int(total_data_len / BATCH_SIZE),
                                                                                ckpt_save_path))
                # early log에 저장
                early_log_['epoch'].append(epoch + 1)
                early_log_['batch'].append(batch)
                early_log_['loss'].append(train_loss.result())
                early_log_['accuracy'].append(train_accuracy.result())

                div_num *= 2

            if batch % 300 == 0 and batch != 0:
                print('Epoch {} Batch {}/{} Loss {:.12f} Accuracy {:.12f}'.format(epoch + 1, batch,
                                                                                  int(total_data_len / BATCH_SIZE),
                                                                                  train_loss.result(),
                                                                                  train_accuracy.result()))

                # # log에 저장
                log_['epoch'].append(epoch + 1)
                log_['batch'].append(batch)
                log_['loss'].append(train_loss.result())
                log_['accuracy'].append(train_accuracy.result())

        ckpt_save_path = epoch_ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        print("current step : ", step)
        print('Epoch {} ENDED --- Loss {:.12f} Accuracy {:.12f}\n\n'.format(epoch + 1, train_loss.result(),
                                                                            train_accuracy.result()))
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        with open(log_file_name, 'wb') as f:
            pickle.dump(log_, f)
        with open(early_log_file_name, 'wb') as f:
            pickle.dump(early_log_, f)

    print('program processing time : {:.2f}s'.format(time.time() - program_start_time))