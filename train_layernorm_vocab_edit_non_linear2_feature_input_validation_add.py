import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import random, pickle, time, json, copy
from model.layernorm_transformer_non_linear2_feature_input_model import Tr_caption
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import math
# from transformers import XLMTokenizer

# tf.enable_eager_execution()

# 소스: conv Feature 14*14*512 -> 196*512
# 타겟: annotation text

MASK_PROB = 0.15
MAX_PRED_PER_SEQ = 20
rng = random.Random(12345)
MAX_LENGTH = 180


def cal_lr(step, warmup_steps=10000, d_model=1024):
    arg1 = step ** -0.5
    arg2 = step * (warmup_steps ** -1.5)
    return (d_model ** -.05) * min(arg1, arg2)


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
    input_tokens = tokenizer_decoder.encode(keyword.numpy())
    label_tokens = tokenizer_decoder.encode(text.numpy())
    source = [tokenizer_decoder.vocab_size] + input_tokens + [tokenizer_decoder.vocab_size + 1]
    target = [tokenizer_decoder.vocab_size] + label_tokens + [tokenizer_decoder.vocab_size + 1]
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


# @tf.function(input_signature=train_step_signature)
# @tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(tf.reduce_sum(inp, 2), tar_inp)
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


def val_eval(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(tf.reduce_sum(inp, 2), tar_inp)
    predictions, _ = transformer(inp, tar_inp,
                                 False,
                                 enc_padding_mask,
                                 combined_mask,
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)
    train_loss(loss)
    train_accuracy(tar_real, predictions)


def map_func(img_name, cap):
    file_path = 'dataset/MSCOCO/preprocessing/feature/' + img_name.decode('utf-8') + '.npy'
    img_tensor = np.load(file_path)
    img_tensor = img_tensor.reshape((-1))
    cap_ = cap.decode('utf-8')
    label = [tokenizer_decoder.vocab_size] + tokenizer_decoder.encode(cap_) + [tokenizer_decoder.vocab_size + 1]
    return img_tensor, label


if __name__ == '__main__':
    print("==== image caption training start ====")
    tokenizer_decoder = tfds.features.text.SubwordTextEncoder.load_from_file('utils/tokenizer/en_tfds_wmt8K.MASK')
    target_vocab_size = tokenizer_decoder.vocab_size + 2
    print("corpus loading finished!")

    # model parameter setting
    num_layers = 4
    d_model = 512
    dff = d_model * 4
    num_heads = 8
    dropout_rate = 0.1
    print("hyperparameters confirmed")

    EPOCHS = 20
    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    learning_rate = CustomSchedule()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00009, beta_1=0.9, beta_2=0.999, epsilon=1e-6)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    transformer = Tr_caption(num_layers, d_model, num_heads, dff, target_vocab_size,
                              pe_target=target_vocab_size, rate=dropout_rate)
    print('Model create complete!!!')

    # edit directory and path
    file_name = '1211_layernorm_non_linear2_feature_input_pre_1'
    directory = f"./ckpt/early_phase/feature_input/{file_name}/"
    checkpoint_path = directory + 'ckpt/'
    log_path = directory + 'log/'
    log_file_name = f'{log_path}/{file_name}.json.txt'
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1000)
    print(f'==== ckpt path : {checkpoint_path} ====')

    init = True
    pre_train = True

    # model init or continue training
    print('=' * 50)
    if init:
        if ckpt_manager.latest_checkpoint:
            print('There is ckpt files!!!')
            exit()
        log_ = {'epoch': [], 'batch': [], 'loss': [], 'accuracy': [], 'validation_loss': [], 'validation_accu': []}
        if pre_train:
            pre_dir = 'ckpt/pre_trained_weight/vocab_edit_512d'
            decoder_ckpt_obj = tf.train.Checkpoint(decoder=transformer.transformer.decoder, optimizer=optimizer)
            fn_ckpt_obj = tf.train.Checkpoint(final_layer=transformer.transformer.final_layer, optimizer=optimizer)
            dec_ckpt_path = f'{pre_dir}/decoder-1'
            fn_ckpt_path = f'{pre_dir}/fn-1'
            decoder_ckpt_obj.restore(dec_ckpt_path).expect_partial()
            fn_ckpt_obj.restore(fn_ckpt_path).expect_partial()
            print('Pre-trained weight restore! : Decoder, Fn')
            print('Training model from scratch')
        else:
            print('All weight initialized!!')
    else:
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print('Latest ckpt restored!!')
            print(ckpt_manager.latest_checkpoint)
            if os.path.isfile(log_file_name):
                with open(log_file_name, 'rb') as f:
                    log_ = pickle.load(f)
                print('Log file load complete!!')
            else:
                print('!!!! No log file exist !!!!')
                exit()
        else:
            print('!!!!! NO checkpoint restored!!!!!')
            exit()

    print('=' * 50 + '\n')
    print('=' * 40)
    print("just before training loop")
    print('=' * 40)
    ckpt_num = 0
    # Prepare Dataset
    BUFFER_SIZE = 1000
    BATCH_SIZE = 20

    with open('dataset/MSCOCO/k_split/train.json', 'r') as f:
        train_json = json.load(f)
    # with open('dataset/MSCOCO/k_split/restval.json', 'r') as f:
    #     restval_ = json.load(f)
    # train_json = train_ + restval_
    # del train_
    # del restval_
    img_name_train = []
    cap_train = []
    for ele in train_json:
        img_name_train.append(ele['filename'])
        cap_train.append(ele['sentence'])

    with open('dataset/MSCOCO/k_split/valid.json', 'r') as f:
        val_json = json.load(f)
    img_name_val = []
    cap_val = []
    for ele in val_json:
        img_name_val.append(ele['filename'])
        cap_val.append(ele['sentence'])

    num_steps = len(img_name_train) // BATCH_SIZE
    total_data_len = len(img_name_train)

    train_dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
    train_dataset = train_dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int64]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
    val_dataset = val_dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int64]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))

    train_start_time = time.time()
    loss_plot = []
    accu_plot = []
    val_loss_plot = []
    val_accu_plot = []

    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        total_loss = 0
        total_accu = 0
        for (batch, (inp, tar)) in enumerate(train_dataset):
            # inp shape : (batch, object, 2048)
            inp = tf.reshape(inp, (inp.shape[0], -1, 2048))
            train_step(inp, tar)
            # total_loss += train_loss.result()
            # total_accu += train_accuracy.result()
            if batch % 200 == 0 and batch != 0:
                log_['epoch'].append(epoch + 1)
                log_['batch'].append(batch)
                log_['loss'].append(train_loss.result().numpy())
                log_['accuracy'].append(train_accuracy.result().numpy())
                if not os.path.isdir(log_path):
                    os.makedirs(log_path)
                with open(log_file_name, 'wb') as f:
                    pickle.dump(log_, f)
                    # print('saving log file...')
                loss_plot.append(train_loss.result())
                accu_plot.append(train_accuracy.result())
            if batch % 500 == 0 and batch != 0:
                print('Epoch {} Batch {}/{} Loss {:.12f} Accuracy {:.12f}'.format(epoch + 1, batch,
                                                                                  int(total_data_len / BATCH_SIZE),
                                                                                  train_loss.result(),
                                                                                  train_accuracy.result()))
        log_['epoch'].append(epoch + 1)
        log_['batch'].append(batch)
        log_['loss'].append(train_loss.result().numpy())
        log_['accuracy'].append(train_accuracy.result().numpy())
        with open(log_file_name, 'wb') as f:
            pickle.dump(log_, f)
            # print('saving log file...')
        loss_plot.append(train_loss.result())
        accu_plot.append(train_accuracy.result())

        # save ckpt when end of epoch
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        print('Epoch {} ENDED --- Loss {:.12f} Accuracy {:.12f}\n\n'.format(epoch + 1, train_loss.result(),
                                                                            train_accuracy.result()))
        train_loss.reset_states()
        train_accuracy.reset_states()
        for (val_batch, (val_inp, val_tar)) in enumerate(val_dataset):
            # inp shape : (batch, object, 2048)
            val_inp = tf.reshape(val_inp, (val_inp.shape[0], -1, 2048))
            val_eval(val_inp, val_tar)
        val_loss_plot.append(train_loss.result().numpy())
        val_accu_plot.append(train_accuracy.result().numpy())
        log_['validation_loss'].append(train_loss.result().numpy())
        log_['validation_accu'].append(train_accuracy.result().numpy())
        with open(log_file_name, 'wb') as f:
            pickle.dump(log_, f)
            # print('saving log file...')
        print('Validation Loss {:.12f} Accuracy {:.12f}\n\n'.format(train_loss.result(), train_accuracy.result()))

    print("Training time spend : {:.2f}s".format(time.time() - train_start_time))
    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()
    plt.plot(accu_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Plot')
    plt.show()
    plt.plot(val_loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Val Loss Plot')
    plt.show()
    plt.plot(val_accu_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Val Accuracy Plot')
    plt.show()
