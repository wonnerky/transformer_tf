import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import random, pickle, time, json, copy
from models.tr_caption_non_linear import Tr_caption
from tqdm import tqdm
import os, argparse
import matplotlib.pyplot as plt
import math
from utils.mscoco_dataset import build_data_loader
from transformers import XLMTokenizer

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




train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# @tf.function(input_signature=train_step_signature)
# @tf.function
''' model train '''
def train(args):

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        # return tf.reduce_mean(loss_)
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp, True)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    def val_eval(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        predictions, _ = transformer(inp, tar_inp, False)
        loss = loss_function(tar_real, predictions)
        train_loss(loss)
        train_accuracy(tar_real, predictions)

    print("==== image caption training start ====")
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
    print("corpus loading finished!")

    # model parameter setting
    num_layers = args.num_layers
    d_model = args.emb_dim
    dff = d_model * 4
    num_heads = args.num_heads

    input_vocab_size = tokenizer.vocab_size
    target_vocab_size = tokenizer.vocab_size
    dropout_rate = args.dropout_rate
    print("hyperparameters confirmed")

    EPOCHS = args.epoch
    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    # learning_rate = CustomSchedule()   "custom learning rate"
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, epsilon=args.adam_epsilon)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    transformer = Tr_caption(num_layers, d_model, num_heads, dff, target_vocab_size,
                             pe_target=target_vocab_size, rate=dropout_rate)
    print('Model create complete!!!')

    # edit directory and path
    directory = args.ckpt_path
    model_id = args.model_id
    checkpoint_path = os.path.join(directory, model_id)
    log_path = os.path.join(checkpoint_path, 'log')
    log_file_name = os.path.join(log_path, f'{model_id}.pkl')
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1000)
    print(f'==== ckpt path : {checkpoint_path} ====')

    init = not args.isContinue
    pre_train = not args.from_scratch

    # model init or continue training
    print('=' * 50)
    if init:
        if ckpt_manager.latest_checkpoint:
            print('There is ckpt files!!!')
            exit()
        log_ = {'epoch': [], 'batch': [], 'loss': [], 'accuracy': [], 'validation_loss': [], 'validation_accu': []}
        if pre_train:
            pre_dir = args.pt_dec_path
            decoder_ckpt_obj = tf.train.Checkpoint(decoder=transformer.transformer.decoder, optimizer=optimizer)
            fn_ckpt_obj = tf.train.Checkpoint(final_layer=transformer.transformer.final_layer, optimizer=optimizer)
            dec_ckpt_path = f'{pre_dir}decoder-1'
            fn_ckpt_path = f'{pre_dir}fn-1'
            print(dec_ckpt_path)
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
    BATCH_SIZE = args.dataset_batchSize
    train_dataset, train_len = build_data_loader(args, tokenizer, type='train')
    valid_dataset, valid_len = build_data_loader(args, tokenizer, type='valid')

    train_start_time = time.time()
    loss_plot = []
    accu_plot = []
    val_loss_plot = []
    val_accu_plot = []

    _valid_loss = None
    __valid_loss = None
    ___valid_loss = None
    early_stop_flag = False

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
                                                                                  int(train_len / BATCH_SIZE),
                                                                                  train_loss.result(),
                                                                                  train_accuracy.result()))
        log_['epoch']= epoch + 1
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
        total_val_loss = 0
        cnt_val = 0
        for (val_batch, (val_inp, val_tar)) in enumerate(valid_dataset):
            # inp shape : (batch, object, 2048)
            val_inp = tf.reshape(val_inp, (val_inp.shape[0], -1, 2048))
            val_eval(val_inp, val_tar)
            cnt_val += 1
            total_val_loss += train_loss.result().numpy()
        val_loss_plot.append(train_loss.result().numpy())
        val_accu_plot.append(train_accuracy.result().numpy())
        log_['validation_loss'].append(train_loss.result().numpy())
        log_['validation_accu'].append(train_accuracy.result().numpy())
        with open(log_file_name, 'wb') as f:
            pickle.dump(log_, f)
            # print('saving log file...')
        print('Validation Loss {:.12f} Accuracy {:.12f}\n\n'.format(train_loss.result(), train_accuracy.result()))

        val_loss = total_val_loss / cnt_val
        # early stopping check
        if _valid_loss is None:
            _valid_loss = val_loss
        elif __valid_loss is None:
            __valid_loss = _valid_loss
            _valid_loss = val_loss
        elif ___valid_loss is None:
            ___valid_loss = __valid_loss
            __valid_loss = _valid_loss
            _valid_loss = val_loss
        else:
            cal_early = val_loss - ___valid_loss
            if cal_early > 0:
                early_stop_flag = True
            ___valid_loss = __valid_loss
            __valid_loss = _valid_loss
            _valid_loss = val_loss

        # Stop if early stopping
        if early_stop_flag:
            train_time = time.time() - train_start_time
            with open(os.path.join(log_path, 'train_time.txt'), 'w') as f:
                f.write("train time : {}".format(train_time))
            break

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



if __name__ == '__main__':

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        # dataset configuration
        parser.add_argument("--file_path", default="data/MSCOCO/k_split/", type=str, required=False,
                            help="mscoco data path")
        parser.add_argument('--dataset_bufferSize', type=int, default=1000, required=False,
                            help="dataset buffer size")
        parser.add_argument('--dataset_batchSize', type=int, default=20, required=False,
                            help="dataset batch size")

        # model configuration
        parser.add_argument("--positional_encoding", default=False, type=str2bool, required=False,
                            help="create or not encoder positional encoding")
        parser.add_argument("--num_layers", default=6, type=int, required=False,
                            help="number of transformer layers")
        parser.add_argument("--num_heads", default=8, type=int, required=False,
                            help="number of transformer layer heads")
        parser.add_argument("--emb_dim", default=512, type=int, required=False,
                            help="embedding dimension")
        # train configuration
        parser.add_argument("--epoch", default=30, type=int, required=False,
                            help="epoch")
        parser.add_argument('--seed', type=int, default=42, required=False,
                            help="random seed for initialization")
        parser.add_argument('--weight_decay', type=float, default=0, required=False,
                            help="weight decay")
        parser.add_argument('--dropout_rate', type=float, default=0.1, required=False,
                            help="dropout rate")
        parser.add_argument('--learning_rate', type=float, default=9e-5, required=False,
                            help="learning rate")
        parser.add_argument('--adam_epsilon', type=float, default=1e-8, required=False,
                            help="adam epsilon")
        parser.add_argument('--warmup_steps', type=float, default=0, required=False,
                            help="warmup steps")
        parser.add_argument('--isContinue', type=str2bool, default=False, required=False,
                            help="train from ckpt")
        parser.add_argument('--from_scratch', type=str2bool, default=False, required=False,
                            help="train decoder from scratch or not")
        parser.add_argument("--ckpt_path", default="ckpt/", type=str, required=True,
                            help="save ckpt path")
        parser.add_argument("--model_id", default="", type=str, required=True,
                            help="model id")
        parser.add_argument("--pt_dec_path", default="", type=str, required=True,
                            help="pre trained decoder path")


        args = parser.parse_args()
        train(args)