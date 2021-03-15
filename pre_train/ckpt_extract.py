import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from model.model_keyword_transformer import Transformer as Tr_keyword
from model.model_original_transformer import Transformer as Tr_orig
from utils.wiki_dataset import build_data_loader
from transformers import XLMTokenizer

import pickle, random, os, argparse


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


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_ckpt(args):

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        # return tf.reduce_mean(loss_)
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    # @tf.function(input_signature=train_step_signature)
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

    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
    num_layers = args.num_layers
    d_model = args.emb_dim
    dff = d_model * 4
    num_heads = args.num_heads
    input_vocab_size = tokenizer.vocab_size
    target_vocab_size = tokenizer.vocab_size
    dropout_rate = args.dropout_rate

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00009, beta_1=0.9, beta_2=0.999, epsilon=1e-6)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    transformer = Tr_keyword(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)

    ckpt_dir = args.extract_ckpt_path
    if not os.path.isdir(ckpt_dir):
        print('No ckpt file!!!')
        exit()
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=10)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('!!!!! Latest checkpoint restored!!!!!')
        print(ckpt_manager.latest_checkpoint)
    else:
        print("Can't load pretrained model!!!")
        exit()

    valid_dataset, _ = build_data_loader(args, tokenizer, type='valid')

    for (batch, (inp, tar)) in enumerate(valid_dataset):
        train_step(inp, tar)
        break

    enc = transformer.encoder
    dec = transformer.decoder
    fn = transformer.final_layer
    encoder_ckpt_obj = tf.train.Checkpoint(encoder=enc, optimizer=optimizer)
    decoder_ckpt_obj = tf.train.Checkpoint(decoder=dec, optimizer=optimizer)
    fn_ckpt_obj = tf.train.Checkpoint(final_layer=fn, optimizer=optimizer)

    save_path = args.save_ckpt_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    enc_ckpt_path = save_path + 'encoder'
    dec_ckpt_path = save_path + 'decoder'
    fn_ckpt_path = save_path + 'fn'

    encoder_path = encoder_ckpt_obj.save(enc_ckpt_path)
    print(f'encoder weight save success: {encoder_path}')
    decoder_path = decoder_ckpt_obj.save(dec_ckpt_path)
    print(f'decoder weight save success: {decoder_path}')
    fn_path = fn_ckpt_obj.save(fn_ckpt_path)
    print(f'fn weight save success: {fn_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--wiki_shuffle', type=str2bool, default=True, required=False,
                        help="determine inp shuffle")
    parser.add_argument('--wiki_isStopword', type=str2bool, default=True, required=False,
                        help="determine inp shuffle")
    parser.add_argument('--wiki_isMask', type=str2bool, default=True, required=False,
                        help="determine inp masking")
    parser.add_argument("--wiki_isMax", default=False, type=str2bool, required=False,
                        help="keyword max or fix select")
    parser.add_argument('--wiki_maskRate', type=float, default=0.3, required=False,
                        help="inp mask rate")
    parser.add_argument('--enc_word_max_length', type=int, default=20, required=False,
                        help="encoder input max length")
    parser.add_argument('--dec_word_max_length', type=int, default=39, required=False,
                        help="decoder input max length")
    parser.add_argument('--dec_max_length', type=int, default=100, required=False,
                        help="decoder input max length")
    parser.add_argument('--enc_max_first', default=True, type=str2bool, required=False,
                        help="determine encoder length limit")
    parser.add_argument("--file_path", default="./data/preprocessing/wiki/", type=str, required=False,
                        help="wiki data path")
    parser.add_argument('--wiki_bufferSize', type=int, default=20000, required=False,
                        help="dataset buffer size")
    parser.add_argument('--wiki_batchSize', type=int, default=5, required=False,
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
    parser.add_argument("--config", default="config.json", type=str, required=False,
                        help="config file")
    parser.add_argument("--save_best", default="save_best.pth", type=str, required=False,
                        help="save best file name")
    parser.add_argument("--epoch", default=5, type=int, required=False,
                        help="epoch")
    parser.add_argument("--batch", default=8, type=int, required=False,
                        help="batch")
    parser.add_argument('--seed', type=int, default=42, required=False,
                        help="random seed for initialization")
    parser.add_argument('--weight_decay', type=float, default=0, required=False,
                        help="weight decay")
    parser.add_argument('--dropout_rate', type=float, default=0.1, required=False,
                        help="dropout rate")
    parser.add_argument('--learning_rate', type=float, default=5e-5, required=False,
                        help="learning rate")
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, required=False,
                        help="adam epsilon")
    parser.add_argument('--warmup_steps', type=float, default=0, required=False,
                        help="warmup steps")
    parser.add_argument('--isContinue', type=str2bool, default=False, required=False,
                        help="train from ckpt")
    parser.add_argument("--extract_ckpt_path", default="ckpt/", type=str, required=True,
                        help="extract ckpt path")
    parser.add_argument("--save_ckpt_path", default="extract_ckpt/", type=str, required=True,
                        help="save ckpt path")

    args = parser.parse_args()
    extract_ckpt(args)