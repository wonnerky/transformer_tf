import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from model.model_original_transformer import Transformer as Tr_orig
from model.model_keyword_transformer import Transformer as Tr_keyword
import os, argparse, datetime, time, re, collections, random, pickle, json, copy

from utils.wiki_dataset import build_data_loader
from transformers import XLMTokenizer

# tf.enable_eager_execution()

# 소스: Keyword
# 타겟: Text

rng = random.Random(12345)
MAX_LENGTH = 180


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




'''
    pre train for Image Captioning
    method : Make sentence from keyword was extracted the sentence
    tokenizer : word tokenizer
    model : basic transformer
'''

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


''' model train '''
def train(args):

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

    print("==== text gen training start ====")
    # tokenizer_decoder = tfds.features.text.SubwordTextEncoder.load_from_file('en_tfds_wmt8K.MASK')
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
    print("corpus loading finished!")

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

    # learning_rate = CustomSchedule() #
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, epsilon=args.adam_epsilon)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    if args.positional_encoding:
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
    directory = args.ckpt_path
    model_id = args.model_id
    checkpoint_path = os.path.join(directory, model_id)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=100)
    print(f'==== ckpt path : {checkpoint_path} ====')

    if args.isContinue:
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('         !!!!! Latest checkpoint restored!!!!!')
            print('              ', ckpt_manager.latest_checkpoint)
        else:
            raise ValueError('         !!!!! NO checkpoint restored!!!!!')
    else:
        print('initialize all model weight')

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
        train_dataset, train_data_len = build_data_loader(args, tokenizer, type='train')
        print('keyword shuffle complete: {:.2f}s'.format(time.time() - start_time))

        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            step = step + 1

            if batch % 300 == 0 and batch != 0:
                print('Epoch {} Batch {}/{} Loss {:.12f} Accuracy {:.12f}'.format(epoch + 1, batch,
                                                                                  int(train_data_len / args.wiki_batchSize),
                                                                                  train_loss.result(),
                                                                                  train_accuracy.result()))
                log_['epoch'].append(epoch + 1)
                log_['batch'].append(batch)
                log_['loss'].append(train_loss.result().numpy())
                log_['accuracy'].append(train_accuracy.result().numpy())

        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        print("current step : ", step)
        print('Epoch {} ENDED --- Loss {:.12f} Accuracy {:.12f}\n\n'.format(epoch + 1, train_loss.result(),
                                                                            train_accuracy.result()))

    if not os.path.isdir(f'{checkpoint_path}log'):
        os.makedirs(f'{checkpoint_path}log')
    with open(f'{checkpoint_path}log/{model_id}.json.txt', 'wb') as f:
        pickle.dump(log_, f)


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
    parser.add_argument('--wiki_batchSize', type=int, default=60, required=False,
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
    parser.add_argument("--ckpt_path", default="ckpt/", type=str, required=True,
                        help="save ckpt path")
    parser.add_argument("--model_id", default="", type=str, required=True,
                        help="model id")

    args = parser.parse_args()
    train(args)


