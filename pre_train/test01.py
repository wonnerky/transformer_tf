import tensorflow as tf
import pickle
import numpy as np
import tensorflow_datasets as tfds


MAX_LENGTH = 64
def filter_max_length(x, y, max_length = MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)

def encode(lang1):
    tokens = tokenizer_en.encode(lang1.numpy())
    source = [tokenizer_en.vocab_size] + tokens + [tokenizer_en.vocab_size + 1]
    target = [tokenizer_en.vocab_size] + tokens + [tokenizer_en.vocab_size + 1]

    return source, target


def tf_encode(en):
    result_pt, result_en = tf.py_function(encode, [en], [tf.int64, tf.int64])
    # result_en = result_pt.copy()
    print(type(result_pt))
    # result_pt.set_shape([None])
    # result_en.set_shape([None])

    return result_pt, result_en


file_path = "./data/preprocessing/wiki103/"
with open(f'{file_path}test.txt', 'rb') as f:
    wiki_test = pickle.load(f)

print("data length : ", len(wiki_test))
print(np.array(wiki_test).shape)

wiki_test_ts = tf.Variable(wiki_test, tf.string)
# wiki_test_ts = tf.expand_dims(wiki_test_ts, 0)

print('tensor shape: ', wiki_test_ts.shape)
# for i, line in enumerate(wiki_test_ts):
#     if i == 10:
#         break
#     print(line)


wiki = tf.data.Dataset.from_tensor_slices(wiki_test_ts)
print(wiki)

for idx, line in enumerate(list(wiki.as_numpy_iterator())):
    if idx == 10:
        break
    print('wiki orig:   ', wiki_test[idx])
    print('wiki tensor: ', line)
    print('=' * 30)
print(type(line))

tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file('en_tfds_wmt8K.MASK')
BUFFER_SIZE = 20000
BATCH_SIZE = 64

wiki_dataset = wiki.map(tf_encode)
wiki_dataset = wiki_dataset.filter(filter_max_length)
wiki_dataset = wiki_dataset.cache()
wiki_dataset = wiki_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
wiki_dataset = wiki_dataset.prefetch(tf.data.experimental.AUTOTUNE)
print(wiki_dataset)
print(type(wiki_dataset))


for (batch, (inp, tar)) in enumerate(wiki_dataset):
    print(inp[0])
    break
# txt = []
# for ids in inp[0]:
#     pass
for ids in inp:
    predicted_sentence = tokenizer_en.decode([i for i in ids if i < tokenizer_en.vocab_size])
    print(predicted_sentence)
    print('=' * 30)