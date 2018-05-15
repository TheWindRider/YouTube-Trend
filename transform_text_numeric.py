import numpy
import pandas
import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn.model_selection import train_test_split
from global_params import Params

# Create word index
video_english = pandas.read_csv(Params.FILE_NAME_VIDEO)
title_english = video_english["title"].apply(lambda x: x.encode('utf-8'))
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(Params.MAX_DOC_LENGTH)
vocab_processor.fit(video_english["title"])

with gfile.Open('vocabulary.tsv', 'wb') as f: 
    f.write("{}\n".format(Params.PADWORD))
    for word, index in vocab_processor.vocabulary_._mapping.items():
        f.write("{}\n".format(word))
        
# Transform video titles to numeric vectors
def word2vec(sentences, vocab_file): 
    word_english = pandas.read_csv(vocab_file, sep="\t", header=None)
    word_index = tf.contrib.lookup.index_table_from_file(vocab_file, num_oov_buckets=1, default_value=-1)
    words = tf.string_split(sentences)
    word_dense = tf.sparse_tensor_to_dense(words, default_value=Params.PADWORD)
    word_vec = word_index.lookup(word_dense)
    padding = tf.constant([[0, 0],[0, Params.MAX_DOC_LENGTH]])
    word_padded = tf.pad(word_vec, padding)
    word_vec = tf.slice(word_padded, [0, 0], [-1, Params.MAX_DOC_LENGTH])
    return word_vec

titles = tf.placeholder(tf.string, [None,])
title_vec = word2vec(titles, Params.FILE_NAME_VOCAB)

with tf.Session() as sess: 
    tf.tables_initializer().run()
    video_english["words"] = list(title_vec.eval(feed_dict={titles: title_english}))

# Split to training and validation
id_train, id_valid, X_train, X_valid, y_train, y_valid = \
    train_test_split(video_english["video_id"], video_english["words"], video_english["category_id"], test_size=Params.SPLIT_RATE)
numpy.savez(Params.FOLDER_NAME + Params.FILE_NAME_MODEL, 
    id_train=id_train, id_valid=id_valid, feature_train=X_train, feature_valid=X_valid, label_train=y_train, label_valid=y_valid)
