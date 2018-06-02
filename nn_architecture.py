import tensorflow as tf
from global_params import Params

def textCNN1(x_dict, vocab_size, reuse, is_training): 
    with tf.variable_scope("NeuralNet", reuse=reuse): 
        input_X = x_dict["title"]
        # 1 embedding layer
        embed1 = tf.contrib.layers.embed_sequence(input_X, vocab_size, Params.EMBED_SIZE)
        embed1 = tf.expand_dims(embed1, -1)
        embed1 = tf.layers.dropout(embed1, rate=Params.DROP_RATE, training=is_training)
        # 1 convolutional layer
        conv1 = tf.contrib.layers.conv2d(embed1, Params.CONV_FILTER, Params.CONV_SIZE, Params.CONV_STRIDE, 'SAME')
        conv1 = tf.layers.dropout(conv1, rate=Params.DROP_RATE, training=is_training)
        # 1 hidden fully connected layer
        fc1 = tf.contrib.layers.flatten(conv1)
        fc2 = tf.layers.dense(fc1, Params.HIDDEN_FILTER, activation=tf.nn.relu)
        fc2 = tf.layers.dropout(fc2, rate=Params.DROP_RATE, training=is_training)
        # final output layer
        out = tf.layers.dense(fc2, Params.NUM_CLASS)
    return out

def textCNN2(x_dict, vocab_size, reuse, is_training): 
    """ Reference
    http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
    """
    with tf.variable_scope("NeuralNet", reuse=reuse): 
        input_X = x_dict["title"]
        # 1 embedding layer
        embed1 = tf.contrib.layers.embed_sequence(input_X, vocab_size, Params.EMBED_SIZE)
        embed1 = tf.expand_dims(embed1, -1)
        embed1 = tf.layers.dropout(embed1, rate=Params.DROP_RATE, training=is_training)
        # 3 convolutional layers (parallel)
        pool_multiple = []
        for i, conv_size in enumerate(Params.CONV_SIZES):
            # relu activation by default
            convi = tf.contrib.layers.conv2d(embed1, Params.CONV_FILTER, [conv_size, Params.EMBED_SIZE], 1, 'VALID')
            pooli = tf.contrib.layers.max_pool2d(convi, [Params.MAX_DOC_LENGTH - conv_size + 1, 1], 1, 'VALID')
            pool_multiple.append(pooli)
        conv1 = tf.concat(pool_multiple, 3)
        # 1 hidden fully connected layer
        fc1 = tf.contrib.layers.flatten(conv1)
        fc1 = tf.layers.dropout(fc1, rate=Params.DROP_RATE, training=is_training)
        fc2 = tf.layers.dense(fc1, Params.HIDDEN_FILTER, activation=tf.nn.relu)
        fc2 = tf.layers.dropout(fc2, rate=Params.DROP_RATE, training=is_training)
        # final output layer
        out = tf.layers.dense(fc2, Params.NUM_CLASS)
    return out
