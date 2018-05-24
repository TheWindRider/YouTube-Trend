import tensorflow as tf
from global_params import Params

def textCNN1(x_dict, vocab_size, reuse, is_training): 
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
        
class textCNN(object): 
    def __init__(self, seq_length, num_class, vocab_size, embed_size, conv_sizes, num_filter, num_hidden): 
        self.input_X = tf.placeholder(tf.int32, [None, seq_length])
        self.input_y = tf.placeholder(tf.int64, [None])
        self.drop_rate = tf.placeholder(tf.float32)
        
        with tf.device('/cpu:0'), tf.name_scope("embedding"): # embedding doesn't support GPU
            W = tf.Variable(tf.random_uniform([vocab_size, embed_size], -1.0, 1.0))
            self.embed_layer = tf.nn.embedding_lookup(W, self.input_X)
            self.embed_expand = tf.expand_dims(self.embed_layer, -1)
            self.embed_drop = tf.nn.dropout(self.embed_expand, 1 - self.drop_rate)
            
        pool_output = [] # multiple conv-pooling layers with different patch sizes
        for i, conv_size in enumerate(conv_sizes): 
            with tf.name_scope("conv-maxpool-%s" % conv_size): 
                conv_dim = [conv_size, embed_size, 1, num_filter]
                W = tf.Variable(tf.truncated_normal(conv_dim, stddev=0.1))
                b = tf.Variable(tf.constant(0.1, shape=[num_filter]))
                conv_layer = tf.nn.conv2d(self.embed_drop, W, strides=[1, 1, 1, 1], padding="VALID")
                conv_relu = tf.nn.relu(tf.nn.bias_add(conv_layer, b))
                pool_dim = [1, seq_length - conv_size + 1, 1, 1]
                conv_pool = tf.nn.max_pool(conv_relu, ksize=pool_dim, strides=[1, 1, 1, 1], padding='VALID')
                pool_output.append(conv_pool)
        num_feature = num_filter * len(conv_sizes)
        self.pool_layer = tf.concat(pool_output, 3)
        self.pool_flatten = tf.reshape(self.pool_layer, [-1, num_feature])
        self.pool_drop = tf.nn.dropout(self.pool_flatten, 1 - self.drop_rate)
            
        with tf.name_scope("output"): 
            # A fully-connected hidden layer
            W_h = tf.Variable(tf.truncated_normal([num_feature, num_hidden], stddev=0.1))
            b_h = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
            self.hidden_layer = tf.nn.xw_plus_b(self.pool_drop, W_h, b_h, name="hidden")
            self.hidden_drop = tf.nn.dropout(self.hidden_layer, 1 - self.drop_rate)
            # final layer with num_class
            W = tf.Variable(tf.truncated_normal([num_hidden, num_class], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_class]))
            self.final_layer = tf.nn.xw_plus_b(self.hidden_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.final_layer, 1)
            
        with tf.name_scope("metrics"): 
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.final_layer, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
            corrects = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(corrects, "float"))