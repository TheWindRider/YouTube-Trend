import data_helpers
import tensorflow as tf
from global_params import Params
from nn_architecture import textCNN1

X_train, X_valid, y_train, y_valid, total_words = data_helpers.load_data()

# Model setup using tf.estimator
def model_classify(features, labels, mode): 
    logits_train = textCNN1(features, total_words, False, True)
    logits_test = textCNN1(features, total_words, True, False)
    # for prediction 
    classes_test = tf.argmax(logits_test, axis=1)
    probas_test = tf.nn.softmax(logits_test)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=classes_test)
    acc_op = tf.metrics.accuracy(labels=labels, predictions=classes_test, name="acc_test")
    # for training
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=Params.LEARN_RATE)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    # model specs
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=classes_test,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})
    
    return estim_specs

model_estimator = tf.estimator.Estimator(model_classify)
tf.logging.set_verbosity("INFO")
# Multiple steps of training
input_train = tf.estimator.inputs.numpy_input_fn(
    x={'title': X_train}, y=y_train,
    batch_size=Params.BATCH_SIZE, num_epochs=None, shuffle=True)
model_estimator.train(input_train, steps=Params.NUM_STEP)
# Evaluation
input_test = tf.estimator.inputs.numpy_input_fn(
    x={'title': X_valid}, y=y_valid,
    batch_size=Params.BATCH_SIZE, shuffle=False)
model_estimator.evaluate(input_test)