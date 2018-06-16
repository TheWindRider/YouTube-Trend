import numpy
import data_helpers
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from nn_architecture import textCNN1
from global_params import Params

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

def model_train(model_estimator, X_train, y_train): 
    tf.logging.set_verbosity("INFO")
    # Multiple steps of training
    input_train = tf.estimator.inputs.numpy_input_fn(
        x={'title': X_train}, y=y_train,
        batch_size=Params.BATCH_SIZE, num_epochs=None, shuffle=True)
    model_estimator.train(input_train, steps=Params.NUM_STEP)

def model_evaluate(model_estimator, X_valid, y_valid): 
    tf.logging.set_verbosity("INFO")
    label_category, category_label = data_helpers.video_label()
    # Overall accuracy
    input_test = tf.estimator.inputs.numpy_input_fn(
        x={'title': X_valid}, y=y_valid,
        batch_size=Params.BATCH_SIZE, shuffle=False)
    model_estimator.evaluate(input_test)
    # Confusion matrix
    output_test = model_estimator.predict(input_test)
    y_pred_category = [label_category[y_pred] for y_pred in output_test]
    y_category = [label_category[y_truth] for y_truth in y_valid]
    accuracy_matrix = confusion_matrix(y_category, y_pred_category)
    
    plt.imshow(accuracy_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xticks(numpy.arange(len(label_category)), label_category)
    plt.yticks(numpy.arange(len(label_category)), label_category)