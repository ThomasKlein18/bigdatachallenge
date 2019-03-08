import sys
import numpy as np 
import tensorflow as tf 

def model_fn(features, labels, mode, params):
    """Model Function"""

    config = params['config']
    inp = tf.unstack(tf.cast(features,config.dtype), axis=1)
    print(inp)

    cell = tf.contrib.rnn.BasicLSTMCell(config.layer_dim, dtype=config.dtype)
    outputs, _ = tf.nn.static_rnn(cell, inp, dtype=config.dtype)
    logits = tf.layers.dense(outputs[-1], config.output_dim, activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    summary_op = tf.summary.merge_all()

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = config.optimizer
    
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main(save_path):

    class Config(object):

        def __init__(self):
            self.optimizer = tf.train.AdamOptimizer()
            self.layer_dim = 25

    config = Config()

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=save_path,
        params={
            'config': config
        })

if __name__ == "__main__":
    main(sys.argv[1])