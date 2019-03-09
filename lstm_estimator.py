import sys
import numpy as np 
import tensorflow as tf 
from preprocessing import *

def model_fn(features, labels, mode, params):

    global final_state
    #global dropout
    global old_hs

    config = params['config']

    old_hs = tf.placeholder(dtype=tf.float32, shape=[None,config.layer_dim])

    inp = tf.unstack(tf.cast(features,tf.float32), axis=1)

    cell = tf.contrib.rnn.BasicLSTMCell(config.layer_dim, dtype=tf.float32)
    hidden_states, final_state = tf.nn.static_rnn(cell, inp, 
                        initial_state=(tf.nn.rnn_cell.LSTMStateTuple(c=old_hs,h=old_hs)), dtype=tf.float32)
    logits = tf.layers.dense(hidden_states[-1], config.output_dim, activation=None)

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

def input_fn(x, y, config):
    return tf.estimator.inputs.numpy_input_fn(
        x=x,#{"x": x},
        y=y,
        num_epochs=None,
        batch_size=config.batchsize,
        shuffle=False
)

def main(data_path, save_path):

    class Config(object):

        def __init__(self):
            self.optimizer = tf.train.AdamOptimizer()
            self.layer_dim = 25
            self.output_dim = 22
            self.num_epochs = 10
            self.batchsize = 1

    config = Config()

    class FeedHook(tf.train.SessionRunHook):

        def __init__(self):
            super(FeedHook, self).__init__()

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(
                fetches=final_state,
                feed_dict={
                    old_hs:hidden})

        def after_run(self, run_context, run_values):
            global hidden
            hidden_state = run_values.results
            hidden = hidden_state.c


    feed_hook = FeedHook()

    print("Generating Data")
    (x_train, y_train), (x_test, y_test) = create_subsequence_dataset(data_path+"train.csv", sub_size=80)
    print("Data Dimensions Sanity Check:")
    print("x_train shape should be [a lot, 80, 19]:",x_train.shape)
    print("y_train shape should be [a lot]",y_train.shape)
    print("x_test shape should be [not as much, 8020, 19]:", x_test.shape)
    print("y_test shape should be [not as much]",y_test.shape)


    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=save_path,
        params={
            'config': config
        })

    for epoch in range(config.num_epochs):

        hidden = np.zeros([config.batchsize, config.layer_dim])

        # Train the Model.
        classifier.train(
            input_fn=input_fn(x_train, y_train, config),
            hooks = [feed_hook],
            steps=len(y_train)) #500*128 = 64000 = number of training samples

        hidden = np.zeros([config.batchsize, config.layer_dim])

        #Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=input_fn(x_test, y_test),
            hooks = [feed_hook],
            steps=len(y_test),
            name="validation")

        print('\nValidation set accuracy after epoch {}: {accuracy:0.3f}\n'.format(epoch+1,**eval_result))

if __name__ == "__main__":
    main(data_path="/Users/thomasklein/Projects/BremenBigDataChallenge2019/bbdc_2019_Bewegungsdaten/",
         save_path="/Users/thomasklein/Projects/BremenBigDataChallenge2019/networks/LSTM/")