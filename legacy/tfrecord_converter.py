import numpy as np 
import pandas as pd 
import tensorflow as tf 


#tf.enable_eager_execution()


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def convert_csv_to_tfrecords(file):
    csv = pd.read_csv(file).values

    def serialize_example(featurelist):
        """
        Creates a tf.Example message ready to be written to a file.
        """

        # Create a dictionary mapping the feature name to the type of list
        feature = {}
        for i in range(csv.shape[1]-1):
            feature['feature'+str(i)] = _float_feature(featurelist[i])
        feature['label'] = _int64_feature(int(featurelist[-1]))

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()


    with tf.python_io.TFRecordWriter("features.tfrecords") as writer:
        for row in csv:
            example = serialize_example(row)
            writer.write(example)


def read_dataset(training_path):
    # set values to shape-1
    features = {}
    for i in range(30):
        features['feature'+str(i)] = tf.FixedLenFeature([], tf.float32, default_value=0)
    features['label'] = tf.FixedLenFeature([], tf.int64, default_value=0)

    def _parse_function(example_proto):

        parsed_features = tf.parse_single_example(example_proto, features)

        data = []
        for i in range(30):
            data.append(parsed_features['feature'+str(i)])

        return data, tf.one_hot(parsed_features['label'],22)


    dataset = tf.data.TFRecordDataset(training_path)

    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32, drop_remainder=True)
    dataset = dataset.prefetch(1)
    dataset = dataset.repeat()
    return dataset 

def read_recurrent_dataset(path):
    sequence_length = 100
    features = {}
    for i in range(19):
        features['feature'+str(i)] = tf.FixedLenFeature([sequence_length], tf.float32, default_value=np.zeros((sequence_length)))
    features['label'] = tf.FixedLenFeature([], tf.int64, default_value=0)

    def _parse_function(example_proto):

        parsed_features = tf.parse_single_example(example_proto, features)

        data = []
        for i in range(19):
            data.append(parsed_features['feature'+str(i)])

        return tf.reshape(data, (sequence_length,19)), tf.one_hot(parsed_features['label'],22)


    dataset = tf.data.TFRecordDataset(path)

    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32, drop_remainder=True)
    dataset = dataset.prefetch(1)
    dataset = dataset.repeat()
    return dataset 

if __name__ == "__main__":
    print(tf.__version__)
    #convert_csv_to_tfrecords("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/smallfeatures.csv")

    # tf.enable_eager_execution()
    # res = read_recurrent_dataset("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/recurrent/rec_features.tfrecords")
    # for val in res.take(1):
    #     print(val)