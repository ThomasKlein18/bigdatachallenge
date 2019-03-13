import numpy as np 
import tensorflow as tf 
import pandas as pd

from tensorflow.keras import layers 

classes = ['run', 'walk', 'stand', 'sit', 'sit-to-stand', 'stand-to-sit', 
        'stair-up', 'stair-down', 'jump-one-leg', 'jump-two-leg', 'curve-left-step', 
        'curve-right-step', 'curve-left-spin-Lfirst', 'curve-left-spin-Rfirst', 
        'curve-right-spin-Lfirst', 'curve-right-spin-Rfirst', 'lateral-shuffle-left', 
        'lateral-shuffle-right','v-cut-left-Lfirst', 'v-cut-left-Rfirst', 'v-cut-right-Lfirst', 'v-cut-right-Rfirst']




def read_recurrent_dataset(path):
    sequence_length = 80
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
    #dataset = dataset.shuffle(1000)
    dataset = dataset.batch(1, drop_remainder=False)
    #dataset = dataset.prefetch(1)
    #dataset = dataset.repeat()
    return dataset 


dataset = read_recurrent_dataset("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/recurrent/abgabe.tfrecords")


model = tf.keras.models.load_model("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/recurrent/model/readout_lstm.h5")

results = model.predict(dataset, steps=1738)#should be of size examples,22
print(results)
predictions = np.argmax(results, axis=1) # should be of shape num_samples
print(predictions)
df = pd.read_csv("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/abgabe1.csv")
predicted_labels = [classes[int(x)] for x in predictions]
#for index, row in df.iterrows():
df['Label'] = predicted_labels

df.to_csv("abgabe1.csv", index=False)