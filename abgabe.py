import numpy as np 
import tensorflow as tf 
import pandas as pd
from utils import read_recurrent_dataset
from tensorflow.keras import layers 

name = "shorty_mit0806_full"
sequence_length = 50
classes = ['run', 'walk', 'stand', 'sit', 'sit-to-stand', 'stand-to-sit', 
        'stair-up', 'stair-down', 'jump-one-leg', 'jump-two-leg', 'curve-left-step', 
        'curve-right-step', 'curve-left-spin-Lfirst', 'curve-left-spin-Rfirst', 
        'curve-right-spin-Lfirst', 'curve-right-spin-Rfirst', 'lateral-shuffle-left', 
        'lateral-shuffle-right','v-cut-left-Lfirst', 'v-cut-left-Rfirst', 'v-cut-right-Lfirst', 'v-cut-right-Rfirst']

dataset = read_recurrent_dataset("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/data/short/rawchallenge.tfrecords",
                            sequence_length, 
                            filter_subjects=None, 
                            filter_ids=None,
                            mode=None, 
                            training=False)


model = tf.keras.models.load_model("/Users/thomasklein/Projects/BremenBigDataChallenge2019/models/model_archive/"+name+"/"+name+".h5")

results = model.predict(dataset, steps=1738)#should be of size examples,22
#print(results)
predictions = np.argmax(results, axis=1) # should be of shape num_samples
#print(predictions)
df = pd.read_csv("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/challenge.csv")
predicted_labels = [classes[int(x)] for x in predictions]
#for index, row in df.iterrows():
df['Label'] = predicted_labels

df.to_csv("abgabe_"+name+".csv", index=False)