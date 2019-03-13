import sys
import os
import csv
import itertools

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import signal
from scipy import ndimage
from multiprocessing import Pool

from multiprocessing.dummy import Pool as ThreadPool


data_path = "/Users/thomasklein/Projects/BremenBigDataChallenge2019/bbdc_2019_Bewegungsdaten/"


classes = ['run', 'walk', 'stand', 'sit', 'sit-to-stand', 'stand-to-sit', 
        'stair-up', 'stair-down', 'jump-one-leg', 'jump-two-leg', 'curve-left-step', 
        'curve-right-step', 'curve-left-spin-Lfirst', 'curve-left-spin-Rfirst', 
        'curve-right-spin-Lfirst', 'curve-right-spin-Rfirst', 'lateral-shuffle-left', 
        'lateral-shuffle-right','v-cut-left-Lfirst', 'v-cut-left-Rfirst', 'v-cut-right-Lfirst', 'v-cut-right-Rfirst']

sensors = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'Microphone', 'ACC upper X', 'ACC upper Y','ACC upper Z', 'Goniometer X',
          'ACC lower X', 'ACC lower Y', 'ACC lower Z', 'Goniometer Y', 'Gyro upper X', 'Gyro upper Y', 'Gyro upper Z',
          'Gyro lower X', 'Gyro lower Y', 'Gyro lower Z']

variance_sensors = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'Microphone']

smooth_sensors = ['ACC upper X', 'ACC upper Y','ACC upper Z', 'Goniometer X','ACC lower X', 'ACC lower Y', 
                  'ACC lower Z', 'Goniometer Y', 'Gyro upper X', 'Gyro upper Y', 'Gyro upper Z', 'Gyro lower X', 
                  'Gyro lower Y', 'Gyro lower Z']

def anakin_filewalker(file):
    """
    Walks over a csv-file and makes predictions. For each line: extracts the file with the input data. Calls the processing function and retrieves result. Writes result to file. 
    """

    df = pd.read_csv(file)

    df['Label'] = df['Datafile'].apply(classify_file)
    
    df.to_csv(file)


def classify_file(file):
    """
    Classifies the movement that is stored in file "file". 
    """
    classes = ['run', 'walk', 'stand', 'sit', 'sit-to-stand', 'stand-to-sit', 
        'stair-up', 'stair-down', 'jump-one-leg', 'jump-two-leg', 'curve-left-step', 
        'curve-right-step', 'curve-left-spin-Lfirst', 'curve-left-spin-Rfirst', 
        'curve-right-spin-Lfirst', 'curve-right-spin-Rfirst', 'lateral-shuffle-left', 
        'lateral-shuffle-right','v-cut-left-Lfirst', 'v-cut-left-Rfirst', 'v-cut-right-Lfirst', 'v-cut-right-Rfirst']
    #print("Reading file "+data_path+file)
    df = pd.read_csv(data_path+file)
    model = tf.keras.models.load_model("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/recurrent/model/readout_lstm.ckpt")

    dataset_creator(in_file, outfile):

    probabilities = classify_movement(model, df.values)
    return classes[np.argmax(probabilities)]

def smooth(data, windowsize, std):
    kernel = signal.gaussian(windowsize, std=std)
    kernel /= np.sum(kernel)
    return np.convolve(data, kernel, 'valid')

def variance_filter(data, windowsize):
    half = windowsize//2
    res = np.zeros(data.shape[0]-windowsize)
    for i in range(half,len(data)-half):
        res[i-half] = np.std(data[i-half:i+half])
    maxi = np.max(res)
    if(maxi == 0):
        maxi = 0.00001
    return res / maxi

def sample(data, num_samples):
    samples = [int(sample) for sample in np.linspace(0, data.shape[0]-1, num_samples)]
    return data[samples]
    
def smooth_extractor(data, num_samples):
    """
    data = 1d-numpy array of length timestep:
    """
    smoothed = smooth(data,200,50)
    smax = np.max(smoothed)
    if smax == 0:
        smax = 0.00001
    normalized = (smoothed-np.mean(smoothed))/smax
    return sample(normalized, num_samples)

def variance_extractor(data, num_samples):
    """
    data = 1d-numpy array of length timesteps
    """
    var_data = smooth(variance_filter(data,windowsize=100),windowsize=100,std=25)
    return sample(var_data, num_samples)


def dataset_creator(in_file, outfile):
    """
    
    """
    df = pd.read_csv(in_file)
    featurelist = []
    
    with tf.python_io.TFRecordWriter(outfile+".tfrecords") as writer:
        
        for index, row in df.iterrows():
            if(index % 100 == 0):
                print("Digesting",row['Datafile'])
            if(row['Label'] in classes):
                path = row['Datafile']
                data = pd.read_csv(data_path+path).values

                label = classes.index(row['Label'])
                extracted_featurelist = recurrent_feature_extractor(data, 100)
                
                # this is where the fun begins: extracted_featurelist is a 19-element-list of  np-arrays of length 80
                # We need to get that into a tf.train.Example, which we can then serialize to string and 
                # write to a tfrecords-file.
                serialized_example = serialize_example(extracted_featurelist, label)

                writer.write(serialized_example)


def classify_movement(model, data):
    """
    This is the heart of the algorithm.
    """

    num_samples = 80
    #model = tf.keras.models.load_model("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/recurrent/model/readout_lstm.ckpt")
    
    features = []
        
    for sensor in variance_sensors:
        features.append(variance_extractor(data[:,sensors.index(sensor)], num_samples))
        
    if(np.isnan(np.array(features)).any()):
        raise ValueError("Error in variance")
        
    for sensor in smooth_sensors:
        features.append(smooth_extractor(data[:,sensors.index(sensor)], num_samples))
        
    if(np.isnan(np.array(features)).any()):
        raise ValueError("Error in smooth")

    x = np.array(features)
    x = np.reshape(x, (19,80,1))

    return model.predict(x=x.T,
                batch_size=1)


def main(filename):
    """
    Solves the given file.
    """
    anakin_filewalker(filename)


if __name__ == "__main__":
    main("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/abgabe1.csv")

