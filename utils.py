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

data_path = "/Users/thomasklein/Projects/BremenBigDataChallenge2019/bbdc_2019_Bewegungsdaten/"


# --------- utility functions ---------- #

def smooth(data, windowsize, std):
    """
    Smoothes a 1d-data array with a gaussian of given size and sigma
    """
    kernel = signal.gaussian(windowsize, std=std)
    kernel /= np.sum(kernel)
    return np.convolve(data, kernel, 'valid')

def variance_filter(data, windowsize):
    """
    Calculates the local variance of a signal by evaluating a sliding window.
    """
    half = windowsize//2
    res = np.zeros(data.shape[0]-windowsize)
    for i in range(half,len(data)-half):
        res[i-half] = np.std(data[i-half:i+half])
    maxi = np.max(res)
    if(maxi == 0):
        maxi = 0.00001
    return res / maxi

def sample(data, num_samples):
    """
    Samples a 1d-signal num_samples times.
    """
    samples = [int(sample) for sample in np.linspace(0, data.shape[0]-1, num_samples)]
    return data[samples]
    
def smooth_extractor(data, num_samples):
    """
    Samples a signal after smoothing it.

    data = 1d-numpy array of length timestep
    num_samples = how many samples to extract
    """
    smoothed = smooth(data,200,50)
    smax = np.max(smoothed)
    if smax == 0:
        smax = 0.00001
    normalized = (smoothed-np.mean(smoothed))/smax
    return sample(normalized, num_samples)

def variance_extractor(data, num_samples):
    """
    Samples the local variance of a signal.

    data = 1d-numpy array of length timesteps
    num_samples = how many samples to extract
    """
    var_data = smooth(variance_filter(data,windowsize=100),windowsize=100,std=25)
    return sample(var_data, num_samples)


def recurrent_feature_extractor(data, num_samples):
    """
    Extracts features from a 19-dimensional sequence.

    data = 2d-numpy array of shape [timesteps, sensors]
    num_samples = how many samples to extract
    """
    features = []
        
    for sensor in variance_sensors:
        features.append(variance_extractor(data[:,sensors.index(sensor)], num_samples))
        
    if(np.isnan(np.array(features)).any()):
        raise ValueError("Error in variance")
        
    for sensor in smooth_sensors:
        features.append(smooth_extractor(data[:,sensors.index(sensor)], num_samples))
        
    if(np.isnan(np.array(features)).any()):
        raise ValueError("Error in smooth")
        
    return features

def threaded_recurrent_feature_extractor(data, num_samples):
    """
    data = 2d-numpy array of shape [timesteps, sensors]
    
    """
    pool = ThreadPool(8)
    
    variance_sequences = []
    smooth_sequences = []
    
    for sensor in variance_sensors:
        variance_sequences.append(data[:,sensors.index(sensor)])
        
    for sensor in smooth_sensors:
        smooth_sequences.append(data[:,sensors.index(sensor)])
        
    var_results = pool.starmap(variance_extractor, zip(variance_sequences, itertools.repeat(num_samples)))
    
    if(np.isnan(np.array(var_results)).any()):
        raise ValueError("NaN after variance feature extraction")
        
    smo_results = pool.starmap(smooth_extractor, zip(smooth_sequences, itertools.repeat(num_samples)))
        
    if(np.isnan(np.array(smo_results)).any()):
        raise ValueError("NaN after smoothing variance extraction")
        
    pool.close()
    pool.join()
    
    return var_results + smo_results

def split_dataset(file, train_name, test_name, percentage=10):
    """
    Splits the file that contains the original dataset in two files, one for training and one for testing.
    
    file = the original file
    """
    df = pd.read_csv(file)
    headers = list(df)
    files = df.values
    
    indices = np.random.randint(low=0, high=files.shape[0], size=files.shape[0]//percentage)
    
    testset = np.take(files, indices, axis=0)
    files = np.delete(files, indices, axis=0)
    
    odf = pd.DataFrame(files)
    odf.columns = headers
    odf.to_csv(train_name+".csv")
    
    tdf = pd.DataFrame(testset)
    tdf.columns = headers
    tdf.to_csv(test_name+".csv")

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(featurelist, label):
    """
    Creates a tf.Example message from the list of features and the label, where
    every element in the featurelist is actually a sequence=ndarray
    """

    feature = {}
    for i in range(len(featurelist)):
        feature['feature'+str(i)] = tf.train.Feature(float_list=tf.train.FloatList(value=list(featurelist[i])))
        #_float_feature(featurelist[i])
    feature['label'] = _int64_feature(label)

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def dataset_creator(in_file, outfile):
    """
    Creates a dataset (i.e. outfile.tfrecords) from in_file.csv
    """
    df = pd.read_csv(in_file)
    
    with tf.python_io.TFRecordWriter(outfile+".tfrecords") as writer:
        
        for index, row in df.iterrows():
            if(index % 100 == 0):
                print("Digesting",row['Datafile'])
            if(row['Label'] in classes):
                path = row['Datafile']
                data = pd.read_csv(data_path+path).values

                label = classes.index(row['Label'])
                extracted_featurelist = recurrent_feature_extractor(data, 80)

                serialized_example = serialize_example(extracted_featurelist, label)

                writer.write(serialized_example)