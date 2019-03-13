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


def recurrent_feature_extractor(data, num_samples):
    """
    data = 2d-numpy array of shape [timesteps, sensors]
    
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


def dataset_creator(in_file, outfile):
    """
    
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

                # this is where the fun begins: extracted_featurelist is a 19-element-list of  np-arrays of length 80
                # We need to get that into a tf.train.Example, which we can then serialize to string and 
                # write to a tfrecords-file.
                serialized_example = serialize_example(extracted_featurelist, label)

                writer.write(serialized_example)