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
    return res

def sample(data, num_samples):
    """
    Samples a 1d-signal num_samples times.
    """
    samples = [int(sample) for sample in np.linspace(0, data.shape[0]-1, num_samples)]
    return data[samples]
    

def recurrent_feature_extractor(data, num_samples):
    """
    Extracts features from a 19-dimensional sequence.

    data = 2d-numpy array of shape [timesteps, sensors]
    num_samples = how many samples to extract
    """

    def smooth_extractor(data, num_samples):
        """
        Samples a signal after smoothing it.

        data = 1d-numpy array of length timestep
        num_samples = how many samples to extract
        """
        smoothed = smooth(data,200,50)
        return sample(smoothed, num_samples)

    def variance_extractor(data, num_samples):
        """
        Samples the local variance of a signal.

        data = 1d-numpy array of length timesteps
        num_samples = how many samples to extract
        """
        var_data = variance_filter(data,windowsize=100)
        return sample(var_data, num_samples)


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



def old_feature_extractor(data, num_samples):
    """
    I wrote a new version of this, but apparently the extracted features were worse, so...
    """

    def old_variance_extractor(data, num_samples):
        """
        Samples the local variance of a signal.
        Differences: variance-data is smoothed, and it is not normalized to the mean, only divided by the max

        data = 1d-numpy array of length timesteps
        num_samples = how many samples to extract
        """
        var_data = smooth(variance_filter(data,windowsize=100), windowsize=100, std=25)
        vmax = np.max(var_data)
        if(vmax == 0):
            vmax = 0.00001
        var_data = var_data/vmax
        return sample(var_data, num_samples)

    def old_smooth_extractor(data, num_samples):
        """
        Samples a signal after smoothing it.

        data = 1d-numpy array of length timestep
        num_samples = how many samples to extract
        """
        smoothed = smooth(data,200,50)
        smax = np.max(smoothed)
        if smax == 0:
            smax = 0.00001
        normalized = smoothed/smax
        return sample(normalized, num_samples)

    features = []
        
    for sensor in variance_sensors:
        features.append(old_variance_extractor(data[:,sensors.index(sensor)], num_samples))
        
    if(np.isnan(np.array(features)).any()):
        raise ValueError("Error in variance")
        
    for sensor in smooth_sensors:
        features.append(old_smooth_extractor(data[:,sensors.index(sensor)], num_samples))
        
    if(np.isnan(np.array(features)).any()):
        raise ValueError("Error in smooth")
        
    return features

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

def serialize_example(featurelist, label, id, subject):
    """
    Creates a tf.Example message from the list of features and the label, where
    every element in the featurelist is actually a sequence=ndarray
    """

    feature = {}
    for i in range(len(featurelist)):
        feature['feature'+str(i)] = tf.train.Feature(float_list=tf.train.FloatList(value=list(featurelist[i])))
        #_float_feature(featurelist[i])
    feature['label'] = _int64_feature(label)
    feature['subject'] = _int64_feature(subject)
    feature['id'] = _int64_feature(id)

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def dataset_creator(in_file, outfile, feature_extractor, *args):
    """
    Creates a dataset (i.e. outfile.tfrecords) from in_file.csv
    """
    df = pd.read_csv(in_file)
    id = 0
    with tf.python_io.TFRecordWriter(outfile+".tfrecords") as writer:
        
        for index, row in df.iterrows():
            if(index % 100 == 0):
                print("Digesting",row['Datafile'])
            if(row['Label'] in classes):
                path = row['Datafile']
                data = pd.read_csv(data_path+path).values

                label = classes.index(row['Label'])
                subject = int(row['Subject'][-2:])
                extracted_featurelist = feature_extractor(data, args[0])

                serialized_example = serialize_example(extracted_featurelist, label, id, subject)
                id = id + 1
                writer.write(serialized_example)
            else:
                print(row['Label'],"not in known classes!")


def challenge_dataset_creator(in_file, outfile, feature_extractor, *args):
    """
    Creates a dataset (i.e. outfile.tfrecords) from in_file.csv
    """
    df = pd.read_csv(in_file)
    id = 0
    with tf.python_io.TFRecordWriter(outfile+".tfrecords") as writer:
        
        for index, row in df.iterrows():
            if(index % 100 == 0):
                print("Digesting",row['Datafile'])
            path = row['Datafile']
            data = pd.read_csv(data_path+path).values

            subject = int(row['Subject'][-2:])
            extracted_featurelist = feature_extractor(data, args[0])

            serialized_example = serialize_example(extracted_featurelist, 0, id, subject)
            id = id + 1
            writer.write(serialized_example)



def read_recurrent_dataset(path,
                            sequence_length, 
                            filter_subjects=None, 
                            filter_ids=None,
                            mode=None, 
                            training=True,):
    """
    mode = whether to only yield elements that are in the lists or whether to ignore elements that are in the list
    """
    if not mode is None and not mode in ['include', 'exclude']:
        raise ValueError("Mode unknwon: Has to be 'include' or 'exclude'")

    if not filter_subjects is None:
        filter_subjects_tensor = tf.constant(filter_subjects, dtype=tf.int64)
    if not filter_ids is None:
        filter_ids_tensor = tf.constant(filter_ids, dtype=tf.int64)

    features = {}
    for i in range(19):
        features['feature'+str(i)] = tf.FixedLenFeature([sequence_length], tf.float32, default_value=np.zeros((sequence_length)))
    features['label'] = tf.FixedLenFeature([], tf.int64, default_value=0)
    features['subject'] = tf.FixedLenFeature([], tf.int64, default_value=0)
    features['id'] = tf.FixedLenFeature([], tf.int64, default_value=0)

    global_means = tf.constant(np.load("global_means.npy"), dtype=tf.float32)
    global_vars = tf.constant(np.load("global_vars.npy"), dtype=tf.float32)

    def _parse_function(example_proto):

        parsed_features = tf.parse_single_example(example_proto, features)

        data = []
        for i in range(19):
            data.append(parsed_features['feature'+str(i)])

        data = tf.reshape(data, (sequence_length,19))
        # data 80, 19
        data = data - global_means
        data = data / global_vars

        return data, tf.one_hot(parsed_features['label'],22)

    def _filter_by_subjects(example_proto):

        parsed_features = tf.parse_single_example(example_proto, features)
        subject = parsed_features['subject']

        if(mode == 'exclude'):
            #return not subject in filter_subjects
            return tf.logical_not(tf.reduce_any(tf.equal(subject,filter_subjects_tensor), axis=0))
        else:
            #return subject in filter_subjects
            return tf.reduce_any(tf.equal(subject,filter_subjects_tensor), axis=0)

    def _filter_by_ids(example_proto):

        parsed_features = tf.parse_single_example(example_proto, features)
        id = parsed_features['id']

        if(mode == 'exclude'):
            #return not id in filter_ids
            return tf.logical_not(tf.reduce_any(tf.equal(id,filter_ids_tensor), axis=0))
        else: 
            # mode == include, return id in filter_ids
            return tf.reduce_any(tf.equal(id,filter_ids_tensor), axis=0)


    dataset = tf.data.TFRecordDataset(path)
    if not filter_subjects is None:
        dataset = dataset.filter(_filter_by_subjects)
    if not filter_ids is None:
        dataset = dataset.filter(_filter_by_ids)
    dataset = dataset.map(_parse_function)
    if training:
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(32, drop_remainder=True)
    else:
        dataset = dataset.batch(1, drop_remainder=False)
    #dataset = dataset.prefetch(1)
    dataset = dataset.repeat()
    return dataset 


def get_partial_mean(data):
    return np.mean(data, axis=0), data.shape[0]
     
def get_partial_variance(data):
    return np.std(data, axis=0), data.shape[0]


def global_info(directory):

    from pathlib import Path

    pathlist = Path(directory).glob('**/*.csv')
    meanlist = []
    varlist = []
    weightlist = []
    
    for filename in pathlist:
        if not "challenge.csv" in str(filename) and not "train.csv" in str(filename): 
            data = pd.read_csv(filename).values
            meanlist.append(np.mean(data, axis=0))
            weightlist.append(data.shape[0])
            varlist.append(np.std(data, axis=0))

    means = np.array(meanlist)
    vars = np.array(varlist)
    weights = np.array(weightlist) / np.sum(weightlist)

    weighted_means = (means.T * weights).T 
    weighted_vars = (vars.T * weights).T 

    print(weighted_means.shape)
    print(weighted_vars.shape)

    np.save("global_means.npy",np.sum(weighted_means, axis=0))
    np.save("global_vars.npy",np.sum(weighted_vars, axis=0))



if __name__ == "__main__":
    #global_info("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bbdc_2019_Bewegungsdaten/")

    # dataset_creator(data_path+"train.csv",
    #            "./data/raw/rawdata", recurrent_feature_extractor, 80)

    # challenge_dataset_creator(data_path+"challenge.csv",
    #            "./data/raw/rawchallenge", recurrent_feature_extractor, 80)
    tf.enable_eager_execution()

    np.random.seed(42)
    #indices = np.random.randint(0, 6384, 638)
    indices = np.arange(0,96)
    ds = read_recurrent_dataset("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/data/raw/rawdata.tfrecords", 80, filter_ids=indices, mode='include')

    res = ds.take(10)
    for r in res:
        print(r)