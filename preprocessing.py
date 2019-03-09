import sys
import numpy as np
import pandas as pd
import tensorflow as tf

data_path = "/Users/thomasklein/Projects/BremenBigDataChallenge2019/bbdc_2019_Bewegungsdaten/"

def create_dataset(file, skip=1, pad=True):
    """
    Walks over a csv-file and makes predictions. For each line: extracts the file with the input data. Calls the processing function and retrieves result. Writes result to file. 

    file = the file from which to generate the dataset (train.csv)
    skip = how many steps to skip when sampling from a sequence. Choose 1 to skip nothing, but 2 for taking every second value
    """
    X = []
    Y = []
    df = pd.read_csv(file)

    maxlen = 8020 # maximum length of sequences
    classes = ['run', 'walk', 'stand', 'sit', 'sit-to-stand', 'stand-to-sit', 
        'stair-up', 'stair-down', 'jump-one-leg', 'jump-two-leg', 'curve-left-step', 
        'curve-right-step', 'curve-left-spin-Lfirst', 'curve-left-spin-Rfirst', 
        'curve-right-spin-Lfirst', 'curve-right-spin-Rfirst', 'lateral-shuffle-left', 
        'lateral-shuffle-right','v-cut-left-Lfirst', 'v-cut-left-Rfirst', 'v-cut-right-Lfirst', 'v-cut-right-Rfirst']

    for index, row in df.iterrows():
        part = row['Datafile']
        #print("Reading file",part)
        xs = pd.read_csv(data_path+part).values
        xs = xs[::skip] # 100 Hz should be high enough, unless the participants are Ninjas, so skip = 10
        # xs is a np-array of shape (length,19) where length ranges from 1.000 to 10.000
        if(pad):
            length, dim = xs.shape
            limit = maxlen // skip
            if(length == limit):
                pass
            elif(length > limit):
                xs = xs[0:limit,:]
            else:
                xs = np.pad(xs, [(0,limit-length), (0,0)], 'constant', constant_values=0)
            #print(xs.shape)
        try:
            idx = classes.index(row['Label'])
            Y.append(idx)
            X.append(xs)
        except ValueError:
            pass

    if(pad):
        train_X = np.stack(X)
    else:
        train_X = np.array(X)
    train_Y = np.array(Y)

    indices = np.random.randint(low=0, high=len(X), size=maxlen//skip//10)
    test_X = np.take(train_X, indices, axis=0)
    test_Y = np.take(train_Y, indices, axis=0)
    train_X = np.delete(train_X, indices, axis=0)
    train_Y = np.delete(train_Y, indices, axis=0)

    return (train_X, train_Y), (test_X, test_Y)

def slice_that_shit(dataset_X, dataset_Y, sub_size):
    X = []
    Y = []
    for sequence, label in zip(dataset_X, dataset_Y): # sequence is 8020x19
        seq_length, _ = sequence.shape
        residue = seq_length % sub_size
        for sub in np.split(sequence[0:seq_length-residue,:],(seq_length-residue)/sub_size):
            X.append(sub)
            Y.append(label)
    return np.array(X), np.array(Y)

def create_subsequence_dataset(file,sub_size):
    (train_X, train_Y), (test_X, test_Y) = create_dataset(file, 1, False) # yields [6321, 802, 19]
    train_X, train_Y = slice_that_shit(train_X, train_Y, sub_size) # should yield [56.000, 10, 19]
    #test_X, test_Y = slice_that_shit(test_X, test_Y, sub_size)
    return (train_X, train_Y), (test_X, test_Y)


if __name__ == "__main__":
    create_dataset("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bbdc_2019_Bewegungsdaten/train.csv", skip=1)