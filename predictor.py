import sys
import numpy as np
import pandas as pd
import tensorflow as tf

data_path = "/Users/thomasklein/Projects/BremenBigDataChallenge2019/bbdc_2019_Bewegungsdaten/"

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
    probabilities = classify_movement(df.values)
    return classes[np.argmax(probabilities)]


def classify_movement(array):
    """
    This is the heart of the algorithm.
    """
    prob = np.random.rand(22)
    print(prob)
    return prob

def main(filename):
    """
    Solves the given file.
    """
    anakin_filewalker(filename)


if __name__ == "__main__":
    main(sys.argv[1])