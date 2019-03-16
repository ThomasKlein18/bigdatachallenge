import numpy as np 
import tensorflow as tf 
from utils import *
from tensorflow.keras import layers 
from models import *

name = "RNN_ohne"
sequence_length = 80
data = "/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/data/raw/rawdata.tfrecords"
model_path = "/Users/thomasklein/Projects/BremenBigDataChallenge2019/models/"
archive_path = model_path + "model_archive/"

modus = 'full' 

model = get_rnn(sequence_length, rec_units=32, drop1=0.0, dense_units=32, drop2=0.0)#get_bartimaeus(sequence_length, rec_units=80, drop1=0.0, dense_units=48, drop2=0.8)

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if modus == 'train':

    #np.random.seed(42)
    indices = np.random.randint(0, 6384, 638)
    dataset = read_recurrent_dataset(data, sequence_length, filter_ids=indices, mode='exclude')
    validation_set = read_recurrent_dataset(data, sequence_length, filter_ids=indices, mode='include')

elif modus == 'subject_train':

    indices = [18,19]
    dataset = read_recurrent_dataset(data, sequence_length, filter_subjects=indices, mode='exclude')
    validation_set = read_recurrent_dataset(data, sequence_length, filter_subjects=indices, mode='include')


elif modus == 'full':

    dataset = read_recurrent_dataset(data, sequence_length)
    validation_set = dataset


callbacks = [
# Write TensorBoard logs to `./logs` directory
tf.keras.callbacks.TensorBoard(log_dir=archive_path+name+"_"+modus),
tf.keras.callbacks.ModelCheckpoint(filepath=archive_path+name+"_"+modus+"/"+name+"_"+modus+".h5",
                                    save_best_only=True,
                                    period=1)
]


model.fit(x=dataset, 
        epochs=85,
        steps_per_epoch=6384//32,
        validation_data=validation_set,
        validation_steps=638//32,
        callbacks = callbacks)

#tf.keras.models.save_model(model,archive_path+name+"_"+modus+"/"+name+"_"+modus+".h5",overwrite=False)


print("Mission accomplished.")