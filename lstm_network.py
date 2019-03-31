import numpy as np 
import tensorflow as tf 
from utils import *
from tensorflow.keras import layers 
from models import *

name = "BigRandomAvoider"
sequence_length = 80
batchsize = 32
data = "/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/data/sparse/rawdata.tfrecords"
model_path = "/Users/thomasklein/Projects/BremenBigDataChallenge2019/models/"
archive_path = model_path + "model_archive/new_shit/"

modus = 'train' 

model = get_bartimaeus(sequence_length, rec_units=128, drop1=0.6, dense_units=64, drop2=0.4)#get_dwarf(sequence_length, rec_units=19, drop=0.35)

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if modus == 'train':

    #np.random.seed(42)
    indices = np.random.randint(0, 6384, 638)
    avoided_subjects = [6,7]
    dataset = read_recurrent_dataset(data, sequence_length, batchsize, filter_ids=indices, filter_subjects=avoided_subjects, id_mode='exclude', sub_mode='exclude', training=True)
    validation_set = read_recurrent_dataset(data, sequence_length, batchsize, filter_ids=indices, filter_subjects=avoided_subjects, id_mode='include', sub_mode='exclude', training=False)

# elif modus == 'subject_train':

#     indices = [18,19]
#     dataset = read_recurrent_dataset(data, sequence_length, batchsize, filter_subjects=indices, mode='exclude', training=True)
#     validation_set = read_recurrent_dataset(data, sequence_length, batchsize, filter_subjects=indices, mode='include', training=False)


# elif modus == 'full':

#     dataset = read_recurrent_dataset(data, sequence_length, batchsize, training=True)
#     validation_set = read_recurrent_dataset(data, sequence_length, batchsize, training=False)


callbacks = [
# Write TensorBoard logs to `./logs` directory
tf.keras.callbacks.TensorBoard(log_dir=archive_path+name+"_"+modus),
tf.keras.callbacks.ModelCheckpoint(filepath=archive_path+name+"_"+modus+"/"+name+"_"+modus+".h5",
                                    save_best_only=True,
                                    period=1)
]


model.fit(x=dataset, 
        epochs=150,
        steps_per_epoch=6384//batchsize,
        validation_data=validation_set,
        validation_steps=638//batchsize,
        callbacks = callbacks)

#tf.keras.models.save_model(model,archive_path+name+"_"+modus+"/"+name+"_"+modus+".h5",overwrite=False)


print("Mission accomplished.")