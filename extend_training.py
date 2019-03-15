
import numpy as np 
import tensorflow as tf 
from utils import *
from tensorflow.keras import layers 

name = "Bartimaeus"
sequence_length = 80
data = "/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/data/rawdata.tfrecords"
model_path = "/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/models/"
archive_path = model_path + "model_archive/"

model = tf.keras.models.load_model(archive_path+name+"/"+name+".h5")

# model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

np.random.seed(42)
indices = np.random.randint(0, 6384, 638)

dataset = read_recurrent_dataset(data, sequence_length, filter_ids=indices, mode='exclude')
validation_set = read_recurrent_dataset(data, sequence_length, filter_ids=indices, mode='include')

callbacks = [
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir=archive_path+name),
  tf.keras.callbacks.ModelCheckpoint(filepath=model_path+"checkpoints/"+name+".ckpt",
                                     save_best_only=True,
                                     period=20)
]


model.fit(x=dataset, 
          epochs=50,
          steps_per_epoch=6384//32,
          validation_data=validation_set,
          validation_steps=638//32,
          callbacks = callbacks)

tf.keras.models.save_model(model,archive_path+name+".h5",overwrite=True)

print("Mission accomplished.")