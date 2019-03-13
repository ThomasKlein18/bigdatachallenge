import numpy as np 
import tensorflow as tf 

from tfrecord_converter import read_recurrent_dataset
from tensorflow.keras import layers 

name = "Bartimaeus"
model = tf.keras.Sequential()
model.add(layers.LSTM(60, input_shape=[80,19]))
model.add(layers.Dropout(0.8))
#model.add(LSTM(50))
model.add(layers.Dense(32, activation='tanh'))
model.add(layers.Dense(22, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

dataset = read_recurrent_dataset("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/recurrent/rec_full.tfrecords")
testset = read_recurrent_dataset("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/recurrent/rec_test.tfrecords")

callbacks = [
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./recurrent/logs/'+name),
  tf.keras.callbacks.ModelCheckpoint(filepath="/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/recurrent/checkpoints/"+name+".ckpt",
                                     save_best_only=True,
                                     period=20)
]


model.fit(x=dataset, 
          epochs=150,
          steps_per_epoch=6384//32,
          validation_data=testset,
          validation_steps=100,
          callbacks = callbacks)

tf.keras.models.save_model(model,"/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/recurrent/model/"+name+".h5",overwrite=True)

print("Mission accomplished.")