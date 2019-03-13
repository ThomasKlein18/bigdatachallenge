import numpy as np 
import tensorflow as tf 

from tfrecord_converter import read_dataset
from tensorflow.keras import layers 

model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='tanh', input_shape=(30,))) #set value to shape-1
model.add(layers.Dense(64, activation='tanh'))
model.add(layers.Dense(22, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

dataset = read_dataset("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/cafeteria/minifeat.tfrecords")
testset = read_dataset("/Users/thomasklein/Projects/BremenBigDataChallenge2019/bigdatachallenge/cafeteria/minifeat_test.tfrecords")
#val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
#val_dataset = val_dataset.batch(32).repeat()

callbacks = [
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./cafeteria/logs/minifeat2')
]


model.fit(x=dataset, 
          epochs=3000,
          steps_per_epoch=6384//32,
          validation_data=testset,
          validation_steps=100,
          callbacks = callbacks)

model.save_weights('./cafeteria/weights/minifeatmodel2')

#model.load_weights('./weights/my_model')
result = model.predict(dataset, steps=1)

print(result)