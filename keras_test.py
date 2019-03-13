
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, SimpleRNN

from preprocessing import *

# cut sequences at this length
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = create_median_filtered_dataset(data_path+"train.csv", 100)#create_subsequence_dataset(data_path+"train.csv", sub_size=80)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('x_train shape:', x_train.shape) # should be [6321,802,19]
print('x_test shape:', x_test.shape)
print("y_train shape:", y_train.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(1000, 19, mask_zero = True))
model.add(LSTM(50, return_sequences=True, input_shape=(None)))
model.add(LSTM(50))
model.add(Dense(22, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)

# best params

# 20% train set: skip 100, 50 lstm-units, 22 dense with softmax