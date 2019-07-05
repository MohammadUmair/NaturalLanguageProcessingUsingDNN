# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 23:30:29 2019

@author: Umair
"""

import keras as k
import numpy as np

(train_data, train_labels), (test_data, test_labels) = k.datasets.imdb.load_data(num_words=10000)

#print(len(train_data))
#print(train_data[0])
#print(len(train_data[0]))


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

train = vectorize_sequences(train_data)
test = vectorize_sequences(test_data)

#len(train[0])

#y_train = np.asarray(train_labels).astype('float32')
#y_test = np.asarray(test_labels).astype('float32')


model = k.Sequential()
model.add(k.layers.Dense(256, activation='relu', input_shape=(10000,)))
model.add(k.layers.Dense(128, activation='relu'))
model.add(k.layers.Dense(16, activation='relu'))
model.add(k.layers.Dense(1 , activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(train, train_labels, epochs=10, batch_size=512)

result = model.evaluate(test,test_labels)

#result

test[:5]

model.predict(test[:5])

from sklearn.metrics import confusion_matrix

predicted = model.predict_classes(test)

confusion_matrix(test,predicted)





