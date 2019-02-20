# Iris dataset classification example with Keras
import tensorflow as tf
import numpy as np

# Reading Iris csv file and create train, validation, and test data
import csv
# data = np.random.random((1000, 10))
# labels = np.random.random((1000, 2))
data = []
labels = []
with open('Data\\iris.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    try:
        for row in csvReader:
            data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
            if row[4] == 'Iris-setosa':
                labels.append(0)
            elif row[4] == 'Iris-versicolor':
                labels.append(1)
            else:
                labels.append(2)
    except:
        pass
data = np.asarray(data)
labels = np.asarray(labels)

from tensorflow.keras.utils import to_categorical
labels = to_categorical(labels, num_classes=3)

# Creating indoces for train, test, and validation
train_ind = list(range(0,30))+list(range(50,80))+list(range(100,130))
val_ind = list(range(30,35))+list(range(80,85))+list(range(130,135))
test_ind = list(range(35,50))+list(range(85,100))+list(range(135,150))
x_train = data[train_ind,:]
y_train = labels[train_ind]
x_val = data[val_ind,:]
y_val = labels[val_ind]
x_test = data[test_ind,:]
y_test = labels[test_ind]

## Train, validate, and test a simple classifier
from tensorflow.keras import layers
#print(tf.VERSION)
#print(tf.keras.__version__)
model = tf.keras.Sequential([
layers.Dense(64, activation='relu', input_shape=(4,)),
layers.Dense(64, activation='relu'),
layers.Dense(3, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=90, verbose=0)
_, accuracy = model.evaluate(x_val, y_val, batch_size=45, verbose=0)
print('Validation accuracy = ', accuracy)
predicted = model.predict(x_test)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test.argmax(axis=1), predicted.argmax(axis=1))
print('Test data confusion matrix (1):\n',matrix)

## Configuring model with datasets from tensor slices and running it
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.batch(90)
dataset = dataset.repeat()
model.fit(dataset, epochs=10, steps_per_epoch=1, verbose=0)

dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
dataset = dataset.batch(45)
predicted = model.predict(dataset, steps = 1)
matrix = confusion_matrix(y_test.argmax(axis=1), predicted.argmax(axis=1))
print('Test data confusion matrix (2):\n',matrix)

