# doing the same as SampleDNNClassifier.py but with Keras API instead of estimators.

import tensorflow as tf
import numpy as np
import time, os, sys
tf.enable_eager_execution()
startTick = time.time()

if os.environ['COMPUTERNAME'] == 'DESKTOP-7KDG5DC':
    dataFolder = "C:\\Users\\shhey\\OneDrive - SUNY ESF\\Thesis\\TensorFlow\\Data\\"
elif os.environ['COMPUTERNAME'] == 'ESF-ERE107-1':
    dataFolder = "D:\\Shahriar\\OneDrive - SUNY ESF\\Thesis\\TensorFlow\\Data\\"
else:
    print('Unknown computer. Please add your computer name and OneDrive path to the code')
    sys.exit()

###############################################################################################
# Part I: Reading training/testing features from TFRecords and training a classifier with it  #
###############################################################################################

# These training and test data has been created before
tfrTrainFile = dataFolder + 'tf_demo_train.gz'
tfrTestFile = dataFolder + 'tf_demo_test.gz'
# Dataset = tf.data.TFRecordDataset(tfrTestFile, compression_type='GZIP')
# iterator = Dataset.make_one_shot_iterator()
# with tf.Session() as sess:
#     try:
#       while True:
#         foo = iterator.get_next()
#         print(sess.run([foo]))
#     except tf.errors.OutOfRangeError:
#         pass

# We should know the data structure (6 Landsat bands and one integer landcover class per each pixel
bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7','landcover']
columns = [tf.FixedLenFeature(shape=[], dtype=tf.float32) for k in bands]
featuresDict = dict(zip(bands, columns))

def parse_tfrecord(example):
    features = tf.parse_single_example(example, featuresDict)
    # Extract landcover and remove it from dictionary
    labels = features.pop('landcover')
    labels = tf.one_hot(tf.cast(labels, tf.uint8), 3)
    return features, labels

# Optional processing: Add normalized differences to the dataset
def addFeatures(features, label=tf.one_hot(tf.cast(0, tf.uint8), 3)):
    # Compute normalized difference of two inputs.  If denomenator is zero, add a small delta.
    def normalizedDifference(a, b):
        nd = (a - b) / (a + b)
        nd_inf = (a - b) / (a + b + 0.000001)
        return tf.where(tf.is_finite(nd), nd, nd_inf)

    features['NDVI'] = normalizedDifference(features['B5'], features['B4'])
    # Return list of dictionary values (to be convertible to numpy array for Keras) and pixel label in one-hot format
    return list(features.values()), label

# Generator function to create batches of Keras-compatible training data and one-hot labels
def trainDataGenerator(fileName, numEpochs=None, shuffle=True, batchSize=None):

  dataset = tf.data.TFRecordDataset(fileName, compression_type='GZIP')

  # Map the parsing function over the dataset
  dataset = dataset.map(parse_tfrecord)
  dataset = dataset.map(addFeatures)
  # Shuffle, batch, and repeat.
  if shuffle:
    dataset = dataset.shuffle(buffer_size=batchSize * 10)
  dataset = dataset.batch(batchSize)
  dataset = dataset.repeat(numEpochs)

  return dataset

TR_EPOCHS = 64
training_data = trainDataGenerator(fileName=tfrTrainFile, numEpochs=TR_EPOCHS, batchSize=1, shuffle=False)
testing_data = trainDataGenerator(fileName=tfrTestFile, batchSize=1, shuffle=False)

def keras_model():
    from tensorflow.keras.layers import Dense, Input

    inputs = Input(shape=(7,))
    x = Dense(5, activation='relu')(inputs)
    x = Dense(7, activation='relu')(x)
    x = Dense(5, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

model = keras_model()
learning_rate = 0.05
model.compile(optimizer=tf.train.AdagradOptimizer(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_data, steps_per_epoch=65, epochs=TR_EPOCHS, verbose=0)
model.summary()
loss, test_acc = model.evaluate(testing_data, steps=33, verbose=0)
print('Test accuracy = ', test_acc)
trainingTick = time.time()
trainingTime = trainingTick - startTick
print("Training run time: %s seconds" % (trainingTime))

#model.save_weights(dataFolder+'keras_model_weights_'+str(test_acc)[0:4]+'.h5')
#for layer in model.layers:
#    weights = layer.get_weights()

###############################################################################################
# Part II: Reading a test image from TFRecords for prediction and saving the results          #
###############################################################################################

# Actual image size is 2048 (rows) x 3072 (columns)
image_array = np.zeros((2048, 3072))

# NOTE: You have to know the following data about your TFRecords image to be read.
PATCH_WIDTH = 256
PATCH_HEIGHT = 256
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT
NUM_PATCHES = int(2048*3072 / (PATCH_WIDTH*PATCH_HEIGHT))

# Set input file name(s)
fileNames = [dataFolder+'tf_demo_image_256patch-00000.tfrecord.gz', dataFolder+'tf_demo_image_256patch-00001.tfrecord.gz']

# Defining the data structure is similar to part I, but this time at patch length
bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
columns = [tf.FixedLenFeature(shape=[PATCH_SIZE], dtype=tf.float32) for k in bands]
imageFeaturesDict = dict(zip(bands, columns))

def parse_image(example_proto):
    parsed_features = tf.parse_single_example(example_proto, imageFeaturesDict)
    return parsed_features

# Similar to part I, below input function reads in the TFRecord files exported from an image.
# We also moved parser and feature processing functions into it for more code clearance. Note that
# because the pixels are arranged in patches, we need some additional code to reshape the tensors.
def predictDataGenerator(fileNames):
    # Note that you can make one dataset from many files by specifying a list.
    dataset = tf.data.TFRecordDataset(fileNames, compression_type='GZIP')

    dataset = dataset.map(parse_image, num_parallel_calls=5)
    # Break our long tensors into many littler ones
    dataset = dataset.flat_map(lambda features: tf.data.Dataset.from_tensor_slices(features))
    # Add additional features (NDVI).
    dataset = dataset.map(addFeatures)
    # Read in batches corresponding to patch size.
    dataset = dataset.batch(PATCH_WIDTH * PATCH_HEIGHT)
    return dataset

dataset = predictDataGenerator(fileNames)
# iterator = dataset.make_one_shot_iterator()
# print(iterator.get_next())

def predict_classes(probs):
    if probs.shape[-1] > 1:
        return probs.argmax(axis=-1)
    else:
        return (probs > 0.5).astype('int32')

# Do the prediction from the trained classifier
predictions = predict_classes(model.predict(dataset, steps = NUM_PATCHES))

predictionTick = time.time()
predictionTime = predictionTick - trainingTick
print("Prediction run time: %s seconds" % (predictionTime))

PATCH_PER_ROW , _ = divmod(3072, PATCH_WIDTH)
for patchIndex in range(0,NUM_PATCHES):
    pr, pc = divmod(patchIndex, PATCH_PER_ROW)
    image_array[pr*PATCH_HEIGHT:(pr+1)*PATCH_HEIGHT,pc*PATCH_WIDTH:(pc+1)*PATCH_WIDTH] = \
        predictions[patchIndex * PATCH_SIZE: (patchIndex+1)*PATCH_SIZE].reshape(PATCH_WIDTH,PATCH_HEIGHT)
    print('Patch # ', patchIndex, ' filled')

exportTick = time.time()
exportTime = exportTick - predictionTick
print("Image export run time: %s seconds" % (exportTime))

import matplotlib.pyplot as plt
import matplotlib.image as img
fig = plt.figure()
plt.imshow(image_array)
plt.show()
img.imsave(dataFolder+'image_classified_keras_epochs'+str(TR_EPOCHS)+'_'+str(test_acc)[0:4]+'.png', image_array)

