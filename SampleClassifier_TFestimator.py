# This script uses a 3-class training and test data provided in
# https://colab.research.google.com/drive/18ozzb6Jzmf16x-f2c65hw_-UwOSLcAFF
# which is a collaboratory ipython script provided by Nick Clinton and Chris Brown
# during their presentation titled "Introduction to TensorFlow models and Earth Engine"
# in Google Earth Engine User Summit 2018, 12-14 June 2018, Google Campus, Dublin, Ireland

import tensorflow as tf
import numpy as np
import time, os, sys
startTick = time.time()

import shutil
shutil.rmtree('output', ignore_errors = True)

if os.environ['COMPUTERNAME'] == 'DESKTOP-7KDG5DC':
    dataFolder = "C:\\Users\\shhey\\OneDrive - SUNY ESF\\Thesis\\TensorFlow\\Data\\"
elif os.environ['COMPUTERNAME'] == 'ESF-ERE107-1':
    dataFolder = "D:\\Shahriar\\OneDrive - SUNY ESF\\Thesis\\TensorFlow\\Data\\"
else:
    print('Unknown computer. Please add your computer name and OneDrive path to the code')
    sys.exit()

# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()
#
# #from google.colab import auth
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from oauth2client.client import GoogleCredentials
#
# #auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)

# tf.logging.set_verbosity(tf.logging.INFO)

###############################################################################################
# Part I: Reading training/testing features from TFRecords and training a classifier with it  #
###############################################################################################

# These training and test data has been created before
tfrTrainFile = dataFolder + 'tf_demo_train.gz'
tfrTestFile = dataFolder + 'tf_demo_test.gz'
# trainDataset = tf.data.TFRecordDataset(tfrTrainFile, compression_type='GZIP')
# iterator = trainDataset.make_one_shot_iterator()
# with tf.Session() as sess:
#     try:
#       while True:
#         foo = iterator.get_next()
#         print(sess.run([foo]))
#     except tf.errors.OutOfRangeError:
#         pass

# We should know the data structure (6 Landsat bands and one integer landcover class per each pixel
bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7','landcover']
columns = [tf.FixedLenFeature(shape=[1], dtype=tf.float32) for k in bands]
featuresDict = dict(zip(bands, columns))

# Define parser function over dataset examples (will match the above dictionary)
def parse_tfrecord(example):
  parsed_features = tf.parse_single_example(example, featuresDict)
  labels = parsed_features.pop('landcover')    # Extract landcover and drop it from dictionary
  return parsed_features, tf.cast(labels, tf.int32)
# parsedDataset = trainDataset.map(parse_tfrecord)
# iterator = parsedDataset.make_one_shot_iterator()
# foo = iterator.get_next()
# with tf.Session() as sess:
#     print(sess.run([foo]))

# Optional processing: Add normalized differences to the dataset
def addFeatures(features, label):
    # Compute normalized difference of two inputs.  If denomenator is zero, add a small delta.
    def normalizedDifference(a, b):
        nd = (a - b) / (a + b)
        nd_inf = (a - b) / (a + b + 0.000001)
        return tf.where(tf.is_finite(nd), nd, nd_inf)

    features['NDVI'] = normalizedDifference(features['B5'], features['B4'])
    return features, label

# Defining an input processing function for classifier training. The result of this processing will be a
# batch of input examples, optionally processed for additional features, ready to be fetched on each run
# of this input function by the classifier (we will not be engaged with calling it. Classifier will do it).
def tfrecord_input_fn(fileName, numEpochs=None, shuffle=True, batchSize=None):

  dataset = tf.data.TFRecordDataset(fileName, compression_type='GZIP')

  # Map the parsing function over the dataset
  dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)

  # Add additional features.
  dataset = dataset.map(addFeatures)

  # Shuffle, batch, and repeat.
  if shuffle:
    dataset = dataset.shuffle(buffer_size=batchSize * 10)
  dataset = dataset.batch(batchSize)
  dataset = dataset.repeat(numEpochs)

  # Make a one-shot iterator.
  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels

# Now we define a 3-layer DNN classifier using the estimator API. Good news with estimator API is that they
# bring all the necessary works on setting up a session and graph execution, don't worry about it :)
inputColumns = {tf.feature_column.numeric_column(k) for k in ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'NDVI']}
learning_rate = 0.05
optimizer = tf.train.AdagradOptimizer(learning_rate)
classifier = tf.estimator.DNNClassifier(feature_columns=inputColumns,
                                  hidden_units=[5, 7, 5],
                                  n_classes=3,
                                  model_dir='output',
                                  optimizer=optimizer)

# Train classifier and evaluate it over test set
TR_EPOCHS = 16
classifier.train(input_fn=lambda: tfrecord_input_fn(fileName=tfrTrainFile, numEpochs=TR_EPOCHS, batchSize=1, shuffle=False))
test_acc = classifier.evaluate(
    input_fn=lambda: tfrecord_input_fn(fileName=tfrTestFile, numEpochs=1, batchSize=1, shuffle=False)
)['accuracy']
print('Training acuracy = ', test_acc)
trainingTick = time.time()
trainingTime = trainingTick - startTick
print("Training run time: %s seconds" % (trainingTime))

###############################################################################################
# Part II: Reading a test image from TFRecords for prediction and saving the results          #
###############################################################################################

# NOTE: You have to know the following data about your TFRecords image to be read.
PATCH_WIDTH = 256
PATCH_HEIGHT = 256
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT
PATCH_DIMENSIONS_FLAT = [PATCH_WIDTH * PATCH_HEIGHT, 1]
# Set input file name(s)
fileNames = [dataFolder+'tf_demo_image_256patch-00000.tfrecord.gz', dataFolder+'tf_demo_image_256patch-00001.tfrecord.gz']
# Actual image size is 2048 (rows) x 3072 (columns)

# Defining the data structure is similar to part I, but this time at patch length
bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
columns = [tf.FixedLenFeature(shape=PATCH_DIMENSIONS_FLAT, dtype=tf.float32) for k in bands]
imageFeaturesDict = dict(zip(bands, columns))

# Similar to part I, below input function reads in the TFRecord files exported from an image.
# We also moved parser and feature processing functions into it for more code clearance. Note that
# because the pixels are arranged in patches, we need some additional code to reshape the tensors.
def predict_input_fn(fileNames):
    # Note that you can make one dataset from many files by specifying a list.
    dataset = tf.data.TFRecordDataset(fileNames, compression_type='GZIP')

    def parse_image(example_proto):
        parsed_features = tf.parse_single_example(example_proto, imageFeaturesDict)
        return parsed_features

    # This function adds NDVI to a feature that doesn't have a label.
    def addImageFeatures(features):
        return addFeatures(features, None)[0]

    dataset = dataset.map(parse_image, num_parallel_calls=5)

    # Break our long tensors into many littler ones
    dataset = dataset.flat_map(lambda features: tf.data.Dataset.from_tensor_slices(features))

    # Add additional features (NDVI).
    dataset = dataset.map(addImageFeatures)

    # Read in batches corresponding to patch size.
    dataset = dataset.batch(PATCH_WIDTH * PATCH_HEIGHT)

    # Make a one-shot iterator.
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

# Do the prediction from the trained classifier
# Important note: Set yield_single_examples to True if you want to run the commented section to
# generate output in TFRecords format.
predictions = classifier.predict(input_fn=lambda: predict_input_fn(fileNames), yield_single_examples=False)

# # For writing in TFRecords format
# baseName = 'D:\\Shahriar\\TensorFlow\\SampleTrainingData\\tf_demo_image'
# outputImageFile = baseName + '_predictions.TFRecord'
# outputJsonFile = baseName + '_predictions.json'
# print('Writing to: ' + outputImageFile)
# writer = tf.python_io.TFRecordWriter(outputImageFile)
# # Every patch-worth of predictions we'll dump an example into the output
# # file with a single feature that holds our predictions. Since are predictions
# # are already in the order of the exported data, our patches we create here
# # will also be in the right order.
# patch = [[], [], [], []]
# curPatch = 1
# for pred_dict in predictions:
#     patch[0].append(pred_dict['class_ids'])
#     patch[1].append(pred_dict['probabilities'][0])
#     patch[2].append(pred_dict['probabilities'][1])
#     patch[3].append(pred_dict['probabilities'][2])
#     # Once we've seen a patches-worth of class_ids...
#     if (len(patch[0]) == PATCH_WIDTH * PATCH_HEIGHT):
#         print('Done with patch ' + str(curPatch) + '...')
#         # Create an example
#         example = tf.train.Example(
#             features=tf.train.Features(
#                 feature={
#                     'prediction': tf.train.Feature(
#                         int64_list=tf.train.Int64List(
#                             value=patch[0])),
#                     'bareProb': tf.train.Feature(
#                         float_list=tf.train.FloatList(
#                             value=patch[1])),
#                     'vegProb': tf.train.Feature(
#                         float_list=tf.train.FloatList(
#                             value=patch[2])),
#                     'waterProb': tf.train.Feature(
#                         float_list=tf.train.FloatList(
#                             value=patch[3])),
#
#                 }
#             )
#         )
#         # Write the example to the file and clear our patch array so it's ready for
#         # another batch of class ids
#         writer.write(example.SerializeToString())
#         patch = [[], [], [], []]
#         curPatch += 1
# writer.close()
#!C:|Python36\Scripts\earthengine authenticate --quiet
#outputAssetID = 'users/shshheydari/TF_foobar_predictions'
#!C:|Python36\Scripts\earthengine upload image --asset_id={outputAssetID} {outputImageFile} {outputJsonFile}
#import ee
#ee.Initialize()
#tasks = ee.batch.Task.list()
#print tasks

predictionTick = time.time()
predictionTime = predictionTick - trainingTick
print("Prediction run time: %s seconds" % (predictionTime))


# Making output as a 2-d array for saving as image

# To make the jpeg image directly we need to know the image size
image_array = np.zeros((2048,3072))
PATCH_PER_ROW,_ = divmod(3072, PATCH_WIDTH)
patch_index = 0
for pred_dict in predictions:
    # p, res = divmod(patch_index, 65536)
    # pr, pc = divmod(p, 12)
    # r,c = divmod(res, 256)
    # image_array[pr*256+r][pc*256+c] = pred_dict['class_ids']
    # patch_index = patch_index+1
    # if (patch_index % 65536 == 0):
    #     print('filling patch#',patch_index/65536)
   pr, pc = divmod(patch_index, PATCH_PER_ROW)
   image_array[pr*PATCH_HEIGHT:(pr+1)*PATCH_HEIGHT,pc*PATCH_WIDTH:(pc+1)*PATCH_WIDTH] = pred_dict['class_ids'].reshape(PATCH_WIDTH,PATCH_HEIGHT)
   patch_index = patch_index + 1
   print('Patch # ', patch_index, ' filled')

exportTick = time.time()
exportTime = exportTick - predictionTick
print("Image export run time: %s seconds" % (exportTime))

import matplotlib.pyplot as plt
import matplotlib.image as img
fig = plt.figure()
plt.imshow(image_array)
plt.show()
img.imsave(dataFolder+'image_classified_TFestimator_epochs'+str(TR_EPOCHS)+'_'+str(test_acc)[0:4]+'.png', image_array)

