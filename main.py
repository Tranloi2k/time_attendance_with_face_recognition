import tensorflow as tf
import src.facenet as facenet
import math
import numpy as np
import os
import sys
import math
import pickle
from sklearn.svm import SVC


FACENET_MODEL_PATH = '20180402-114759.pb'
INPUT_IMAGE_SIZE = 160
data_dir=r'C:\\Users\\Admin\\PycharmProjects\\facenet\\data\\processed\\train'
batch_size = 1000
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
facenet.load_model(FACENET_MODEL_PATH)

# Get input and output tensors
images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]


dataset = facenet.get_dataset(data_dir)
paths, labels = facenet.get_image_paths_and_labels(dataset)

nrof_images = len(paths)
nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
emb_array = np.zeros((nrof_images, embedding_size))
for i in range(nrof_batches_per_epoch):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, nrof_images)
    paths_batch = paths[start_index:end_index]
    images = facenet.load_data(paths_batch, False, False, INPUT_IMAGE_SIZE)
    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)


classifier_filename_exp = os.path.expanduser('my_classifier.pkl')


# Train classifier
print('Training classifier')
model = SVC(kernel='linear', probability=True)
model.fit(emb_array, labels)

# Create a list of class names
class_names = [cls.name.replace('_', ' ') for cls in dataset]

# Saving classifier model
with open(classifier_filename_exp, 'wb') as outfile:
    pickle.dump((model, class_names), outfile)
print('Saved classifier model to file "%s"' % classifier_filename_exp)

