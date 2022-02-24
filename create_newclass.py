import tensorflow as tf
import cv2
import numpy as np
import os
import imageio
from deepface import DeepFace
from skimage.transform import resize
d = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20, brightness_range=[0.2, 1.4])
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
metrics = ["cosine", "euclidean", "euclidean_l2"]

def add_data(frame, id):

    paths = r'C:\\Users\\Admin\\PycharmProjects\\facenet\\data\\processed\\train\\'+str(id)
    if os.path.exists(paths):
        files = os.listdir(paths)
        for file in files:  # loop to delete each file in folder
            os.remove(paths + '\\' + str(file))  # delete file
        os.rmdir(paths)

    os.mkdir(paths)


    # Display the resulting frame

    frame = tf.keras.preprocessing.image.img_to_array(
        frame, data_format=None, dtype=None)
    frame = np.expand_dims(frame, axis=0)

    i=0
    for batch in d.flow(x=frame, y=None, batch_size=1, shuffle=True, sample_weight=None, seed=None,save_to_dir=paths, save_prefix='', save_format='png',subset=None):
        i+=1
        if i==9:
            break

    img_paths = list(os.listdir(paths))
    for i in img_paths:
        img_path = paths +'\\'+ str(i)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped = DeepFace.detectFace(img_path=img,
                                      detector_backend=backends[3])

        scaled = resize(cropped, (160,160))
        imageio.imwrite(img_path, scaled)


