import luigi
import csv
import zipfile,io
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax
import os
from zipfile import ZipFile
import h5py
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import math
import tensorflow as tf
from tensorflow.python.client import device_lib
import glob
from PIL import Image
from boto3 import client
from boto3.session import Session
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint
from numpy import array

class getData(luigi.Task):

    accessKey = luigi.Parameter()
    secretAccessKey = luigi.Parameter()
    bucketName = 'data-brand-logos'

    def output(self):
        return [accessKey, secretAccessKey]

    def run(self):
        session = Session(aws_access_key_id=self.accessKey,
                  aws_secret_access_key=self.secretAccessKey)
        s3 = session.resource('s3')
        your_bucket = s3.Bucket(bucketName)
        for s3_file in your_bucket.objects.all():
            if not s3_file.key.endswith("/"):
                your_bucket.download_file(s3_file.key, s3_file.key)
            else:
                if not os.path.exists(s3_file.key):
                    os.makedirs(s3_file.key)


class BuildKeras(luigi.Task):


    accessKey = luigi.Parameter()
    secretAccessKey = luigi.Parameter()
    bucketName = 'data-brand-logos'

    def output(self):
        print('Success')

    def run(self):
        session = Session(aws_access_key_id=self.accessKey,
                  aws_secret_access_key=self.secretAccessKey)
        s3 = session.resource('s3')
        your_bucket = s3.Bucket(self.bucketName)
        for s3_file in your_bucket.objects.all():
            if not s3_file.key.endswith("/"):
                your_bucket.download_file(s3_file.key, s3_file.key)
            else:
                if not os.path.exists(s3_file.key):
                    os.makedirs(s3_file.key)
        folder ="images/*.jpg"
        images=glob.glob(folder)
        annot_train = np.loadtxt('training_annotation.txt', dtype='a')
        annot_test = np.loadtxt('validating_annotation.txt', dtype='a')
        annot_train = annot_train.astype(str)
        annot_test = annot_test.astype(str)
        brands = []
        for names in annot_train[:,1]:
            if names not in brands:
                brands.append(names)
        for name in brands:
            if not os.path.exists(os.getcwd()+'/train/'+name):
                os.makedirs('train/'+name)
            if not os.path.exists(os.getcwd()+'/validate/'+name):
                os.makedirs('validate/'+name)
        trainImageNameList = annot_train[:,0]
        testImageNameList = annot_test[:,0]
        cwd = os.getcwd()
        for image in images:
            imageName = image.split(sep='\\')[1]
            if imageName in trainImageNameList:
                count = 0
                for names in trainImageNameList:
                    if imageName in names:
                        a = annot_train[count,1]
                        im = Image.open(image)
                        im.load()
                        im.save(os.path.join(cwd, 'train', a)+'\\'+imageName, "JPEG")
                        break
                    count = count + 1
            else:
                count = 0
                for names in testImageNameList:
                    if imageName in names:
                        a = annot_test[count,1]
                        if a not in 'none':
                            im = Image.open(image)
                            im.load()
                            im.save(os.path.join(cwd, 'validate',a)+'\\'+imageName, "JPEG")
                            break
                    count = count + 1
        def create_datagen():
          datagen = ImageDataGenerator(
            rotation_range=20, # Degree range for random rotations
            width_shift_range=0.2, # Range for random horizontal shifts
            height_shift_range=0.2, # Range for random vertical shifts
            shear_range=0.2, # hear Intensity (Shear angle in counter-clockwise direction as radians)
            zoom_range=0.2, # Range for random zoom. If a float
            channel_shift_range=0.2, # Range for random channel shifts
            fill_mode='nearest', # Points outside the boundaries of the input are filled according to the given mode
            horizontal_flip=True, # Randomly flip inputs horizontally
            vertical_flip=True, # Randomly flip inputs vertically
            rescale=1./255
          )
          return datagen

        datagen = create_datagen()
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = datagen.flow_from_directory('Data/train', target_size=(64,64))
        validation_generator = test_datagen.flow_from_directory('Data/test', target_size=(64,64))

        def createModel():
            model = Sequential()
            model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=(64, 64, 3)))
            model.add(Conv2D(64, 3, activation='relu'))
            model.add(MaxPooling2D(pool_size=(3, 3),strides=2 ))
            model.add(BatchNormalization())
        #     model.add(Dropout(0.25))
         
            model.add(Conv2D(128, 3, padding='same', activation='relu'))
            model.add(Conv2D(128, 3, activation='relu'))
            model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
            model.add(BatchNormalization())
        #     model.add(Dropout(0.25))
         
            model.add(Conv2D(256, 3, padding='same', activation='relu'))
            model.add(Conv2D(256, 3, activation='relu'))
            model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
            model.add(BatchNormalization())
        #     model.add(Dropout(0.25))
            
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(27, activation='softmax'))
            return model

        def poly_decay(epoch):
            maxEpochs = 30
            baseLR = 5e-3
            power = 1.0
            # compute the new learning rate based on polynomial decay
            alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
            # return the new learning rate
            return alpha

        def get_available_gpus():
            local_device_protos = device_lib.list_local_devices()
            return [x.name for x in local_device_protos if x.device_type == 'GPU']
        if array(get_available_gpus()).size >= 1:
            with tf.device("/gpu:0"):
                model = createModel()
        else:
            model = createModel()
        opt = SGD(momentum=0.9, lr=5e-3)
        model.compile(metrics=['accuracy'], optimizer=opt, loss='categorical_crossentropy')
        filepath="weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [LearningRateScheduler(poly_decay), checkpoint]
        hist = model.fit_generator(train_generator, steps_per_epoch=300, epochs=30, workers=3, callbacks=callbacks_list, validation_data = validation_generator)

        # grab the history object dictionary
        H = hist.history
         
        # plot the training loss and accuracy
        N = np.arange(0, len(H["loss"]))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H["loss"], label="train_loss")
        plt.plot(N, H["val_loss"], label="test_loss")
        plt.plot(N, H["acc"], label="train_acc")
        plt.plot(N, H["val_acc"], label="test_acc")
        plt.title("MiniGoogLeNet on CIFAR-10")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
         
        # save the figure
        plt.savefig(args["output-keras"])
        plt.close()

        session = Session(aws_access_key_id=self.input()[0],
                  aws_secret_access_key=self.input()[1])
        s3 = session.resource('s3')
        your_bucket = s3.Bucket(self.bucketName)
        your_bucket.upload_file('weights.best.hdf5', '/weights.best.hdf5')
        your_bucket.upload_file('output-keras', '/output-keras.jpg')


if __name__ == '__main__':
    luigi.run()