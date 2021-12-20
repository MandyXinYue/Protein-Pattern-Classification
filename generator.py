import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import tensorflow as tf
import numpy as np
import pandas as pd
import operator
import imageio
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, Reshape, Input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

allow_growth = True

#-------------------------------------------------------------------------------------

class CustomDataGen(tf.keras.utils.Sequence):
    """
    This Data generator takes as input a training or validation dataframe and outputs the preprocessed (stacked and same    dimension for every images) in batches with batchsize = 32) 
    """
    def __init__(self, df, X_col, y_col,
                 batch_size,
                 input_size,
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col # df["ID"]
        self.y_col = y_col # df["Label"]
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        
        self.n = len(self.df)
        self.n_name = 19

    def on_epoch_end(self):
        if self.shuffle:
            self.df.sample(frac=1).reset_index(drop=True)


    def __get_input(self, ID, target_size):
        
        path_1 = "../train_full_aug/"
        path_2 = "../../data/segmented_train/"

        if os.path.isfile(path_1+ID+ "_mt.png"): 
            image_mt = tf.keras.preprocessing.image.load_img(path_1+ ID + "_mt.png", color_mode = "grayscale")
            image_nu = tf.keras.preprocessing.image.load_img(path_1+ ID + "_nu.png", color_mode = "grayscale")
            image_er = tf.keras.preprocessing.image.load_img(path_1+ ID + "_er.png", color_mode = "grayscale")
            image_tp = tf.keras.preprocessing.image.load_img(path_1+ ID + "_tp.png", color_mode = "grayscale")
                
        else: 
            image_mt = tf.keras.preprocessing.image.load_img(path_2+ ID + "_mt.png", color_mode = "grayscale")
            image_nu = tf.keras.preprocessing.image.load_img(path_2+ ID + "_nu.png", color_mode = "grayscale")
            image_er = tf.keras.preprocessing.image.load_img(path_2+ ID + "_er.png", color_mode = "grayscale")
            image_tp = tf.keras.preprocessing.image.load_img(path_2+ ID + "_tp.png", color_mode = "grayscale")
        
        # stack the 4 channels 
        img = np.dstack((image_mt, image_nu, image_er, image_tp)).astype(np.float32) 
        
        # pad the images if smaller 
        if img.shape[0]<= target_size[0] and img.shape[1]<= target_size[1]:
                n_to_add_h = (target_size[0] - img.shape[0])
                top_pad_h = n_to_add_h//2
                bottom_pad_h = n_to_add_h-top_pad_h
                
                n_to_add_w = (target_size[1] - img.shape[1])
                top_pad_w = n_to_add_w//2
                bottom_pad_w = n_to_add_w-top_pad_w
                
                image_arr = np.pad(img, [(top_pad_h, bottom_pad_h), (top_pad_w, bottom_pad_w ), (0,0)], mode = 'constant', constant_values=0)
        # resize the images if bigger
        else: 
            image_arr = tf.image.resize(img,(target_size[0], target_size[1]))

        return image_arr/255.


    def __get_output(self, label):
        #return self.one_hot(label)
    
        y_label = label.split('|')
        one_hot = np.zeros(19)
        for i in y_label:
            nmbr = int(i)
            one_hot[nmbr] = 1
        
        return one_hot
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        id_batch = batches[self.X_col['ID']]

        label_batch = batches[self.y_col['Label']]

        X_batch = np.asarray([self.__get_input(x, self.input_size) for x in id_batch])

        y_batch = np.asarray([self.__get_output(y) for y in label_batch])

        return X_batch, y_batch

    def __getitem__(self, index):

        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size

#-------------------------------------------------------------------------------

#images path
img_f_aug_path = "../train_full_aug/"
img_f_path = "../../data/segmented_train/"

df_f_train = pd.read_csv("../CSVinput/df_f_train.csv")
df_f_val = pd.read_csv("../CSVinput/df_f_val.csv")

# Hyperparameters
target_size_f = (583, 915)
train_batch_size = 32
val_batch_size = 32

#Data generators
traingen = CustomDataGen(df_f_train,
                         X_col = {'ID': 'ID'},
                         y_col = {'Label': 'Label'},
                         batch_size = train_batch_size,
                         input_size = target_size_f)

valgen = CustomDataGen(df_f_val,
                       X_col = {'ID': 'ID'},
                       y_col = {'Label': 'Label'},
                       batch_size = val_batch_size,
                       input_size = target_size_f)

# form steps
train_steps = traingen.n//traingen.batch_size
val_steps = valgen.n//valgen.batch_size


#---------------------MODEL-------------------------------------------

def model_try():
    kernel_size=(3,3)
    pool_size=(4,4)
    first_filters=32
    second_filters=64

    model = Sequential()
    model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (583, 915, 4)))
    model.add(MaxPool2D(pool_size = pool_size))
    model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
    model.add(MaxPool2D(pool_size = pool_size))
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(19, activation = 'sigmoid'))

    # compile the model
    model.compile(optimizer="adam", loss = 'categorical_crossentropy', metrics=['accuracy'])

    return model 

model = model_try() 

# ----------------------------------------------------------------
    
# compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# save the model and weights
model_name = 'Model_test'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]

# since the model is trained for only 10 "mini-epochs", i.e. half of the data is
# not used during training

model.fit(traingen, 
            steps_per_epoch=train_steps,
            validation_data=valgen,
            validation_steps=val_steps,
            epochs=10,
            batch_size = 32,
            callbacks=callbacks_list)