
# labels=['石張り','吹付タイル','押出成形セメント板','タイル張り','スレート波板張り', "スパンドレル",
#         "コンクリート打ち放し", "コンクリートブロック","ガラスブロック","ガラスカーテンウォール", "ALC板" ]
from keras.models import Model,Sequential
import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers import Activation, Dense, GlobalAveragePooling2D,Input, InputLayer, Lambda, Dropout, BatchNormalization
from keras.backend import sigmoid
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard, EarlyStopping
import sys
from multiprocessing import freeze_support
from keras import regularizers
import numpy as np
from collections import Counter
import continue_fit as cf

from keras.utils import multi_gpu_model

def vgg_based_model(input_shape, n_categories, fulltraining = False):
    base_model=VGG16(weights='imagenet',include_top=False,
                    input_tensor=Input(shape=input_shape))

    #add new layers instead of FC networks
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,kernel_regularizer=regularizers.l2(0.000001),activity_regularizer=regularizers.l1(0.000001))(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Dropout(0.5)(x)
    x=Dense(1024,kernel_regularizer=regularizers.l2(0.000001),activity_regularizer=regularizers.l1(0.000001))(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Dropout(0.5)(x)
    prediction=Dense(n_categories,activation='softmax')(x)
    model=Model(inputs=base_model.input,outputs=prediction)

    if not fulltraining:
        # fix weights before VGG16 14layers
        for layer in base_model.layers[:15]:
            layer.trainable=False
    return model 

import argparse 
if __name__ == "__main__":
    batch_size=32
    input_shape = (224,224,3)
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("-t", "--train_dir", default='resized_cleaned')
    parser.add_argument("-v","--validation_dir",default='resized_val')
    args = parser.parse_args()
    file_name = args.model_name
    train_dir=args.train_dir
    validation_dir=args.validation_dir
    train_datagen=ImageDataGenerator(
        preprocessing_function=preprocess_input,
        height_shift_range=0.02,
        width_shift_range=0.02,
        shear_range=0.05,
        zoom_range=0.05,
        rotation_range=5,
        horizontal_flip=True,
        )

    validation_datagen=ImageDataGenerator(
        preprocessing_function=preprocess_input,
        )

    train_generator=train_datagen.flow_from_directory(
        train_dir,
        target_size=input_shape[0:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
        )
    
    n_categories=len(train_generator.class_indices)
    class_weight ={ clss: len(train_generator.classes) / len(train_generator.class_indices) / count
                     for (clss,count) in Counter(train_generator.classes).most_common() }
    print(train_generator.class_indices)
    print(class_weight)
    print(train_generator.directory)
    validation_generator=validation_datagen.flow_from_directory(
        validation_dir,
        target_size=input_shape[0:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    model = vgg_based_model(input_shape, n_categories)
    # parallel_model = multi_gpu_model(model, gpus=2)

    model.compile(optimizer=Adam(lr=1e-3),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    hist=model.fit_generator(
        train_generator,
        epochs=200,
        workers=4,
        max_queue_size=8,
        initial_epoch=cf.load_epoch_init(file_name),
        # use_multiprocessing=True,
        verbose=1,
        validation_data=validation_generator,
        class_weight=class_weight,
        callbacks=[
            CSVLogger(file_name+'.csv'),
            TensorBoard(file_name),
            cf.early_stopping(model, file_name),
            ])