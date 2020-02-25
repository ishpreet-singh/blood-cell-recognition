"""
    Code Flow

    1 -> Main method is called, main()
    2 -> Inside main method, data object are initialised. Head to Data Constructor(Data.py) to know more(Just Constructor)
    3 -> A flag is used to test or train
    4 -> In Training, A checkpoint is created to save the progress of the model
    5 -> Then the model is defined using height, width of image of dimension height * weight * 3, 3 -> RGB
    6 -> Note we are using 20_4 model so just head to this model
    7 -> First Feature learning is done using consecutive steps of Conv2d, Batch Normalization and Dropout
    8 -> After Feature Learning Classification is done and layers are added like input, hidden and output
    9 -> Then the Loss Function is Defined RMSprop and cross entropy
   10 -> Now the model is trained using fit_generator method, batch by batch
   11 -> In testing all the results are evaluated using evaluate_generator method, gives out a measure of performance (accuracy)
    
"""

import os
from datetime import datetime

import numpy as np
import tensorflow as tf

my_random_seed = 1337
np.random.seed(my_random_seed)
tf.random.set_seed(my_random_seed)
# tf.set_random_seed(my_random_seed)    

# Intentsionally added step to avoid tensorflow Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam, Adadelta, Adagrad, RMSprop
from keras.callbacks import ModelCheckpoint
from keras import regularizers

from data import Data

# this_path = os.path.dirname(os.path.abspath(__file__))
this_path = os.path.abspath('')


def get_model(out_ht, out_wd, model_id):
    inputs = Input(shape=(out_ht, out_wd, 3))
    # Input is used to instantiate a Keras tensor. A Keras tensor is a tensor object from the underlying 
    # backend (TensorFlow in out case), which we augment with certain attributes that allow us to build a 
    # Keras model just by knowing the inputs and outputs of the model.
    # shape => height/2 , width/2, 3 Here 3 -> RGB

    # Note -> Since we are using 20_4 model for use, directly head to case where model_id = 20_4, line_no: 221

    if model_id == '0':
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu')(inputs)
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id == '1':
        # Ran for 100 epochs: Shows overfitting. best validation accuracy: 78%
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu')(inputs)
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(8, activation='relu')(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id == '2_0':
        # L2 regularization
        # It does slow down the overfitting but validation accuracy gets stuck at ~60%
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu', kernel_regularizer=regularizers.l2())(inputs)
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu', kernel_regularizer=regularizers.l2())(x)
        x = Flatten()(x)
        x = Dense(16, activation='relu', kernel_regularizer=regularizers.l2())(x)
        x = Dense(8, activation='relu', kernel_regularizer=regularizers.l2())(x)
        x = Dense(4, activation='softmax', kernel_regularizer=regularizers.l2())(x)
    elif model_id == '2_1':
        # L1 regularization
        # Accuracy of training and validation got stuck at 25%
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu', kernel_regularizer=regularizers.l1())(inputs)
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu', kernel_regularizer=regularizers.l1())(x)
        x = Flatten()(x)
        x = Dense(16, activation='relu', kernel_regularizer=regularizers.l1())(x)
        x = Dense(8, activation='relu', kernel_regularizer=regularizers.l1())(x)
        x = Dense(4, activation='softmax', kernel_regularizer=regularizers.l1())(x)
    elif model_id == '3_0':
        # Have dropout
        # No overfitting. training loss was still decreasing. train acc: 70%, val_acc: 75%
        # Need more epochs
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(8, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id == '3_1':
        # Batch normalization
        # Could not prevent from overfitting. Train acc: 93% val acc 70%
        x = Conv2D(4, 5, strides=(4, 4), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(4, 5, strides=(4, 4), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(16)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(8)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id == '3_2':
        # Batch normalization + Dropout
        # Faster convergence. Has overvitting. train acc 82% val acc 66%
        x = Conv2D(4, 5, strides=(4, 4), padding='same')(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(16)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(8)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id == '10_0':
        # 3_0 with more epochs
        # No overfitting. train acc: 70%, val_acc: 75%
        # It gets hard to get more gains beyond it
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        # x = Dense(8, activation='relu')(x)
        # x = Dropout(0.2)(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id == '20_0':
        # Reducing the stride on conv layers
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(inputs)
        # x = Dropout(0.2)(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        # x = Dropout(0.2)(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        # x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        # x = Dropout(0.2)(x)
        x = Dense(8, activation='relu')(x)
        # x = Dropout(0.2)(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id == '20_1':
        # 20_0 with dropout
        # Achieves 88% val accuracy in ~100 epochs
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(8, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id == '20_2':
        # Increase model complexity with Dropout
        # 88% val_acc in 80 epochs
        # 95% val_acc in 200 epochs
        x = Conv2D(16, 5, strides=(2, 2), padding='same', activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Conv2D(8, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(8, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id == '20_3':
        # Reduce the kernel size from 5 to 3
        # val acc is lower than with kernel 5
        x = Conv2D(4, 3, strides=(2, 2), padding='same', activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Conv2D(4, 3, strides=(2, 2), padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(4, 3, strides=(2, 2), padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(8, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(4, activation='softmax')(x)

    # --------------------------------------
    #           Just Focus Here
    # --------------------------------------
    elif model_id == '20_4':
        # 20_2 with BatchNorm for faster convergence
        # Gives 97% accuracy. Model saved as model_20_4_e1000.h5

        # In Conv2d -> 2D Convolution Layer, This layer creates a convolution kernel that is 
        # convolved with the layer input to produce a tensor of outputs
        # 1st Argument, Filters -> The number of output channels  i.e. 16
        # 2nd Argument, Kernel Size -> 5, always keep it odd for better performance
        # 3rd Argument, Strides -> (2, 2), Look into doc for better understanding
        # 4th Argument, Padding -> Same, Look into doc for better understanding
        # 5th Argument, Activation -> Relu activation function, Rectified Linear Unit

        # Batch normalization layer -> Normalize the activations of the previous layer at each batch
        # applies a transformation that maintains the mean activation close to 0 and 
        # the activation standard deviation close to 1. Detailed explaination in Google Doc

        # Dropout is a technique used to prevent a model from overfitting.

        # --------------------- Feature Learning Starts ---------------------------
        # Input Layer
        x = Conv2D(16, 5, strides=(2, 2), padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        
        x = Conv2D(8, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        # I Think this is by mistake written twice by the author repeated twice!
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # --------------------- Feature Learning Ends ---------------------------
        # --------------------- Classification Starts ---------------------------

        # Pooling 
        x = Flatten()(x)

        # In our neural network, we are using 3 Hidden layers of 32, 16 and 8 dimension.
        # The Dense is used to specify the fully connected layer. 
        # The arguments of Dense are output dimension which are 32 
        
        # First Hidden Layer
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Second Hidden Layer
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Third Hidden Layer
        x = Dense(8, activation='relu')(x)
        x = Dropout(0.2)(x)

        # Output Layer
        # The output Layer for the case of multiclass classification takes softmax as activation function.
        x = Dense(4, activation='softmax')(x)

        # --------------------- Classification Ends ---------------------------

    elif model_id == '100_0':
        # A low capacity model
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu')(inputs)
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id == '100_1':
        # A low capacity model with dropout to show that capacity isn't enough
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Conv2D(4, 5, strides=(4, 4), padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id == '100_2':
        x = Conv2D(16, 5, strides=(2, 2), padding='same', activation='relu')(inputs)
        x = Conv2D(8, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(8, activation='relu')(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id == '100_3':
        # 100_2 with Dropout
        pass  # Same as 20_2
    elif model_id == '100_4':
        # 100_3 with BatchNormaliation
        # 20_2 with BatchNorm for faster convergence
        # Gives 97% accuracy. Model saved as model_20_4_e1000.h5
        x = Conv2D(16, 5, strides=(2, 2), padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv2D(8, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(8, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id in ('100_5_0', '100_5_1', '100_5_2'):
        # Effect of dropout amount
        if model_id == '100_5_0':
            dropout = 0.1
        elif model_id == '100_5_1':
            dropout = 0.2
        elif model_id == '100_5_2':
            dropout = 0.3
        x = Conv2D(16, 5, strides=(2, 2), padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv2D(8, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(8, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id in ('100_6_0', '100_6_1', '100_6_2', '100_6_3'):
        dropout = 0.2
        # Effect of optimizers
        if model_id == '100_6_0':
            opt = Adam()
        elif model_id == '100_6_1':
            opt = Adadelta()
        elif model_id == '100_6_2':
            opt = Adagrad()
        elif model_id == '100_6_3':
            opt = RMSprop()
        x = Conv2D(16, 5, strides=(2, 2), padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv2D(8, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(8, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(4, activation='softmax')(x)

        outputs = x
        m = Model(inputs=inputs, outputs=outputs)
        print(m.summary())
        m.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return m
    elif model_id in ('100_7_0', '100_7_1', '100_7_2'):
        # Effect of activation function
        dropout = 0.2
        if model_id == '100_7_0':
            act_fn = 'sigmoid'
        elif model_id == '100_7_1':
            act_fn = 'tanh'
        elif model_id == '100_7_2':
            act_fn = 'relu'
        x = Conv2D(16, 5, strides=(2, 2), padding='same', activation=act_fn)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv2D(8, 5, strides=(2, 2), padding='same', activation=act_fn)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation=act_fn)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv2D(4, 5, strides=(2, 2), padding='same', activation=act_fn)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Flatten()(x)
        x = Dense(32, activation=act_fn)(x)
        x = Dropout(dropout)(x)
        x = Dense(16, activation=act_fn)(x)
        x = Dropout(dropout)(x)
        x = Dense(8, activation=act_fn)(x)
        x = Dropout(dropout)(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id in ('100_8_0', '100_8_1'):
        # Effect of Conv filter size
        dropout = 0.2
        act_fn = 'relu'
        if model_id == '100_8_0':
            filter_size = 3  # 3x3
        elif model_id == '100_8_1':
            filter_size = 5  # 5x5
        x = Conv2D(16, filter_size, strides=(2, 2), padding='same', activation=act_fn)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv2D(8, filter_size, strides=(2, 2), padding='same', activation=act_fn)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv2D(4, filter_size, strides=(2, 2), padding='same', activation=act_fn)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv2D(4, filter_size, strides=(2, 2), padding='same', activation=act_fn)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Flatten()(x)
        x = Dense(32, activation=act_fn)(x)
        x = Dropout(dropout)(x)
        x = Dense(16, activation=act_fn)(x)
        x = Dropout(dropout)(x)
        x = Dense(8, activation=act_fn)(x)
        x = Dropout(dropout)(x)
        x = Dense(4, activation='softmax')(x)
    elif model_id == '100_9_0':
        # This could be the best model based on hyperparameters experimentation
        # Nope: overfits slightly faster than validation loss
        dropout = 0.1
        act_fn = 'tanh'
        filter_size = 5
        opt = Adam()
        x = Conv2D(16, filter_size, strides=(2, 2), padding='same', activation=act_fn)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv2D(8, filter_size, strides=(2, 2), padding='same', activation=act_fn)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv2D(4, filter_size, strides=(2, 2), padding='same', activation=act_fn)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Conv2D(4, filter_size, strides=(2, 2), padding='same', activation=act_fn)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Flatten()(x)
        x = Dense(32, activation=act_fn)(x)
        x = Dropout(dropout)(x)
        x = Dense(16, activation=act_fn)(x)
        x = Dropout(dropout)(x)
        x = Dense(8, activation=act_fn)(x)
        x = Dropout(dropout)(x)
        x = Dense(4, activation='softmax')(x)

        outputs = x
        m = Model(inputs=inputs, outputs=outputs)
        print(m.summary())
        m.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return m

    outputs = x
    m = Model(inputs=inputs, outputs=outputs)
    print(m.summary())

    # RMS Prop is an optimizer (Root Mean Square).
    # Optimizers are algorithms or methods used to change the attributes 
    # of your neural network such as weights and learning rate in order to reduce the losses
    opt = RMSprop()
    # Categorical_crossentropy -> specifies that we have multiple classes
    # Metrics -> used to specify the way we want to judge the performance of our neural network, via accuracy in out case
    m.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return m


def main():
    batch_size = 128
    # epochs = 1000
    epochs = 10
    # model_list = ['100_4']
    model_list = ['20_4']
    create_stat_image = False

    resource_dir = os.path.join(this_path, 'resources')
    os.makedirs(resource_dir, exist_ok=True)

    try:
        data = Data(batch_size)
    except Data.DataInitError as e:
        print('Failed to initialize Data instance.\n{:s}'.format(str(e)))
        return

    trainFlag = True

    # if 0:  # Training
    if trainFlag == True:

        for model_id in model_list:  # Training
            # model_path -> Output File for Training
            model_path = os.path.join(resource_dir, model_id + '_model.h5')

            # Save The Check point of the Model, It is an approach where a snapshot of the state of the 
            # system is taken in case of system failure
            cb_save = ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True)
            m = get_model(data.out_ht, data.out_wd, model_id)

            # ----------------------- Training the Model ---------------------

            # fit_generator -> Trains the model on data generated batch-by-batch by a Python generator
            # 1st argument -> generator, Training for now since we are training
            # 2nd argument -> steps_per_epoch, It should typically be equal to ceil(num_samples / batch_size)
            # 3rd argument -> validation_data
            # 4th argument -> validation_steps, No of steps to yield from validation data before stopping at the end of every epoch
            # 5th argument -> callbacks, Passing current saved weight

            # print('############### Training Model ID: {:s} #####################'.format(model_id))
            m.fit_generator(data.get_batch('TRAIN'),
                            steps_per_epoch=data.steps_per_epoch,
                            epochs=epochs,
                            validation_data=data.get_batch('VALIDATION'),
                            validation_steps=data.validation_steps,
                            shuffle=False,
                            callbacks=[cb_save])

    # if 1:  # Testing
    if trainFlag == False:
        # model_path = os.path.join(resource_dir, '20_2_model_e1000.h5')
        # model_path = os.path.join(resource_dir, 'model_20_4_e1000.h5')

        print("Inside Testing ^_^")

        # ----------------------- Testing the Model ----------------------

        model_path = os.path.join(resource_dir, '20_4_model.h5')
        m = load_model(model_path)

        # evaluate_generator ->  uses both your test input and output. 
        # It first predicts output using training input and then evaluates performance by comparing it 
        # against your test output. So it gives out a measure of performance, i.e. accuracy in your case

        eval_out = m.evaluate_generator(data.get_batch('TRAIN'),
                                        steps=data.test_steps)
        print('Train error: ', eval_out)

        eval_out = m.evaluate_generator(data.get_batch('VALIDATION'),
                                        steps=data.test_steps)
        print('Validation error: ', eval_out)

        eval_out = m.evaluate_generator(data.get_batch('TEST'),
                                        steps=data.test_steps)
        print('Test error: ', eval_out)


if __name__ == '__main__':
    main()
