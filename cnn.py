# Convolutional Neural Network

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Tutorial: https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Add Convolution layer + MaxPooling layer
# ...

# Step 2 - Adding a second convolutional layer (copy from Step 1)
# ...

# Step 3 - Flattening
# ...

# Step 4 - Full connection
# ...

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
# For Image Augmentation, verify https://keras.io/preprocessing/image/ - ImageDataGenerator class
# Or do it manually if you don't want to use Keras

from keras.preprocessing.image import ImageDataGenerator

# Step 1 - ImageDataGenerator class for data train and test data generation
# Arguments: Rescale, shear_range, zoom_range, horizontal_flip
# For test dataset, only pixel values normalization is required
# ...


# Step 2 - ImageDataGenerator class - for flow_from_directory command, verify https://keras.io/preprocessing/image/ - flow_from_directory
# Arguments: directory of dataset, target_size=(64,64), batch_size=32, class_mode='binary'

#training_set = train_datagen.flow_from_directory(...)

#test_set = test_datagen.flow_from_directory(...)

# Step 3 - fit the model using fit_generator
# Arguments: training_dataset, steps_per_epoch, epochs, validation_data, validation_steps

#classifier.fit_generator(...)

# Part 3 - Validate classifier manually. use predict command to inference model with individual images
# Use images located in dataset/single_prediction

# ...