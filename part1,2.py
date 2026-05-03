import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import scikitplot as skplt
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir

def plot_sample_images():
    # code for displaying multiple images in one figure 
    # create figure 
    fig = plt.figure(figsize=(20, 10)) 

    # setting values to rows and column variables 
    rows = 2
    columns = 2

    # reading images 
    Image1 = cv2.imread('brain_tumor_dataset/yes/Y1.jpg') 
    Image2 = cv2.imread('brain_tumor_dataset/yes/Y2.jpg') 
    Image3 = cv2.imread('brain_tumor_dataset/no/N5.jpg') 
    Image4 = cv2.imread('brain_tumor_dataset/no/N2.jpg') 

    # Adds a subplot at the 1st position 
    fig.add_subplot(rows, columns, 1) 

    # showing image 
    plt.imshow(Image1) 
    plt.axis('off') 
    plt.title("Yes") 

    # Adds a subplot at the 2nd position 
    fig.add_subplot(rows, columns, 2) 

    #showing image 
    plt.imshow(Image3) 
    plt.axis('off') 
    plt.title("No") 

    # Adds a subplot at the 3rd position 
    fig.add_subplot(rows, columns, 3) 

    # showing image 
    plt.imshow(Image2) 
    plt.axis('off') 

    # Adds a subplot at the 4th position 
    fig.add_subplot(rows, columns, 4) 

    # showing image 
    plt.imshow(Image4) 
    plt.axis('off') 
    
plot_sample_images()

yes_file = 'brain_tumor_dataset/yes' 
no_file = 'brain_tumor_dataset/no'

IMG_WIDTH, IMG_HEIGHT = (240, 240)


def load_data(dir_list, image_size):
    """
    Read images, resize and normalize them. 
    Arguments:
        dir_list: list of strings representing file directories.
    Returns:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """

    # load all images in a directory
    X = []
    y = []
    image_width, image_height = image_size
    
    for directory in dir_list:
        for filename in listdir(directory):
            # load the image
            image = cv2.imread(directory + '\\' + filename)
            # resize image
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize values
            image = image / 255.
            # convert image to numpy array and append it to X
            X.append(image)
            # append a value of 1 to the target array if the image
            # is in the folder named 'yes', otherwise append 0.
            if directory[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])
                
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle the data
    X, y = shuffle(X, y)
    
    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')
    
    return X, y

def augment_data(file_dir, n_generated_samples, save_to_dir):
    """
    Arguments:
        file_dir: A string representing the directory where images that we want to augment are found.
        n_generated_samples: A string representing the number of generated samples using the given image.
        save_to_dir: A string representing the directory in which the generated images will be saved.
    """
    
    #from keras.preprocessing.image import ImageDataGenerator
    #from os import listdir
    
    data_gen = ImageDataGenerator(rotation_range=10, 
                                  width_shift_range=0.1, 
                                  height_shift_range=0.1, 
                                  shear_range=0.1, 
                                  brightness_range=(0.3, 1.0),
                                  horizontal_flip=True, 
                                  vertical_flip=True, 
                                  fill_mode='nearest'
                                 )

    
    for filename in listdir(file_dir):
        # load the image
        image = cv2.imread(file_dir + '\\' + filename)
        # reshape the image
        image = image.reshape((1,)+image.shape)
        # prefix of the names for the generated sampels.
        save_prefix = 'aug_' + filename[:-4]
        # generate 'n_generated_samples' sample images
        i=0
        for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir, 
                                           save_prefix=save_prefix, save_format='jpg'):
            i += 1
            if i > n_generated_samples:
                break
            
augmented_data_path = 'brain_tumor_dataset'

# augment data for the examples with label equal to 'yes' representing tumurous examples
#augment_data(file_dir='brain_tumor_dataset/yes', n_generated_samples=6, save_to_dir=augmented_data_path+'/yes')
# augment data for the examples with label equal to 'no' representing non-tumurous examples
#augment_data(file_dir='brain_tumor_dataset/no', n_generated_samples=9, save_to_dir=augmented_data_path+'/no')
augmented_yes = augmented_data_path + '/yes' 
augmented_no = augmented_data_path + '/no'

IMG_WIDTH, IMG_HEIGHT = (240, 240)

X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))

def split_data(X, y, test_size=0.2):
       
    """
    Splits data into training, development and test sets.
    Arguments:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    Returns:
        X_train: A numpy array with shape = (#_train_examples, image_width, image_height, #_channels)
        y_train: A numpy array with shape = (#_train_examples, 1)
        X_val: A numpy array with shape = (#_val_examples, image_width, image_height, #_channels)
        y_val: A numpy array with shape = (#_val_examples, 1)
        X_test: A numpy array with shape = (#_test_examples, image_width, image_height, #_channels)
        y_test: A numpy array with shape = (#_test_examples, 1)
    """
    
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    
    
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of development examples = " + str(X_val.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_val (dev) shape: " + str(X_val.shape))
print ("Y_val (dev) shape: " + str(y_val.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(y_test.shape))

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Add, DepthwiseConv2D, BatchNormalization, ReLU
'''
def convolution_block(x, filters, kernel_size, strides=1):
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def residual_block(x, filters, kernel_size=3, strides=1, activation='relu'):
    # Residual Block with Depthwise Separable Convolutions
    x_shortcut = x
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(X)
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = add([x_shortcut, x]) # Adding back the shortcut path
    return x


def depthwise_separable_conv(x, filters, kernel_size=3, strides=1):
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', name='depthwise_conv2')(x)
    x = BatchNormalization(axis=3, name='bn3')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='pointwise_conv2')(x)
    x = BatchNormalization(axis=3, name='bn4')(x)
    x = Activation('relu')(x)
    return x


def CNN(input_shape, num_classes):
    X_input = Input(input_shape)
    X = ZeroPadding2D((2, 2))(X_input)

    # Initial CONV -> BN -> RELU Block
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # Residual Block with Depthwise Separable Convolutions
    X_shortcut = X
    X = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', name='depthwise_conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)
    X = Conv2D(32, (1, 1), strides=(1, 1), padding='same', name='pointwise_conv1')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)
    X = add([X_shortcut, X]) # Adding back the shortcut path

    # Add Depthwise Separable Convolutional Layer
    X = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', name='depthwise_conv2')(X)
    X = BatchNormalization(axis=3, name='bn3')(X)
    X = Activation('relu')(X)
    X = Conv2D(32, (1, 1), strides=(1, 1), padding='same', name='pointwise_conv2')(X)
    X = BatchNormalization(axis=3, name='bn4')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((4, 4), name='max_pool0')(X)

    # Further reduce dimension and pool
    X = MaxPooling2D((4, 4), name='max_pool1')(X)

    # FLATTEN X and FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(128, activation='relu', name='fc3')(X)
    X = Dense(128, activation='relu', name='fc1')(X)
    X = Dense(1, activation='sigmoid', name='fc2')(X)

    model = Model(inputs=X_input, outputs=X, name='BrainDetectionModel')

    return model
'''

def residual_block(X, filters):
    X_shortcut = X
    X = Conv2D(filters, (3, 3), strides=(1, 1), padding='same', activation='relu')(X)
    X = Conv2D(filters, (3, 3), strides=(1, 1), padding='same', activation='relu')(X)
    X = add([X_shortcut, X])
    X = Activation('relu')(X)
    return X


def CNN(input_shape , num_classes):
    X_input = Input(input_shape)
    X = Conv2D(16, (7, 7), strides=(1, 1), padding='same', activation='relu')(X_input)


    X = residual_block(X, 16)
    X = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (1, 1), strides=(1, 1), padding='same', activation='relu')(X)

    X = residual_block(X, 16)
    X = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (1, 1), strides=(1, 1), padding='same', activation='relu')(X)

    X = MaxPooling2D((4, 4))(X)
    X = MaxPooling2D((4, 4))(X)

    X = Flatten()(X)
    X = Dense(128, activation='relu')(X)
    #X = Dense(128, activation='relu')(X)
    #X = Dense(1, activation='sigmoid')(X)
    X = Dense(num_classes, activation='softmax')(X)
    
    model = Model(inputs=X_input, outputs=X, name='BrainDetectionModel')
    
    return model


# Example usage:
#input_shape = (28, 28, 1) # Example input shape for MNIST dataset
num_classes = 2 # Example number of output classes
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
model = CNN(IMG_SHAPE, num_classes)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# tensorboard
log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')

# checkpoint
# unique file name that will include the epoch and the validation (development) accuracy
filepath="brain_tumor_dataset/accuracy"
# save the model with the best validation (development) accuracy till now
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))

model.fit(x=X_train, y=y_train, batch_size=100, epochs=1, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])
#model.fit(x=X_train, y=y_train, batch_size=32, epochs=3, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])

history = model.history.history
for key in history.keys():
    print(key)

def plot_metrics(history):
    
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['accuracy']
    #val_acc = history['val_acc']
    
    # Loss
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    # Accuracy
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    #plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
    
plot_metrics(history) 

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

#confusion matrix- Predict classes
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Print classification report
print(classification_report(y_true_classes, y_pred_classes))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, range(num_classes), rotation=45)
plt.yticks(tick_marks, range(num_classes))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()


#overall accuracy- Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Overall Accuracy: {accuracy * 100:.2f}%')
plt.show()


# precision-Predict classes
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate precision for each class
precision = precision_score(y_true_classes, y_pred_classes, average=None)

# Print precision for each class
print("precision score:", precision)
    
    
    
#recall- Predict classes
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate recall for each class
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
print("recall score:", recall)
# Print recall for each class
#for i, r in enumerate(recall):
 #   print(f'Recall for class {i}: {r:.4f}')
    
    
#f1-score- Predict classes
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate F1-score for each class
f1 = f1_score(y_true_classes, y_pred_classes , average='weighted')

print("F1 Score:" , f1)
        
y_score = model.predict(X_test)

# Assuming 'y_true' contains the true labels and 'y_score' contains the predicted scores or probabilities
# Plot ROC curve
skplt.metrics.plot_roc(y_test, y_score)
plt.show()




