import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Add, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import scikitplot as skplt
from tensorflow.keras.callbacks import ReduceLROnPlateau
import cv2
import imutils
import torch
import torchvision.models as models
import math
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


initial_learning_rate = 0.01
lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate, decay_steps=1000)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

def convolution_block(x, filters, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
X_train_tensor = X_train_tensor.permute(0, 3, 1, 2)
y_train_tensor = torch.from_numpy(y_train).long()
X_val_tensor = torch.from_numpy(X_val.astype(np.float32))
X_val_tensor = X_val_tensor.permute(0, 3, 1, 2)
y_val_tensor = torch.from_numpy(y_val).long()
X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
X_test_tensor = X_test_tensor.permute(0, 3, 1, 2)
y_test_tensor = torch.from_numpy(y_test).long()



train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=4)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=4)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=10)

# perdic def
def predict(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = outputs.view(-1) > 0
            all_predictions.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    return all_predictions, all_labels

# training def
def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            labels = labels.float().view(-1)
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = outputs > 0
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        labels = labels.float().view(-1)
        optimizer.zero_grad()
        outputs = model(inputs).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted = outputs > 0
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy



# resnet 18
# resnet unfreezed
model_resnet_freezed = models.resnet18(pretrained=True)
num_features = model_resnet_freezed.fc.in_features
model_resnet_freezed.fc = nn.Identity()

for param in model_resnet_freezed.parameters():
    param.requires_grad = False

model_resnet_freezed.fc = nn.Linear(num_features, 1)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_resnet_freezed.fc.parameters(), lr=0.001)

for epoch in range(30):
    train_loss, train_acc = train_epoch(model_resnet_freezed, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate_model(model_resnet_freezed, val_loader, criterion)
    print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

predictions_resnet_freezed, labels_resnet_freezed = predict(model_resnet_freezed, test_loader)
accuracy = accuracy_score(labels_resnet_freezed, predictions_resnet_freezed)
print(f'Overall Accuracy: {accuracy:.4f}')

cm = confusion_matrix(labels_resnet_freezed, predictions_resnet_freezed)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


"""# squeeznet"""


model_squeezenet_freezed = models.squeezenet1_1(pretrained=True)


model_squeezenet_freezed.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1,1), stride=(1,1))
model_squeezenet_freezed.num_classes = 1

for param in model_squeezenet_freezed.features.parameters():
    param.requires_grad = False

for param in model_squeezenet_freezed.classifier.parameters():
    param.requires_grad = True

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_squeezenet_freezed.classifier.parameters(), lr=0.001)


for epoch in range(30):
    train_loss, train_acc = train_epoch(model_squeezenet_freezed, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate_model(model_squeezenet_freezed, val_loader, criterion)
    print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

predictions_squeezenet_freezed, labels_squeezenet_freezed = predict(model_squeezenet_freezed, test_loader)

accuracy = accuracy_score(labels_squeezenet_freezed, predictions_squeezenet_freezed)
print(f'Overall Accuracy: {accuracy:.4f}')

cm = confusion_matrix(labels_squeezenet_freezed, predictions_squeezenet_freezed)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()







