#!/usr/bin/env python
# coding: utf-8

# # WHEAT DISEASE CLASSIFICATION

# # OBJECTIVE
# 
# ## The moto of this project is to Identify the appropriate wheat disease by image processing and classification using Convolutional neural network (CNN).

# # Installing and importing libraries
# 

# In[2]:


get_ipython().system(' pip install opencv-python')


# In[115]:


get_ipython().system(' pip install pandas')


# In[43]:


get_ipython().system('pip install keras')


# In[22]:


get_ipython().system('pip install sklearn')


# In[50]:


get_ipython().system(' pip install --no-cache-dir tensorflow')


# In[109]:


get_ipython().system(' pip install seaborn')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import cv2 #To convert images to array
import os #To specify location of images
import random
import math


# # METHODOLOGY
# 
# ## To achieve the above mentioned objective, a series of cronological methods have been followed as listed below:
# 
# - **Data collection and preprocessing**: Image dataset of various diseased and healthy leaves is stored in a directory and is classified. Images of prominent wheat diseases such as Leaf rust, Nitrogen deficiency and septoria is used in this model. Along with the diseased images a set of healthy leaves images are also fed to the model for better classification
# 
# - **Establishing a baseline** : In this step, a process of building a datum model is carried out. A model is created with certian number of layers (Convolutional, dense, etc.), activation function, epochs, loss function, etc. Furthermore, the model is complied, deployed and evaluated.
# 
# - **Enhancing the model** : In this step, the baseline model is rebuilt by playing with the parameters of the model. With the help of graphical visualization the model's accuracy or performance can be evaluated and enhanced.
# 
# 

# # DATA PREPROCESSING

# In[3]:


DIR = r'C:\Users\Akash Tamate\Desktop\WheatDiseases'
#Creating a Dir
catg = ['Healthy', 'LeafRust', 'NDeficiency', 'septoria']


# In[4]:


IMG_SIZE = 120

data = []

for category in catg:
    folder = os.path.join(DIR, category) 
    #Joins 2 different paths, in this case its DIR and Catg
    label = catg.index(category)
    for img in os.listdir(folder): #Lists all the particular dir present in the folder
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        data.append([img_arr, label])


# In[5]:


len(data)


# In[6]:


random.shuffle(data)


# In[106]:


X = []
y = []

for features, labels in data:
    X.append(features)
    y.append(labels)


# In[107]:


X = np.array(X)
y = np.array(y)


# # Establishing the baseline/datum model

# In[9]:


X = X/255 #scaling


# In[10]:


X.shape


# In[11]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# In[12]:


model = Sequential()


# In[13]:


model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(64, input_shape = X.shape[1:], activation = 'relu'))

model.add(Dense(4,activation = 'softmax'))


# In[14]:


model.compile(optimizer ='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[15]:


history = model.fit(X, y, epochs=10, batch_size = 4, validation_split=0.1)


# In[16]:


hist = history.history
hist = pd.DataFrame(hist)
hist['epoch'] = history.epoch
hist.tail()


# In[17]:


plt.plot(hist['loss'], hist['epoch'])
plt.plot(hist['val_loss'], hist['epoch'], color = 'r')

plt.ylim([0,10])
plt.xlabel('loss')
plt.ylabel('epochs')
plt.show()


# In[18]:


plt.plot(hist['accuracy'], hist['epoch'])
plt.plot(hist['val_accuracy'], hist['epoch'], color = 'r')

plt.ylim([0,10])
plt.xlabel('accuracy')
plt.ylabel('epochs')
plt.show()


# In[19]:


score  = model.evaluate(X,y,verbose = 0)
print("test loss:",score[0])
print('Test accuracy',score[1]*100)


# # # Takeaways from datum model
# 
# - **As the epochs increases the loss and validation loss decreases**
# - **As the epochs increases the accuracy of the model increases**
# - **The model is having an accuracy of 96% at 10th epoch**
# - **The testing accuracy is 94%**

# # Rebuilding the model

# In[20]:


model = Sequential()


# In[21]:


model.add(Conv2D(16, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(16, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(32, input_shape = X.shape[1:], activation = 'relu'))

model.add(Dense(4,activation = 'softmax'))


# In[22]:


model.compile(optimizer ='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[23]:


def loss_accuracy():
    score  = model.evaluate(X,y,verbose = 0)
    print("test loss:",score[0])
    print('Test accuracy',score[1]*100)


# In[24]:


def plot(hist, epochs):
    plt.plot(hist['loss'], hist['epoch'])
    plt.plot(hist['val_loss'], hist['epoch'], color = 'r')

    plt.ylim([0,epochs])
    plt.xlabel('loss')
    plt.ylabel('epochs')
    plt.show()
    loss_accuracy()


# In[25]:


def fit_function(epochs, batch_size, val_split):
    history = model.fit(X, y, epochs = epochs, validation_split = val_split, batch_size = batch_size)
    hist = history.history
    hist = pd.DataFrame(hist)
    hist['epoch'] = history.epoch
    plot(hist, epochs)
    return hist.tail()


# In[26]:


A = fit_function(10, 4, 0.1)
A


# ## After manipulating with layers of the model we arrived at a conclusion that, by decreasing the layers, the model performs well. This generally happens when overfitting arises. 
# 
# # The model accuracy has increased to 99% and validation accuracy has increased to 96%

# # Predicting the results

# In[128]:


y_predicted_labels=[np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]


# In[31]:


y[:5]


# In[120]:


input_image = int(input('ENTER ANY NUMBER BETWEEN 0 TO 907: '))


# In[121]:


plt.matshow(X[input_image])
plt.show()


# In[122]:


y_actual = y[input_image]
y_actual


# In[123]:


y_predicted=model.predict(X)


# In[124]:


y_predicted[input_image]


# In[125]:


result = np.argmax(y_predicted[label1])
result


# In[126]:


if result == 0:
    print('The Wheat crop is healthy')
elif result == 1:
    print('The Wheat crop is affected with the leaf rust')
elif result == 2:
    print('The Wheat crop has nitrogen deficiency')
elif result == 3:
    print('The Wheat crop is affected with Septoria')


# In[129]:


cm=tf.math.confusion_matrix(labels=y,predictions=y_predicted_labels)
cm


# In[130]:


import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('truth')


# # CONCLUSION:
# 
# ## The trained model gives accurate results. The model successfuly identifies the wheat disease with almost 99% accuracy. 
