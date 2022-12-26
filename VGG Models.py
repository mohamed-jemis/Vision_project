import cv2
import keras
import numpy as np
from glob import glob

import pandas as pd
import tensorflow as tf
from keras import applications, Sequential, Model
from keras.applications.vgg16 import layers
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense
from sklearn import preprocessing
from tflearn import optimizers

from helpers import *
# from tf import image


label_encoder = preprocessing.LabelEncoder()
def getFiles2(path, train):
    imlist = {}
    labels = {}
    count = 0
    print(path)
    for each in os.listdir(path):
        print(" #### Reading image category ", each, " ##### ")
        imlist[each] = []
        labels[each] = []
        if train == 1:
            for imagefile in os.listdir(path + '/' + each + '/' + 'Train'):
                if imagefile.__contains__('csv'):
                    df = pd.read_csv('C:/Users/avata/Desktop/Vision project/data/' + each + '/Train/' + imagefile)
                    for i in range(40):
                        im = cv2.imread(
                            'C:/Users/avata/Desktop/Vision project/data/' + each + '/Train/' + df.loc(0)[i][0])
                        im = cv2.resize(im, (150, 150))
                        imlist[each].append(im)
                        labels[each].append(df.loc(0)[i][1])


        elif train == 0:
            for imagefile in os.listdir(path + '/' + each + '/' + 'Test'):
                if imagefile.__contains__('csv'):
                    continue
                print("Reading file ", imagefile)
                im = cv2.imread(path + '/' + each + '/' + 'Test' + '/' + imagefile, 0)
                im = cv2.resize(im,(150,150))
                imlist[each].append(im)
                count += 1

    return imlist, labels


imlist, labels = getFiles2('C:/Users/avata/Desktop/Vision project/data', 1)
labels_encoded = []
persons = []
for person, key in labels.items():
    for i in range(len(key)):
        labels_encoded.append(key[i])
        persons.append(person)

labels_encoded = np.array(labels_encoded)
labels_encoded = pd.DataFrame(zip(labels_encoded,persons,labels_encoded), columns=['labels','names','not_encoded_labels'])
labels_encoded['labels'] = label_encoder.fit_transform(labels_encoded['labels'])
train_labels=np.array(labels_encoded['labels'][0:40])
input_ = (150, 150, 3)
EPOCHS = 2
BS = 64
output_ = 2

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_)

model = Sequential()
model.add(base_model)
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dense(output_, activation='softmax'))

model = Model(inputs=model.input, outputs=model.output)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.summary()

earlyStopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=1)
# train_data = np.array(imlist)
# train_data = train_data.reshape(-1, 150, 150, 3)

print(imlist)
early_stop = [earlyStopping]
progess = model.fit(imlist,train_labels , batch_size=BS, epochs=EPOCHS, callbacks=early_stop, validation_split=.3)
acc = progess.history['accuracy']
val_acc = progess.history['val_accuracy']
loss = progess.history['loss']
val_loss = progess.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()