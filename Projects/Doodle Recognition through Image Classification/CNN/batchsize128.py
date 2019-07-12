from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split

def Read_Data():
    with open('./data/xtrain_doodle.pickle', 'rb') as f:
        X_train = pickle.load(f)

    with open('./data/xtest_doodle.pickle', 'rb') as f:
        X_test = pickle.load(f)

    with open('./data/ytrain_doodle.pickle', 'rb') as f:
        y_train = pickle.load(f)

    with open('./data/ytest_doodle.pickle', 'rb') as f:
        y_test = pickle.load(f)

    return X_train, X_test, y_train, y_test

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


X_train, X_test, y_train, y_test = Read_Data()

X_train = X_train.reshape(24000,28,28,1)
X_test = X_test.reshape(6000,28,28,1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
nbr_classes = 10
img_height = 28
img_width = 28
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1)
datagen.fit(X_train)

res = "batchsize128"
if not os.path.exists("./result/" + res):
    os.mkdir("./result/" + res)
os.chdir("./result/" + res)

np.random.seed(171)

model = Sequential()
#conv + ReLU
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),padding='same',
                 activation='relu',input_shape=(28,28,1)))
model.add(Dropout(0.5))

#conv + ReLU
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(Dropout(0.5))

#Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.5))

#conv + ReLUs
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(Dropout(0.5))

#Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.5))

#Fully connected
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, mode='auto')

checkpoint = ModelCheckpoint( res + "_best.h5",monitor='val_acc', verbose=0, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)

before = time.time()


hist = model.fit_generator(datagen.flow(X_train,y_train, batch_size=128),
                              epochs = 1000, validation_data = (X_val,y_val),
                              steps_per_epoch=X_train.shape[0] // 32, callbacks= [earlystop, checkpoint])


print("Time used: {}".format(time.time() - before))

pred = model.predict(X_test)

with open(res + "_pred.pickle", 'wb') as f:
    pickle.dump(pred, f)

with open(res + "_history.pickle", 'wb') as f:
    pickle.dump(hist.history, f)

model.save(res + "_final.h5")
