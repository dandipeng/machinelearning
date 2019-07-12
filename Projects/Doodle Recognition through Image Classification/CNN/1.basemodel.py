from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import pickle
import numpy as np
import os

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

X_train, X_test, y_train, y_test = Read_Data()

X_train = X_train.reshape(24000,28,28,1)
X_test = X_test.reshape(6000,28,28,1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
nbr_classes = 10
img_height = 28
img_width = 28

res = "basemodel"
if not os.path.exists("./result/" + res):
    os.mkdir("./result/" + res)
os.chdir("./result/" + res)

np.random.seed(171)
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',strides=1, padding ='same',input_shape=(img_height, img_width, 1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',strides=1))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(nbr_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=50, mode='auto')

checkpoint = ModelCheckpoint( res + "_best.h5",monitor='val_acc', verbose=0, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)

before = time.time()
hist = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split = 0.15, callbacks= [earlystop, checkpoint])
print("Time used: {}".format(time.time() - before))

pred = model.predict(X_test)

with open(res + "_pred.pickle", 'wb') as f:
    pickle.dump(pred, f)

with open(res + "_history.pickle", 'wb') as f:
    pickle.dump(hist.history, f)

model.save(res + "_final.h5")
