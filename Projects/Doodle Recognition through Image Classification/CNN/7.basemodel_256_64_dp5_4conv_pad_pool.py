from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import pickle
import numpy as np
import os
import time

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

res = "basemodel_256_64_dp5_4conv_pad_pool.py"
if not os.path.exists("./result/" + res):
    os.mkdir("./result/" + res)
os.chdir("./result/" + res)

np.random.seed(171)
model = Sequential()
#conv layers
n_block = 2
for i in range(n_block):
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',strides=1, padding ='same',input_shape=(img_height, img_width, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',strides=1, padding ='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
#flatten
model.add(Flatten())
#fully connected
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#output: softmax
model.add(Dense(nbr_classes, activation='softmax'))

print("Model: {}".format(res))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
