import numpy as np
import pickle
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split

def Load_Data():
    categories = ['ant', 'bear', 'bee', 'cat', 'crab', 'dragon', 'elephant', 'mouse', 'sea turtle', 'snail']
    url_data = {}
    for category in categories:
        url_data[category] = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/' + category + '.npy'

    classes_dict = {}
    for key, value in url_data.items():
        response = requests.get(value)
        classes_dict[key] = np.load(BytesIO(response.content))

    for i, (key, value) in enumerate(classes_dict.items()):
        value = value.astype('float32') / 255.
        classes_dict[key] = np.c_[value, i * np.ones(len(value))]

    lst = []
    for key, value in classes_dict.items():
        lst.append(value[:3000])
    doodles = np.concatenate(lst)

    y = doodles[:, -1].astype('float32')
    X = doodles[:, :784]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    with open('xtrain_doodle.pickle', 'wb') as f:
        pickle.dump(X_train, f)

    with open('xtest_doodle.pickle', 'wb') as f:
        pickle.dump(X_test, f)

    with open('ytrain_doodle.pickle', 'wb') as f:
        pickle.dump(y_train, f)

    with open('ytest_doodle.pickle', 'wb') as f:
        pickle.dump(y_test, f)


def Read_Data():
    with open('xtrain_doodle.pickle', 'rb') as f:
        X_train = pickle.load(f)

    with open('xtest_doodle.pickle', 'rb') as f:
        X_test = pickle.load(f)

    with open('ytrain_doodle.pickle', 'rb') as f:
        y_train = pickle.load(f)

    with open('ytest_doodle.pickle', 'rb') as f:
        y_test = pickle.load(f)

    return X_train, X_test, y_train, y_test

