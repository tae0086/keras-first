import time

import numpy
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def create_model():
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=8))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


start = time.time()

seed = 123
numpy.random.seed(seed)

data = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = data[:, 0:8]
Y = data[:, 8]
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

model = KerasClassifier(build_fn=create_model, batch_size=10, epochs=50, verbose=0)
result = cross_val_score(model, X, Y, cv=kfold)
print(result)
print(numpy.ndarray.mean(result))

end = time.time()
print('--- %s seconds ---' % (end - start))
