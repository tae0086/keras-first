import time

import numpy
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_baseline():
    model = Sequential()
    model.add(Dense(60, activation='relu', kernel_initializer='normal', input_dim=8))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


start = time.time()

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

estimators = list()
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, batch_size=5, epochs=100, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print('Results: %.2f%% (%.2f%%)' % (results.mean() * 100, results.std() * 100))

end = time.time()
print('--- %s seconds ---' % (end - start))
