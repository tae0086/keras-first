import time

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV


def create_model(initializer='normal', optimizer='adam'):
    model = Sequential()
    model.add(Dense(16, activation='relu', kernel_initializer=initializer, input_dim=8))
    model.add(Dense(8, activation='relu', kernel_initializer=initializer))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=initializer))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # save model
    with open('model.json', 'w') as json:
        json.write(model.to_json())

    return model


# for reproducibility
seed = 1023
numpy.random.seed(seed)

# read data from csv file
data = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = data[:, 0:8]
Y = data[:, 8]

# k-fold cross validation
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# create model and hyperparameter
model = KerasClassifier(build_fn=create_model)
params = dict(
    initializer=['random_normal', 'random_uniform', 'truncated_normal', 'orthogonal', 'identity',
                 'lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
    optimizer=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'],
    batch_size=[10, 20, 30, 40, 50],
    epochs=[50, 75, 100]
)

# grid search
grid = GridSearchCV(estimator=model, param_grid=params, cv=folds, verbose=1)
start = time.time()
result = grid.fit(X, Y)
end = time.time()
print('--- %s seconds ---' % (end - start))

# print results
print('Best Score: %.2f with %s' % (result.best_score_, result.best_params_))
for mean, param in zip(result.cv_results_['mean_test_score'], result.cv_results_['params']):
    print('%.2f with %r' % (mean, param))
