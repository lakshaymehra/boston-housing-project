import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
    path="boston_housing.npz", test_split=0.2, seed=113
)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std

x_test -= mean
x_test /= std

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae'])
    return model


import numpy as np

k = 4
num_val_samples = len(x_train) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print(f'Processing fold # {i}')
    val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [x_train[:i * num_val_samples],
         x_train[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [y_train[:i * num_val_samples],
         y_train[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    model.fit(partial_train_data,
              partial_train_targets,
              epochs=num_epochs,
              batch_size=1,
              verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(f'all_scores : {all_scores}')
print(f'mean all scores : {np.mean(all_scores)}')

model = build_model()
model.fit(x_train, y_train, epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(x_test, y_test)

print(test_mae_score)

model.save('boston_housing.h5')

l = np.array(x_test[1].reshape(1,-1))
print(l.shape)
print(l)

pred = model.predict(l)

print(pred)

print(y_test[1])
#
# from flask import Flask, request, jsonify, render_template
#
# app = Flask(__name__, template_folder='C:/Users/mehra/OneDrive/Documents/GitHub/boston-housing-project')
# import numpy as np
# from tensorflow import keras
#
# model = keras.models.load_model("boston_housing.h5")
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
# @app.route('/predict',methods=['POST'])
# def predict():
#
#     float_features = [float(x) for x in request.form.values()]
#     float_features -= mean
#     float_features /= std
#     final_features = np.array(float_features).reshape(1,-1)
#     prediction = model.predict(final_features).tolist()
#     print(prediction)
#     return render_template('index.html', prediction_text='The House Price Should Be {} USD'.format(round(prediction[0][0] * 1000),2))
#
#
# if __name__ == "__main__":
#     app.run(debug=True)