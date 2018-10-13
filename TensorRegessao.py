import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.image as mpimg

imagem = mpimg.imread('entrada.jpg')

forma = imagem.shape

X = []
Y = []
for i in range(forma[0]):
    for j in range(forma[1]):
        for k in range(forma[2]):
            X.append([i/forma[0], j/forma[1], k/forma[2]]) 
            Y.append(float(imagem[i][j][k])/255)

X = pd.DataFrame(X, columns = list('ijk'))
Y = pd.DataFrame(Y, columns = list('I'))
train_steps = 100000
batch_size = 2000

my_feature_columns = []
for key in X.keys():
      my_feature_columns.append(tf.feature_column.numeric_column(key=key))

meu_regressor = tf.estimator.DNNRegressor(
    feature_columns=my_feature_columns,
    hidden_units=[50, 50, 50, 50, 50])


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def eval_input_fn(features, batch_size):
    """An input function for evaluation or prediction"""

    inputs = dict(features)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

meu_regressor.train(input_fn=lambda:train_input_fn(X, Y, batch_size), steps=train_steps)

# Evaluate the model.
predito = meu_regressor.predict(input_fn=lambda:eval_input_fn(X, batch_size))

predito_L = []
for pred_dict in predito:
      andre = pred_dict['predictions']
      predito_L.append(andre)


imagem = np.zeros(forma)

u = 0
for i in range(forma[0]):
  for j in range(forma[1]):
      for k in range(forma[2]):
          imagem[i][j][k] = predito_L[u]
          u += 1

imagem = (imagem - np.min(imagem)) / (np.max(imagem) - np.min(imagem))


mpimg.imsave('saida.png', imagem)