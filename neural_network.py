import numpy as np
import jax
import jax.numpy as jnp

def gen_jaxnet_params(layers=[3, 10, 1]):
    params = []
    for sender, receiver in zip(layers[:-1], layers[1:]):
        weights = np.random.uniform(-0.1, 0.1, (sender, receiver))
        biases = np.random.uniform(-0.1, 0.1, (receiver,))
        params.append([weights, biases])
    return params

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def relu(x):
    return jnp.maximum(0, x)

def predict(params, features):
    activations = features
    for weights, biases in params:
        activations = sigmoid(jnp.dot(activations, weights) + biases)
    return activations

def jaxnet_loss(params, features, targets):
    predictions = predict(params, features)
    return jnp.mean(jnp.square(targets - predictions))

def jaxnet_train_one_epoch(params, features, targets, lrate=0.1):
    mse, gradients = jax.value_and_grad(jaxnet_loss)(params, features, targets)
    new_params = [(w - lrate * dw, b - lrate * db) for (w, b), (dw, db) in zip(params, gradients)]
    return new_params, mse
