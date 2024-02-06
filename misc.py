
import jax
import jax.numpy as jnp

def model(params, cases):
    return jnp.array([jnp.dot(params[0:-1], case) + params[-1] for case in cases]) 

def loss_funtion(params, feature_vectors, targets):
    preds = model(params, feature_vectors)
    loss = jnp.mean((preds - targets)**2)
    print(loss.primal)

def update(loss_gradient, params, feature_vectors, targets, learning_rate):
    return params - learning_rate * loss_gradient(params, feature_vectors, targets)

def train(steps, params, feature_vectors, targets, learning_rate):
    loss_gradient =jax.grad(loss_funtion)
    for _ in range(steps):
        params = update(loss_gradient, params, feature_vectors, targets, learning_rate)
    return params

