# import numpy as np
# import jax
# import jax.numpy as jnp
# import matplotlib.pyplot as plt
#
# # Feed forward neural network with 5 inputs, 10 hidden nodes, and 5 outputs with JAX
#
# def gen_jaxnet_params(layers=[5,10,5]):
#     sender = layers[0]
#     params = []
#     for receiver in layers[1:]:
#         weights = np.random.uniform(-.1, .1, (sender, receiver))
#         biases = np.random.uniform(-.1, .1, (1, receiver))
#         sender = receiver
#         params.append([weights, biases])
#     return params
#
# def jaxnet_loss(params, features, targets):
#     batched_predict = jax.vmap(predict, in_axes=(None, 0))
#     predictions = batched_predict(params, features)
#     return jnp.mean(jnp.square(targets - predictions))
#
# def sigmoid(x):
#     return 1 / (1 + jnp.exp(-x))
#
# def predict(all_params, features):
#     activations = features
#     for weights, biases in all_params:
#         activations = sigmoid(jnp.dot(activations, weights) + biases)
#     return activations
#
# def jaxnet_train_one_epoch(params, features, targets, lrate=0.1):
#     mse, gradients = jax.value_and_grad(jaxnet_loss)(params, features, targets)
#     return [(w - lrate * dw, b - lrate * db) for (w, b), (dw, db) in zip(params, gradients)], mse
#
# def jaxnet_train(params, features, targets, epochs, lrate=0.1):
#     curr_params = params
#     mse_history = []
#
#     for _ in range(epochs):
#         curr_params, mse = jaxnet_train_one_epoch(curr_params, features, targets, lrate)
#         mse_history.append(mse)
#
#     return curr_params, mse_history
#
# def generate_data_cases(ncases=1000):
#     features = np.random.uniform(-1, 1, (ncases, 5))
#     targets = np.random.uniform(-1, 1, (ncases, 5))
#     return features, targets
#
# class PlantController:
#     def __init__(self, nn_params):
#         self.nn_params = nn_params
#
#     def update(self, features, targets):
#         input_vector = jnp.array(features)
#         control_signal = predict(self.nn_params, input_vector)
#         error = control_signal - targets
#         return jnp.mean(jnp.square(error))
#
#     def reset(self):
#         pass
#
# def disturbance_generator(x1=-0.01, x2=0.01):
#     return np.random.uniform(x1, x2)
#
# def jaxrun(epochs, ncases, layer_sizes, lrate=0.03):
#     features, targets = generate_data_cases(ncases)
#     params = gen_jaxnet_params(layer_sizes)
#     params, mse_history = jaxnet_train(params, features, targets, epochs, lrate=lrate)
#
#     print("mse_history", mse_history)
#
#     plant_controller = PlantController(params)
#     mse_values = []
#
#     for epoch in range(epochs):
#         mse_history = []
#         plant_controller.
#
#         for _ in range(ncases):
#             error = plant_controller.update(features, targets)
#             print("error", error)
#             mse_history.append(error)
#
#         mse = jnp.mean(jnp.array(mse_history))
#         mse_values.append(mse)
#         print(f'Epoch {epoch + 1}: MSE = {mse}')
#
#     # Visualization of the training process
#     plt.figure(figsize=(10, 5))
#     plt.plot(mse_values, label='MSE', color='blue')
#     plt.xlabel('Epoch')
#     plt.ylabel('Mean Squared Error (MSE)')
#     plt.title('MSE Over Training Epochs')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
# if __name__ == "__main__":
#     epochs = 100
#     ncases = 1000
#     layer_sizes = [5, 10, 5]
#     lrate = 0.03
#     jaxrun(epochs, ncases, layer_sizes, lrate=lrate)
