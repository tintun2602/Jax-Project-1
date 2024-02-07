import numpy as np
import jax
import jax.numpy as jnp
from bathtub import bathtub


# Feed forward neural network with 5 inputs, 10 hidden nodes, and 5 outputs with JAX
class NNController:
    def __init__(self, sizes):
        self.layer_sizes = sizes

    def gen_jaxnet_params(self, layers=[3, 10, 1]):
        sender = layers[0]
        params = []
        # one set of weights and biases for each non-input layer
        for receiver in layers[1:]:
            weights = np.random.uniform(-.1, .1, (sender, receiver))  # Using numpy because this is not traced by jax
            biases = np.random.uniform(-.1, .1, (1, receiver))
            sender = receiver
            params.append([weights, biases])
        return params

    def sigmoid(self, x):
        """Activation function"""
        return 1 / (1 + jnp.exp(-x))

    def predict(self, all_params, features):
        # nn_forward
        """
        :param all_params: weights and biases
        :param features: input features, (PID) for the bathtub model
        Returned activations = the net's output"""

        activations = features
        # Feed the features forward through all layers of the net
        for weights, biases in all_params:
            activations = self.sigmoid(jnp.dot(activations, weights) + biases)

        return activations

    def jaxnet_loss(self, params, features, targets):
        # Make a batched version of the `predict` function.
        # None => all params used on each call,
        # 0 => take one row at a time of the cases. vmap = vector map
        batched_predict = jax.vmap(self.predict, in_axes=(None, 0))
        predictions = batched_predict(params, features)
        print(f"Targets: {targets} and predictions: {predictions}")
        return jnp.mean(jnp.square(predictions - targets))

    #def jaxnet_train_one_epoch(self, params, features, targets, lrate=0.1):
    def jaxnet_train_one_epoch(self, params, lrate=0.03):
        """ basically the update findtion i think !
        Mean squared error and gradient function with JAX tracing
        """

        plant = bathtub(A=100, C=0.01, H_0=100)
        num_timesteps = 100
        total_loss = 0.0

        for i in range(num_timesteps):
            noise = disturbance_generator()

            current_state = plant.state
            pid_err = pid_error_controller.calculate_errors(current_state)

            # TODO ta den ut av self kalssen, basically gjør det modulært!!!!!!!!
            control_signal = self.predict(params, pid_err)

            # plant.update_state(control_signal, D)
            plant.update(control_signal, noise)

            error = 100 - current_state
            total_loss += error ** 2

        mse = total_loss / num_timesteps
        return (jnp.squeeze(mse))

    #def jaxnet_train(self, params, features, targets, epochs, lrate=0.1):
    def jaxnet_train(self, params, epochs, lrate=0.1):
        error_history = []
        curr_params = params
        for epoch in range(epochs):
            #curr_params, mse = self.jaxnet_train_one_epoch(curr_params, features, targets, lrate)
            curr_params, mse = self.jaxnet_train_one_epoch(curr_params, lrate)

            error_history.append(mse)

        return curr_params

    def jaxrun(self, epochs, layer_sizes, lrate=0.03):
        # features, targets = generate_data_cases(ncases)
        # features should be PID
        params = self.gen_jaxnet_params(layer_sizes)

        # TODO: gjør setpoint modulært

        targets = 100

        #self.jaxnet_train(params, features, targets, epochs, lrate=lrate)


def disturbance_generator(x1=-0.01, x2=0.01):
    return np.random.uniform(x1, x2)

class PIDErrorCalculator:
    def __init__(self, set_point):
        self.set_point = set_point
        self.integral_error = 0
        self.previous_error = 0

    def reset(self):
        self.integral_error = 0
        self.previous_error = 0

    def calculate_errors(self, current_value):
        error = self.set_point - current_value
        self.integral_error += error
        derivative_error = error - self.previous_error
        self.previous_error = error

        return [error, self.integral_error, derivative_error]


pid_error_controller = PIDErrorCalculator(set_point=100)

if __name__ == "__main__":
    epochs = 100
    lrate = 0.03
    layer_sizes = [3, 10, 1]

    nn = NNController(layer_sizes)
    nn.jaxrun(epochs, layer_sizes, lrate=lrate)
