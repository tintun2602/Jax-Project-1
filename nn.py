

def run_system(num_epochs):

    gradfunc = jax.value_and_grad(run_one_epoch, argnums=0)
    .. init params and state
    
    for in range(num_epochs):
    avg_error, gradients = gradfunc(params, state)
    execute run_one_epoch via gradfunc
    update_params[params, gradients) # Use gradients to update controller params





def run_one_epoch(params, state):
    
    .. state gets dated at each timestep:
    return avg_of_all_timestep_errors
    
    

    pid = PIDController(kp, ki, kd, set_point)
    bathtub = BathtubPlant(initial_height, A, C, g)
    total_error = 0.0
    
    for t in range(num_timesteps):
        key, subkey = random.split(key)
        D = random.uniform(subkey, (), minval=-0.01, maxval=0.01)  # Random disturbance/noise
        current_height = bathtub.get_state()
        U = pid.update(current_height)
        bathtub.update_state(U, D)
        error = set_point - current_height
        total_error += error**2
    
    mse = total_error / num_timesteps
    return mse