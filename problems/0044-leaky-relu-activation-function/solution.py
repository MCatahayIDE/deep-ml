def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
    relu_out = 0
	
	# Relu "passes" input through if larger than 0
    if (z > 0):
        relu_out = z
    else:                   # Multiply it by the alpha parameter if < 0; keep neuron active
        relu_out = z * alpha
	# Return
    return relu_out
