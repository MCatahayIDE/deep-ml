import math

def softmax(scores: list[float]) -> list[float]:
	# Softmax with numerical stabilization; max value extraction and subtraction
	max_z = max(scores)                              # Extract maximum input value z from list
	shifted_z = [i - max_z for i in scores]          # Apply difference with max to all input values

    # Computation of exponentials, storage in list, with stabilized input values used
	exp_z = [math.exp(i) for i in shifted_z]    # Compute the exponentials of the shifted values
	sum_exp_z = sum(exp_z)                      # Sum up all stabilized exponentials

    # exp(z) / sum(exp(z)) for each given input, z_i, in list 
	return [i / sum_exp_z for i in exp_z]       # Normalize to get the softmax probabilities; DEBUG: CHECK SYNTAX !!!