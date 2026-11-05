import math


"""
Helper sigmoid function
"""
def sigmoid(z: float) -> float:
	#Your code here
	result = 1 / (1 + math.exp(-z))
	return result

	
"""
Method simulates a neuron model designed for binary classification with multidimensional input 
features, using the sigmoid activation function to output probabilities. It also involves 
calculating the mean squared error (MSE) to evaluate prediction accuracy.
"""
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):

    # Derive weighted sums and add parameterized bias; sum(weights * features) + bias
    z = [sum(w * f for w, f in zip(weights, feature)) + bias for feature in features]

    # Apply derived z values to sigmoid, derives probabilities 
    probabilities = [sigmoid(i) for i in z] 

    # Extract MSE using labels to compare against sigmoid values; sum(delta(probability_i, label_i))^2 / n
    mse = sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(labels)

    return probabilities, mse