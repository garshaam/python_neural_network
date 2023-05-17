# python_neural_network
Basic neural network completely from scratch in python (even no numpy). Not useful for real-world applications but taught me a ton about neural network architecture.

## **Import Network**
	import MeqNeuralNetwork

## **Create network class**
	nn = MeqNeuralNetwork.NeuralNetwork()

## **Adjust learning rate**
	nn.learningRate = 0.1

	#The learning rate is multiplied to the backpropagation algorithm. It affects gradient descent and weight/bias decay.

## **Adjust weight decay**
	nn.weightDecay = 0.1

	#Setting a weight decay of 0.1 for example will subtract 10% of the weight from itself. This in effect introduces an 
	
	#incentive for smaller weights.

## **Adjust bias decay**
	nn.biasDecay =0.1

	#Similar to weight decay

## **Adjust batch size**
	nn.batchSize = 1

	#How many gradients will be calculated before any are applied.

## **Add layer**
	nn.addLayer(nodes, activation)

	#nodes = how many nodes to put in the layer

	#activation = which function to apply to the weighted sum. Default is "sigmoid" but "linear" is also supported. The 

	#function for the input layer is irrelevant and has no effect.

## **Feed forward**
	nn.feedForward(inputValue, supervisorAnswer)

	#inputValue = list of inputs

	#supervisorAnswer = list of correct outputs

	#Returns the network's error

## **Backpropagation**
	nn.backPropagation(error)

	#error = error from a feedforward attempt.

	#Most commonly used as: nn.backPropagation(nn.feedForward(inputValue, supervisorAnswer))

## **Guess**
	nn.guess(inputValue)

	#inputValue = list of inputs

	#Returns the network's answer
