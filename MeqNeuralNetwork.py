import random #Random numbers for the network to start with
import copy #Allows for deep copying. Technically this could be done without by initializing all lists at the same time.

#Adjust backpropagation algorithm
#Allow batch and mini-batch gradient descent
#Allowing saving and loading nodes/weights/biases


class NeuralNetwork:
    '''Feed forward neural network with adjustable hyperparameters that can be used for a user-defined amount
    of inputs; hidden nodes and layers; and outputs. Created by: Adam M Garsha June 2020'''

    def __init__(self):
        #Hyperparameters
        self.learningRate = 1 #If one wants to reduce learning rate over time (e.g. momentum), that is done outside the class currently
        self.weightDecay = 0 #L2 Regularization.
        self.biasDecay = 0 #L2 Regularization. This is not usually applied but the option is open
        self.batchSize = 1 #Number of gradients to find before they are applied
        
        #Measurements
        self.dataCount = 0

        #Network Architecture
        #A list of layers, each list having a list of node values
        #The first layer is the input, and the last is the output
        self.networkNodes = []

        #A list of layers, each list having a list of nodes, which has a list of weights for that node
        #Weights are currently initialized randomly between 0 and 1, then divided by the number of weights going to the same node
        # in order to prevent saturation.
        self.networkWeights = []

        #List of layers, each list having a list of biases (one for each node)
        #There are no biases given to the input
        self.networkBiases = []

        #List of activation functions, one for each layer
        self.networkActivations = []

        self.savedWeightGradients = []
        self.savedBiasGradients = []

    #Special random that initializes values from -0.5 to 0.5
    def sRand():
        '''Returns a random number from -0.5 to 0.5'''
        return random.random() - 0.5

    def Linear(num):
        '''Returns num'''
        return num

    def dLinear():
        '''Derivative of a standard linear function y=x is 1'''
        return 1

    def Relu(num):
        '''Returns the RELU of a number'''
        return max(num, 0)

    def dRelu(relu):
        '''Returns the derivative if a relu value is inputted for relu'''
        if (relu == 0):
            return 0
        else:
            return 1

    def Sigmoid(num): 
        '''Returns the sigmoid of the num input'''
        num = max(-600,num)
        return (1 / (1 + (2.71828 ** -num)))

    def dSigmoid(sig): 
        '''Returns the derivative if a sigmoid value is inputted for sig'''
        return (sig)*(1-sig)

    def Relu(num): 
        '''Returns the RELU of the num input'''
        num = max(0,num)
        return num

    def dRelu(relu): 
        '''Returns the derivative if a RELU value is inputted for relu'''
        if(relu > 0):
            return 1
        else:
            return 0

    def Leakyrelu(num): 
        '''Returns the LeakyRELU of the num input'''
        if (num < 0):
            num = num / 10
        return num

    def dLeakyrelu(relu): 
        '''Returns the derivative if a LeakyRELU value is inputted for relu'''
        if(relu > 0):
            return 1
        else:
            return 0.1

    def addLayer(self, nodes, activation="sigmoid"):
        '''Adds a layer with the specified number of nodes. This is used for input, hidden, and output layers.
        Each new layer adds a list of nodes. Non-input layers add a list of biases, as well as a list of lists
        of weights (one list of weights for each node the weights are emanating from). The activation variable
        specifies which activation function to use: "linear" or "sigmoid"'''

        if(activation.lower() == "sigmoid" or activation.lower() == "linear" or activation.lower() == "relu" or activation.lower() == "leakyrelu"): #Approved Functions
            self.networkActivations.append(activation.lower())
        else:
            self.networkActivations.append("sigmoid") #Default function
        
        self.networkNodes.append([]) #Adding a new node layer
        for i in range(nodes): #For each new node
            self.networkNodes[-1].append(0) #Adding the node to the new layer

        if (len(self.networkNodes) > 1): #If this is not the first layer
            self.networkWeights.append([])
            self.savedWeightGradients.append([])
            for i in range(len(self.networkNodes[-2])): #For each node in previous layer
                self.networkWeights[-1].append([])
                self.savedWeightGradients[-1].append([])
                for e in range(nodes): #For each node in this layer
                    #Random, divided by number of weights going to same node to prevent saturation
                    self.networkWeights[-1][i].append(NeuralNetwork.sRand() / len(self.networkNodes[-2]))
                    self.savedWeightGradients[-1][i].append(0)
            self.networkBiases.append([])
            self.savedBiasGradients.append([])
            for i in range(nodes): #For each new node, add a bias
                self.networkBiases[-1].append(0)
                self.savedBiasGradients[-1].append(0)

    def feedForward(self, input, supervisorAnswers):
        '''Returns the output error for the list input against the correct supervisorAnswers list.
        It is recommended to normalize data before entering.'''

        #Clear all nodes
        for i in range(len(self.networkNodes)):
            for e in range(len(self.networkNodes[i])):
                self.networkNodes[i][e] = 0

        self.networkNodes[0] = input.copy()
        #What the following loop does is give every non-input node a value
        #based on the weights and values of the previous layer
        for i in range(1, len(self.networkNodes)): #For each layer except the first
            for e in range(0, len(self.networkNodes[i])): #For each node in that layer
                for f in range(0, len(self.networkNodes[i - 1])): #For each node in the previous layer
                    self.networkNodes[i][e] += self.networkNodes[i - 1][f] * self.networkWeights[i - 1][f][e]
                self.networkNodes[i][e] += self.networkBiases[i - 1][e]
                if (self.networkActivations[i] == "sigmoid"):
                    self.networkNodes[i][e] = NeuralNetwork.Sigmoid(self.networkNodes[i][e])
                elif (self.networkActivations[i] == "linear"):
                    self.networkNodes[i][e] = NeuralNetwork.Linear(self.networkNodes[i][e])
                elif (self.networkActivations[i] == "relu"):
                    self.networkNodes[i][e] = NeuralNetwork.Relu(self.networkNodes[i][e])
                elif (self.networkActivations[i] == "leakyrelu"):
                    self.networkNodes[i][e] = NeuralNetwork.Leakyrelu(self.networkNodes[i][e])

        #Calculating error from the supervisorAnswers list and depositing it in an error list
        error = []
        for e in range(0, len(self.networkNodes[-1])): #For each node in the last layer 
            error.append((self.networkNodes[-1][e] - supervisorAnswers[e]))

        return error

    def guess(self, input):
        '''Returns the output guess for the list input. It is recommended to normalize data before entering.'''

        #Clear all nodes
        for i in range(len(self.networkNodes)):
            for e in range(len(self.networkNodes[i])):
                self.networkNodes[i][e] = 0

        self.networkNodes[0] = input.copy()
        #What the following loop does is give every non-input node a value
        # based on the weights and values of the previous layer
        for i in range(1, len(self.networkNodes)): #For each layer except the first
            for e in range(0, len(self.networkNodes[i])): #For each node in that layer
                for f in range(0, len(self.networkNodes[i - 1])): #For each node in the previous layer
                    self.networkNodes[i][e] += self.networkNodes[i - 1][f] * self.networkWeights[i - 1][f][e]
                self.networkNodes[i][e] += self.networkBiases[i - 1][e]
                if (self.networkActivations[i] == "sigmoid"):
                    self.networkNodes[i][e] = NeuralNetwork.Sigmoid(self.networkNodes[i][e])
                elif (self.networkActivations[i] == "linear"):
                    self.networkNodes[i][e] = NeuralNetwork.Linear(self.networkNodes[i][e])
                elif (self.networkActivations[i] == "relu"):
                    self.networkNodes[i][e] = NeuralNetwork.Relu(self.networkNodes[i][e])
                elif (self.networkActivations[i] == "leakyrelu"):
                    self.networkNodes[i][e] = NeuralNetwork.Leakyrelu(self.networkNodes[i][e])

        return self.networkNodes[-1]

    def backPropagation(self, LastLayerError):
        '''This is a back-propagation algorithm that determines the gradients for each weight.
        Once the pieces of data have accumulated one batchSize, then the gradient is applied.
        If batchSize is 1, then the data is applied stochastically'''

        self.dataCount += 1

        #Error for the whole network, separated into layers then nodes, just like the networkNodes list
        self.networkError = copy.deepcopy(self.networkNodes) #This is just to get the same structure as the nodes
        #Deep copy keeps the list of list structure

        #Backpropagation resources
        #https://www.youtube.com/watch?v=p1-FiWjThs8 <-Literal numbers/examples
        #http://neuralnetworksanddeeplearning.com/chap2.html#eqtnBP1 <- Theory
        #https://youtu.be/tIeHLnjs5U8?t=569 <- Theory

        #Technically the networkError list includes the inputs but this data is unused
        for i in range(len(LastLayerError)): #For each output error/value
            if (self.networkActivations[-1] == "sigmoid"):
                self.networkError[-1][i] = -LastLayerError[i] * NeuralNetwork.dSigmoid(self.networkNodes[-1][i]) 
            elif (self.networkActivations[-1] == "linear"):
                self.networkError[-1][i] = -LastLayerError[i] * NeuralNetwork.dLinear() 
            elif (self.networkActivations[-1] == "relu"):
                self.networkError[-1][i] = -LastLayerError[i] * NeuralNetwork.dRelu(self.networkNodes[-1][i])
            elif (self.networkActivations[-1] == "leakyrelu"):
                self.networkError[-1][i] = -LastLayerError[i] * NeuralNetwork.dLeakyrelu(self.networkNodes[-1][i]) 

        if (len(self.networkNodes) > 2):
            #Setting all of the node deltas for the hidden layers
            for i in reversed(range(1, len(self.networkNodes)-1)): #For each layer except the first or last starting backwards
                for e in range(0, len(self.networkNodes[i])): #For each node in that layer
                    sumOfWeightsDeltas = 0;
                    for f in range(0, len(self.networkWeights[i][e])): #For each weight of that node (nth weight = nth node on next layer)
                        sumOfWeightsDeltas += self.networkWeights[i][e][f] * self.networkError[i + 1][f] #Add the subsequent node's error delta * the weight to it
                        #The error for a node is based on how connected it is to nodes with errors
                    if (self.networkActivations[i] == "sigmoid"):
                        self.networkError[i][e] = NeuralNetwork.dSigmoid(self.networkNodes[i][e]) * sumOfWeightsDeltas
                    elif (self.networkActivations[i] == "linear"):
                        self.networkError[i][e] = NeuralNetwork.dLinear() * sumOfWeightsDeltas
                    elif (self.networkActivations[i] == "relu"):
                        self.networkError[i][e] = NeuralNetwork.dRelu(self.networkNodes[i][e]) * sumOfWeightsDeltas
                    elif (self.networkActivations[i] == "leakyrelu"):
                        self.networkError[i][e] = NeuralNetwork.dLeakyrelu(self.networkNodes[i][e]) * sumOfWeightsDeltas

        #Determining gradients for weights
        for i in range(len(self.networkWeights)): #For each layer of weights
            for e in range(len(self.networkWeights[i])): #For each node that a weight starts at
                for f in range(len(self.networkWeights[i][e])): #For each weight from that node
                    if (self.batchSize == 1): #If stochastic, do right away
                        partialGradient = self.networkError[i + 1][f] * self.networkNodes[i][e]
                        self.networkWeights[i][e][f] -= -self.learningRate * partialGradient #Reducing cost
                        self.networkWeights[i][e][f] -= self.learningRate * self.weightDecay * self.networkWeights[i][e][f] #Decaying weights based on weightDecay term
                    else:
                        self.savedWeightGradients[i][e][f] += (self.networkError[i + 1][f] * self.networkNodes[i][e]) / self.batchSize
                        
                    
        #Determining gradients for biases
        for i in range(len(self.networkBiases)): #For each layer of biases
            for e in range(len(self.networkBiases[i])): #For each bias/node in that layer
                if (self.batchSize == 1): #If stochastic, do right away
                    partialGradient = self.networkError[i + 1][f]
                    self.networkBiases[i][e] -= -self.learningRate * partialGradient #Reducing cost
                    self.networkBiases[i][e] -= self.learningRate * self.biasDecay * self.networkBiases[i][e] #Decaying biases based on biasDecay term
                else:
                    self.savedBiasGradients[i][e] += self.networkError[i + 1][e] / self.batchSize
                
        if (self.batchSize != 1 and self.dataCount % self.batchSize == 0):
            NeuralNetwork.applyGradients(self)

    def applyGradients(self):
        '''Applies the saved weight and bias gradients. Also implements L2 regularization using weight/bias decay.'''

        #Adjusting all of the weights based on errors
        for i in range(len(self.networkWeights)): #For each layer of weights
            for e in range(len(self.networkWeights[i])): #For each node that a weight starts at
                for f in range(len(self.networkWeights[i][e])): #For each weight from that node
                    partialGradient = self.savedWeightGradients[i][e][f]
                    self.networkWeights[i][e][f] -= -self.learningRate * partialGradient #Reducing cost
                    self.networkWeights[i][e][f] -= self.learningRate * self.weightDecay * self.networkWeights[i][e][f] #Decaying weights based on weightDecay term
                    self.savedWeightGradients[i][e][f] = 0 #Resetting saved weight
    
        #Adjusting all of the biases based on errors
        for i in range(len(self.networkBiases)): #For each layer of biases
            for e in range(len(self.networkBiases[i])): #For each bias/node in that layer
                partialGradient = self.savedBiasGradients[i][e]
                self.networkBiases[i][e] -= -self.learningRate * partialGradient #Reducing cost
                self.networkBiases[i][e] -= self.learningRate * self.biasDecay * self.networkBiases[i][e] #Decaying biases based on biasDecay term
                self.savedBiasGradients[i][e] = 0 #Resetting saved bias
