import pygame
import random
import time

#Pygame and Colors
pygame.init()
screen_width = 800
screen_height = 800
screen = pygame.display.set_mode([screen_width, screen_height])
windowTitle = "Testing Python"
pygame.display.set_caption(windowTitle)
windowIcon = pygame.image.load(r"C:\Users\garsh\AdamStuff\Python\Images\MeqIcon.png")
pygame.display.set_icon(windowIcon)

#Colors courtesy of coolors.co (sorry w3 schools...)
background_color = (127.5, 127.5, 127.5)
axes_color = (0, 50, 0)
text_color = (0, 75, 0)

#Variables and Constants
testPoints = [] #This is a list of coords of the drawn points (testing set)
trainPoints = [] #These are the points used to train the network
loopDelay = 1
learningRate = 1
weightDecay = 0.0000000
biasDecay = 0.0000000
#Learning rate adjustment stuff
pastAccuracy = 0 #This is the accuracy of the iteration epochSinceErrorCalc iterations ago
epochSinceErrorCalc = -1 #Since this is negative the pastError will be calculated instantly
epochsTillUpdate = 10
accuracyWiggleRoom = 0 #How many percentage points the current accuracy has to be below the past in order to trigger the learningRate change

#Special random that initializes values from -0.5 to 0.5
def sRand():
    return random.random() - 0.5

#A list of layers, each list having a list of node values
#The first layer is the input, and the last is the output
#networkNodes = [[0,0],[0, 0],[0]]
networkNodes = []

#A list of layers, each list having a list of nodes, which has a list of weights for that node
#Weights are currently initialized randomly between 0 and 1
#networkWeights = [[[sRand(),sRand()],[sRand(),sRand()]] , [[sRand()],[sRand()]]]
networkWeights = []

#List of layers, each list having a list of biases (one for each node)
#There are no biases given to the input
#networkBiases = [[sRand(), sRand()], [sRand()]]
#networkBiases = [[0, 0], [0]]
networkBiases = []

def addLayer(nodes):
    global networkNodes, networkBiases, networkWeights
    networkNodes.append([]) #Adding a new node layer
    for i in range(nodes):
        networkNodes[-1].append(0) #Adding the nodes to the layer

    if (len(networkNodes) > 1): #If this is not the first layer
        networkWeights.append([])
        for i in range(len(networkNodes[-2])): #For each node in previous layer
            networkWeights[-1].append([])
            for e in range(nodes): #For each node in this layer
                networkWeights[-1][i].append(sRand() / len(networkNodes[-2])) #Divide by number of weights going to same node to prevent saturation
        networkBiases.append([])
        for i in range(nodes):
            networkBiases[-1].append(0)

#Spawns points as a list and adds them to a specified list of points
def spawnPoints(list, count):
    for i in range(count):
        list.append([round(random.random()*screen_width), round(random.random()*screen_height)])

def Sigmoid(num): #returns the sigmoid of a value
    return (1 / (1 + (2.71828 ** -num)))

def dSigmoid(sig): #returns the derivative if a sigmoid is inputted
    return (sig)*(1-sig)

def feedForward(input, supervisorAnswers):
    '''This is a stochastic feed forward function that returns the guess and output error for one piece of data.
    Could be used for batches by averaging many errors. It is recomended to normalize data before entering.'''
    networkNodes[0] = input
    #What the following loop does is give every non-input node a value
    #based on the weights and values of the previous layer
    for i in range(1, len(networkNodes)): #For each layer except the first
        for e in range(0, len(networkNodes[i])): #For each node in that layer
            for f in range(0, len(networkNodes[i - 1])): #For each node in the previous layer
                networkNodes[i][e] += networkNodes[i - 1][f] * networkWeights[i - 1][f][e]
            networkNodes[i][e] += networkBiases[i - 1][e]
            networkNodes[i][e] = Sigmoid(networkNodes[i][e])

    #Calculating error from the supervisorAnswers list and depositing it in an error list
    error = []
    for e in range(0, len(networkNodes[-1])): #For each node in the last layer 
        error.append((networkNodes[-1][e] - supervisorAnswers[e]))

    returnInformation = {
        "error": error,
        "guesses": networkNodes[-1]
    }
    return returnInformation

#This function would almost always be used in conjunction with the feedForward return value
def backPropagation(LastLayerError):
    '''This is a back-propagation algorithm that adjust the weights of the neural network.
    Can be used stochastically or for batches.'''

    #Error for the whole network, separated into layers then nodes, just like the networkNodes list
    networkError = networkNodes.copy() #This is just to get the same structure as the nodes

    #Technically the networkError list includes the inputs but this data is unused
    for i in range(len(LastLayerError)): #For each output error/value
        networkError[-1][i] = -LastLayerError[i] * dSigmoid(networkNodes[-1][i]) #Calculating error delta https://www.youtube.com/watch?v=p1-FiWjThs8

    if (len(networkNodes) > 2):
        #Setting all of the node deltas for the hidden layers
        for i in reversed(range(1, len(networkNodes)-1)): #For each layer except the first or last starting backwards
            for e in range(0, len(networkNodes[i])): #For each node in that layer
                sumOfWeightsDeltas = 0;
                for f in range(0, len(networkWeights[i][e])): #For each weight of that node (nth weight = nth node on next layer)
                    sumOfWeightsDeltas += networkWeights[i][e][f] * networkError[i + 1][f] #Add the subsequent node's error delta * the weight to it
                    #The error for a node is based on how connected it is to nodes with errors
                networkError[i][e] = dSigmoid(networkNodes[i][e]) * sumOfWeightsDeltas


    #Adjusting all of the weights based on errors
    for i in range(len(networkWeights)): #For each layer of weights
        for e in range(len(networkWeights[i])): #For each node that a weight starts at
            for f in range(len(networkWeights[i][e])): #For each weight from that node
                #(change in cost as weight changes) = (previous activation)(dsigmoid of z)(error of node)
                partialGradient = networkError[i + 1][f] * networkNodes[i][e]
                networkWeights[i][e][f] -= -learningRate * partialGradient #Reducing cost
                networkWeights[i][e][f] -= learningRate * weightDecay * networkWeights[i][e][f] #Decaying weights to prevent saturation
    
    #Adjusting all of the biases based on errors
    for i in range(len(networkBiases)): #For each layer of biases
        for e in range(len(networkBiases[i])): #For each bias/node in that layer
                #(change in cost as bias changes) = (dsigmoid of z)(error of node)    
                partialGradient = networkError[i + 1][f]
                networkBiases[i][e] -= -learningRate * partialGradient #Reducing cost
                networkBiases[i][e] -= learningRate * biasDecay * networkBiases[i][e] #Decaying biases to prevent saturation
                #The book I learned about regularization from said not to decay biases over time, but I find that
                # the decay of the weights makes the biases too important.
                
#Beginning of Program
addLayer(2)
addLayer(12)
addLayer(1)

spawnPoints(testPoints, 2000) #These are the points drawn to the screen

def SupervisorAnswers(x, y):
    supervisorAnswers = []
    #Heart equation
    #x = (x - 0.5) * 3
    #y = (y - 0.5) * 3
    #if (((x**2 + y**2 - 1)**3 - x**2*y**3) < 0): #Heart function
    x = x - 0.5 #Puts the origin in the middle
    y = y - 0.5
    if (y>-0.1 and x>-0.1):
        supervisorAnswers.append(1)
    else:
        supervisorAnswers.append(0)
    return supervisorAnswers

drawTests = True

#Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE: #Space toggles the drawing of the test points
                if (drawTests == True):
                    drawTests = False
                else:
                    drawTests = True

    screen.fill(background_color)

    #pygame.draw.line(screen, axes_color, (400, 0), (400, screen_height), 3)
    #pygame.draw.line(screen, axes_color, (0, 400), (screen_width, 400), 3)

    #Each point is a list so [0] is x and [1] is y in terms of drawing
    accuracy = 0
    for point in testPoints:
        x = point[0]/screen_width
        y = point[1]/screen_height
        value = feedForward([x, y], [404])["guesses"][0]
        if (drawTests):
            #pygame.draw.circle(screen, ((value)*255, 0, (1-value)*255), point, 7)
            pygame.draw.circle(screen, ((value)*255, (value)*255, (value)*255), point, 5)
        if (((value > 0.5) and (SupervisorAnswers(x, y)[0] > 0.5)) or ((value < 0.5) and (SupervisorAnswers(x, y)[0] < 0.5))):
            accuracy += 1
    accuracy = (accuracy / len(testPoints))*100

    myfont = pygame.font.SysFont('Comic Sans MS', 30)
    textsurface = myfont.render(str(round(accuracy, 2)), False, text_color)
    screen.blit(textsurface,(0,760))

    epochSinceErrorCalc += 1
    if (epochSinceErrorCalc > epochsTillUpdate or epochSinceErrorCalc <= 0):
        if (pastAccuracy > accuracy + accuracyWiggleRoom):
            learningRate = learningRate * 0.75
            if (learningRate <= 0.0004):
                learningRate = 0
                running = False
        pastAccuracy = accuracy
        epochSinceErrorCalc = 0
        print("Learning Rate: " + str(learningRate))

    spawnPoints(trainPoints, 4000)
    for point in trainPoints:
        x = point[0]/screen_width
        y = point[1]/screen_height
        backPropagation(feedForward([x, y], SupervisorAnswers(x, y))["error"])
    trainPoints.clear()

    #Updates the display
    pygame.display.flip()

pygame.quit()