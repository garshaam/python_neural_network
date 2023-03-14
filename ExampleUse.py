import pygame #used for visualization/graph
import random #used to generate random points
import MeqNeuralNetwork #My custom neural network

#Pygame and Colors
pygame.init()

screen_width = 700
screen_height = 700
windowTitle = "Testing Python"
screen = pygame.display.set_mode([screen_width, screen_height])
windowIcon = pygame.image.load(r"C:\Users\garsh\AdamStuff\Python\NeuralNetwork\Images\MeqIcon.png")

pygame.display.set_caption(windowTitle)
pygame.display.set_icon(windowIcon)

background_color = (127.5, 127.5, 127.5)
point_color = (255, 50, 255)

#Create the neural network
nn = MeqNeuralNetwork.NeuralNetwork()
nn.addLayer(1) #Input layer (activation function irrelevant since this should be normalized already)
nn.addLayer(3, "sigmoid") #Hidden layer
nn.addLayer(1, "linear") #Output layer (linear activation function i.e. no change)

nn.learningRate = 0.01
nn.weightDecay = 0
nn.biasDecay = 0
nn.batchSize = 1 #Stochastic gradient descent

#Main loop (to improve and test the network)
running = True
while running:
    for event in pygame.event.get(): #Catches pygame exit event to end program
        if event.type == pygame.QUIT:
            running = False

    #Run 500 points through backpropagation to improve the network for each iteration of the loop
    for i in range(500):
        inputValue = [random.random()] #Creating a one item list (x axis input)
        supervisorAnswer = [inputValue[0] ** 2] #Function is quadratic

        nn.backPropagation(nn.feedForward(inputValue, supervisorAnswer)) #Adjusting the weights and biases

    #Draw 100 test points to evaluate performance.
    #This is also where one would check measures such as accuracy, precision, and recall.
    screen.fill(background_color)
    for i in range(1, 101):
        inputValue = [i/100] #Spacing out the points evenly.
        outputValue = nn.guess(inputValue)
        pygame.draw.circle(screen, point_color, (int(inputValue[0]*screen_width), screen_height - int(outputValue[0]*screen_height)), 5)
    pygame.display.flip()

#If loop is exited then halt program
pygame.quit()




