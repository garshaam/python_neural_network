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

#Create the neural network
nn = MeqNeuralNetwork.NeuralNetwork()
nn.addLayer(2) #Input layer (activation function irrelevant since this should be normalized already)
nn.addLayer(8, "leakyrelu") #Hidden layer. As you can see it uses the leaky relu activation function
nn.addLayer(1, "sigmoid") #Output layer

nn.learningRate = 1
nn.weightDecay = 0
nn.biasDecay = 0
nn.batchSize = 10

testPoints = [] #Keeping same test points throughout
spacing = screen_width / 31
for i in range(1, 31):
    for e in range(1, 31):
        testPoints.append([i / spacing, e / spacing])

#Main loop (to improve and test the network)
running = True
while running:
    for event in pygame.event.get(): #Catches pygame exit event to end program
        if event.type == pygame.QUIT:
            running = False

    #Run 500 points through backpropagation to improve the network for each iteration of the loop
    for i in range(2000):
        inputValue = [random.random(), random.random()] #Creating a two item list (x axis and y axis)
        supervisorAnswer = []
        x = inputValue[0] - 0.5
        y = inputValue[1] - 0.5
        if (((10*x**2 + 9*y**2 - 1)**3) - 10*x**2*9*y**3 < 0): #The function that classifies points. A heart in this case.
            supervisorAnswer.append(1)
        else:
            supervisorAnswer.append(0)

        nn.backPropagation(nn.feedForward(inputValue, supervisorAnswer)) #Adjusting the weights and biases

    #Draw 900 test points to evaluate performance.
    #This is also where one would calculate measures such as accuracy, precision, and recall.
    screen.fill(background_color)
    for i in range(900):
        inputValue = testPoints[i] #Using testpoints
        outputValue = nn.guess(inputValue)
        point_color = (255*(outputValue[0]), 255*(outputValue[0]), 255*(outputValue[0])) #More white = closer to 1
        pygame.draw.circle(screen, point_color, (int(inputValue[0]*screen_width), int((1-inputValue[1])*screen_height)), 5) #This draws the point with origin bottom-left
    pygame.display.flip()

#If loop is exited then halt program
pygame.quit()