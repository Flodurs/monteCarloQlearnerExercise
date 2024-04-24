import os
import torch
from torch import nn
import torch.optim as optim
import numpy as np
from enum import Enum
import random
import os
import time



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.inputNum = 8
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
  
        logits = self.linear_relu_stack(x)
        return logits
        
        

    
    
        

model = NeuralNetwork().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


class move(Enum):
    up = 0
    down = 1
    left = 2
    right = 3
    
mapSize = 7

state = np.zeros((mapSize,mapSize))
state[0,0] = 1

currentPos = np.array([0,0])



posVisited = []
actionsPerformed = []
step = 0

epsilon = 0.5

localstep = 0

def printState():
    for x in range(mapSize):
        for y in range(mapSize):
            if state[x][y] == 0:
                print("_ ",end="")
            if state[x][y] == 1:
                print("I ",end="")
            if state[x][y] == 2:
                print("X ",end="")
            if state[x][y] == 3:
                print("O ",end="")
        print("")

def setState():
    global state 
    state = np.zeros((mapSize,mapSize))
    # state[4][0] = 2
    # state[4][1] = 2
    # state[4][2] = 2
    # state[4][3] = 2
    state[mapSize-1][0] = 3


setState()

while True:
    step+=1
    localstep+=1
    #perform action and get reward
    if random.uniform(0,1) > epsilon:
        action = random.randrange(0,4)
    else:
        
        potMoves = [model(torch.FloatTensor([localstep,i/3,1]).to(device))[0].cpu().detach().numpy() for  i in range(4)]
        action = np.argmax(potMoves)
   
    
    actionsPerformed.append(action)
    posVisited.append([currentPos[0],currentPos[1]])
    
    if action == move.up.value and currentPos[0] < mapSize-1:
        setState()
        currentPos[0]+=1
        state[currentPos[0],currentPos[1]] = 1 
    
    if action == move.down.value and currentPos[0] > 0:
        setState()
        currentPos[0]-=1
        state[currentPos[0],currentPos[1]] = 1
        
    if action == move.left.value and currentPos[1] > 0:
        setState()
        currentPos[1]-=1
        state[currentPos[0],currentPos[1]] = 1
        
    if action == move.right.value and currentPos[1] < mapSize-1:
        setState()
        currentPos[1]+=1
        state[currentPos[0],currentPos[1]] = 1
        
    
    #update qNet
    if (currentPos==np.array([mapSize-1,0])).all():
        #print("reached")
        for i in range(10):
            inputs = []
            targetOutputs = []
            for i,ac in enumerate(actionsPerformed):
                inputs.append([i/10,ac/3,1])
                
                
                targetOutputs.append([1/len(actionsPerformed)])
                
                
            inputs = torch.FloatTensor(inputs) 
            targetOutputs = torch.FloatTensor(targetOutputs) 
             
            optimizer.zero_grad()
                
            outputs = model(inputs.to(device)).cpu()
            loss = criterion(outputs, targetOutputs)
            loss.backward()
            optimizer.step()
        
        
        
        
        
        #reset
        localstep = 0
        setState()
        state[0,0] = 1
        currentPos = np.array([0,0])
        posVisited = []
        actionsPerformed = []
        
    if localstep > 10:
        localstep = 0
        setState()
        state[0,0] = 1
        currentPos = np.array([0,0])
        posVisited = []
        actionsPerformed = []
    
    if step > 100000:
        os.system('cls')
        
        epsilon = 1
        printState()
    
        #print(qtable)
        time.sleep(0.3)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    