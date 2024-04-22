import numpy as np
from enum import Enum
import random
import os
import time
 
random.seed()

class move(Enum):
    up = 0
    down = 1
    left = 2
    right = 3
    
mapSize = 10

state = np.zeros((mapSize,mapSize))
state[0,0] = 1

currentPos = np.array([0,0])

qtable = np.zeros((4,mapSize,mapSize))

posVisited = []
actionsPerformed = []
step = 0

epsilon = 0.4

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
    state[4][0] = 2
    state[4][1] = 2
    state[4][2] = 2
    state[4][3] = 2
    state[mapSize-1][0] = 3


setState()

while True:
    step+=1
    #perform action and get reward
    if random.uniform(0,1) > epsilon:
        action = random.randrange(0,4)
    else:
        potMoves = [qtable[i][currentPos[0]][currentPos[1]] for  i in range(4)]
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
    
    
        
        
    
    
    
    # print(actionsPerformed)
    # print(posVisited)
    if step > 500000:
        os.system('cls')
        
        epsilon = 0.9
        printState()
    
        #print(qtable)
        time.sleep(0.3)
    
    #goal reached
    if (currentPos==np.array([mapSize-1,0])).all():
        #update q table
        #print("Goooooooooooooooal")
        for i,ac in enumerate(actionsPerformed):
            
            qtable[ac,posVisited[i][0],posVisited[i][1]]=qtable[ac,posVisited[i][0],posVisited[i][1]]+(1/len(actionsPerformed))*((1/len(actionsPerformed))-qtable[ac,posVisited[i][0],posVisited[i][1]])
            
        #reset
        setState()
        state[0,0] = 1
        currentPos = np.array([0,0])
        posVisited = []
        actionsPerformed = []
    
    #x touched
    if (currentPos==np.array([4,0])).all() or (currentPos==np.array([4,1])).all() or (currentPos==np.array([4,2])).all() or (currentPos==np.array([4,3])).all():
        #update q table
        
        for i,ac in enumerate(actionsPerformed):
            
            qtable[ac,posVisited[i][0],posVisited[i][1]]=qtable[ac,posVisited[i][0],posVisited[i][1]]+(1/len(actionsPerformed))*((0)-qtable[ac,posVisited[i][0],posVisited[i][1]])
            
        #reset
        setState()
        state[0,0] = 1
        currentPos = np.array([0,0])
        posVisited = []
        actionsPerformed = []
        
    
    
    
    
    
    
    