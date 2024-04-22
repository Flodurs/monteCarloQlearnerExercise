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

epsilon = 0.5

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
        state = np.zeros((mapSize,mapSize))
        currentPos[0]+=1
        state[currentPos[0],currentPos[1]] = 1 
    
    if action == move.down.value and currentPos[0] > 0:
        state = np.zeros((mapSize,mapSize))
        currentPos[0]-=1
        state[currentPos[0],currentPos[1]] = 1
        
    if action == move.left.value and currentPos[1] > 0:
        state = np.zeros((mapSize,mapSize))
        currentPos[1]-=1
        state[currentPos[0],currentPos[1]] = 1
        
    if action == move.right.value and currentPos[1] < mapSize-1:
        state = np.zeros((mapSize,mapSize))
        currentPos[1]+=1
        state[currentPos[0],currentPos[1]] = 1
    
    
        
        
    
    
    
    # print(actionsPerformed)
    # print(posVisited)
    if step > 100000:
        os.system('cls')
        
        epsilon = 1
        print(state)
    
        #print(qtable)
        time.sleep(0.3)
    
    #goal reached
    if (currentPos==np.array([mapSize-1,mapSize-1])).all():
        #update q table
        print("Goooooooooooooooal")
        for i,ac in enumerate(actionsPerformed):
            
            qtable[ac,posVisited[i][0],posVisited[i][1]]=qtable[ac,posVisited[i][0],posVisited[i][1]]+(1/len(actionsPerformed))*((1/len(actionsPerformed))-qtable[ac,posVisited[i][0],posVisited[i][1]])
            
        #reset
        state = np.zeros((mapSize,mapSize))
        state[0,0] = 1
        currentPos = np.array([0,0])
        posVisited = []
        actionsPerformed = []
        
    
    
    
    
    
    
    