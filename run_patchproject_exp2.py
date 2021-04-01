# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:24:33 2019

@author: mahi
"""

import model_social as model
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import time
from matplotlib import colors

start = time.time()

if __name__ == "__main__":
    name = sys.argv[1]
    cognition = int(sys.argv[2])
    patch = float(sys.argv[3])
    exploration = float(sys.argv[4])
    social = float(sys.argv[5])
    spread = float(sys.argv[6])
    circum = int(sys.argv[7])
    round = sys.argv[8]

name += str(social) + "_exp" + str(exploration) + "_" + round 
print(name)

print( cognition, patch, exploration, social, spread, circum)

if circum==0:
    circum=False
else:
    circum=True

# cognition, deplete, exploration, social, spread, circum
#sim = model.model(300, cognition, patch, exploration, social, spread, circum) # create world
sim = model.model(300, cognition, patch, exploration, social, spread, True) # create world

for i in range(1000): # run simulations
    sim.step()

# data from simulation
model_hist = sim.history
model_hist.to_csv(r"data/social/" + name + "_hist.csv", sep = ",", header= "True")

end = time.time()
elapsed = end - start
print(elapsed)