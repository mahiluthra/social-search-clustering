# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:24:33 2019

@author: mahi
"""

import model_patch as model
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
    circum = float(sys.argv[7])
    round = sys.argv[8]

name += str(patch) + "_sp" + str(spread) + "_" + round
print(name)

print( cognition, patch, exploration, social, spread, circum)

# cognition, deplete, exploration, social, spread, circum

cognition = spread
spread = 1.0

sim = model.model(300, cognition, patch, exploration, social, spread, circum) # create world

for i in range(1000): # run simulations
    sim.step()
    # save animation plots

# data from simulation
model_hist = sim.history
model_hist.to_csv(r"data/" + name + "_hist.csv", sep = ",", header= "True")

end = time.time()
elapsed = end - start
print(elapsed)