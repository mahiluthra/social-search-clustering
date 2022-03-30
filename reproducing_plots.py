# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:27:38 2022

@author: mahi
"""
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

"""
code for reproducing plots from Luthra, Todd 2022 ALIFE paper (currently under review)
"""

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# experiment 1; figure 4

files = glob.glob("final_data/experiment1/*.csv")

columns = pd.read_csv(files[0]).columns
data = pd.DataFrame(columns = columns)

for f in files:
    d = pd.read_csv(f)
    data = data.append(d, ignore_index = True)

patches = np.unique(np.array(data.patch))
var = "social"

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3.5))
c=1
filt_dat = data[data["combination"]==c]
filt_dat = filt_dat[filt_dat["steps"]>2800]
plot_data = []
for p in patches:
    new = np.array(filt_dat[var][filt_dat.patch==p])
    plot_data.append(new)
plot_data = plot_data[::-1]
parts = ax1.violinplot(plot_data, np.arange(len(patches)), showmeans=True)
for pc in parts['bodies']:
    pc.set_facecolor(colors[0])
    pc.set_alpha(0.5)
for partname in ('cbars','cmins','cmaxes','cmeans'):
    vp = parts[partname]
    vp.set_edgecolor("black")
    vp.set_linewidth(1)
ax1.set_xlabel("Clustering Level", size = 12)
ax1.set_ylabel("Social Search", size = 12)
ax1.set_ylim(-1, 1)
ax1.set_title("A. Normal Perception")
ax1.set_xticks(np.arange(0, 6))
ax1.set_xticklabels(np.arange(1, 7))
ax1.tick_params(right=True, labelright=False)

c=3
filt_dat = data[data["combination"]==c]
filt_dat = filt_dat[filt_dat["steps"]>2800]
plot_data = []
for p in patches:
    new = np.array(filt_dat[var][filt_dat.patch==p])
    plot_data.append(new)
plot_data = plot_data[::-1]
parts = ax2.violinplot(plot_data, np.arange(len(patches)), showmeans=True)
for pc in parts['bodies']:
    pc.set_facecolor(colors[1])
    pc.set_alpha(0.5)
for partname in ('cbars','cmins','cmaxes','cmeans'):
    vp = parts[partname]
    vp.set_edgecolor("black")
    vp.set_linewidth(1)
ax2.set_xlabel("Clustering Level", size = 12)
ax2.set_ylim(-1, 1)
ax2.set_title("B. Enhanced Perception")
ax2.set_xticks(np.arange(0, 6))
ax2.set_xticklabels(np.arange(1, 7))
ax2.tick_params(right=True, labelright=False)

c=2
filt_dat = data[data["combination"]==c]
filt_dat = filt_dat[filt_dat["steps"]>2800]

plot_data = []
for p in patches:
    new = np.array(filt_dat[var][filt_dat.patch==p])
    plot_data.append(new)
plot_data = plot_data[::-1]
parts = ax3.violinplot(plot_data, np.arange(len(patches)), showmeans=True)
for pc in parts['bodies']:
    pc.set_facecolor(colors[2])
    pc.set_alpha(0.5)
for partname in ('cbars','cmins','cmaxes','cmeans'):
    vp = parts[partname]
    vp.set_edgecolor("black")
    vp.set_linewidth(1)
ax3.set_xlabel("Clustering Level", size = 12)
ax3.set_ylim(-1, 1)
ax3.set_title("C. Normal Perception for Resources;\nEnhanced Perception for Consumers")
ax3.set_xticks(np.arange(0, 6))
ax3.tick_params(right=True, labelright=False)
ax3.set_xticklabels(np.arange(1, 7))

# experiment 2; figure 5

files = glob.glob("final_data/experiment2/*.csv")

columns = pd.read_csv(files[0]).columns
data = pd.DataFrame(columns = columns)

for f in files:
    d = pd.read_csv(f)
    data = data.append(d, ignore_index = True)

social = np.unique(np.array(data.social))
var = "neigh10"  

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
filt_dat = data[data["steps"]>2800]
plot_data = []
for s in social:
    new = np.array(filt_dat[var][filt_dat.social==s])
    plot_data.append(1-new)
parts = ax1.violinplot(plot_data, np.arange(len(social)), showmeans=True)
for pc in parts['bodies']:
    pc.set_facecolor(colors[3])
    pc.set_alpha(0.4)
for partname in ('cbars','cmins','cmaxes','cmeans'):
    vp = parts[partname]
    vp.set_edgecolor("black")
    vp.set_linewidth(1)
ax1.set_xlabel("Social Search", size = 12)
ax1.set_ylabel("Resource Clustering", size = 12)
ax1.set_ylim(0, 0.7)
ax1.set_xticks(np.arange(0, 9))
ax1.set_xticklabels(np.arange(-1, 1.1, 0.25))
ax1.tick_params(right=True, labelright=False)

# experiment 3; figure 8

files = glob.glob("final_data/experiment3/*.csv")

columns = pd.read_csv(files[0]).columns
data = pd.DataFrame(columns = columns)

for f in files:
    d = pd.read_csv(f)
    data = data.append(d, ignore_index = True)
    
def moving_average(var_data, window_size):
    i = 0
    moving_averages = []
    while i < len(var_data) - window_size + 1:
        window_average = np.mean(var_data[i : i + window_size])
        moving_averages.append(window_average)
        i += 1
    return moving_averages
    
var1 = "social"
var2 = "neigh10"
start_cogs = np.unique(data.start_cog)

social_data =np.zeros((20, 2991))
patch_data =np.zeros((20, 2991))
for r in range(20):
    filt_dat = data[(data["round"]==(r+1))]
    social = moving_average(filt_dat[var1], 10)
    patch = moving_average(1-filt_dat[var2], 10)    
    social_data[r, :]+=social
    patch_data[r, :]+=patch

social_mean = np.mean(social_data, 0)
social_error = np.std(social_data, 0)/(20**0.5)

patch_mean = np.mean(patch_data, 0)
patch_error = np.std(patch_data, 0)/(20**0.5)

social_mean = social_mean[100:]
social_error = social_error[100:]
patch_mean = patch_mean[100:]
patch_error = patch_error[100:]

plt.figure(figsize = (5,4))
plt.plot(np.arange(105, 2996), social_mean, color = '#1f77b4')
plt.fill_between(np.arange(105, 2996), social_mean-social_error, social_mean+social_error, alpha = 0.4)
plt.plot(np.arange(105, 2996), patch_mean, color = 'crimson')
plt.fill_between(np.arange(105, 2996), patch_mean-patch_error, patch_mean+patch_error, alpha = 0.4, color = "crimson")
plt.ylim(-0.1, 0.55)
plt.xlim(0, 3000)
plt.xlabel("Timesteps", size = 12)
plt.ylabel("Parameter Values", size = 12)
plt.legend(["Social Search", "Resource Clustering"])
plt.tick_params(right=True, labelright=False)
