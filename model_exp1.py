# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 09:37:44 2020

@author: mahi
"""
from mesa import Model, Agent
import random
from mesa.time import RandomActivation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mesa.time import BaseScheduler
import math
import pandas as pd
import sys
from mesa import space
import copy
from scipy.ndimage.filters import gaussian_filter

class A2(Agent):  # second level agent-- predator agent   
    
    """ 
    hyperparameters
    
    eat energy, reproduction energy
    
    step distance, reach distance, reproduction distance
    
    viewing food vs food energy
    
    """
    
    def __init__(self, unique_id, model, start_energy, cognition):
        super().__init__(unique_id, model)  # creates an agent in the world with a unique id
        
        # energy parameters
        self.energy = start_energy
        self.eat_energy = self.model.eat_energy
        self.tire_energy = self.model.tire_energy
        self.reproduction_energy = self.model.reproduction_energy * self.model.anim_reprod
        self.cognition_energy = self.model.cognition_energy
        
        # other initiializations
        self.cognition = cognition
        self.dead = False
        self.identity = 2
        self.age = 0
        
        # movement parameters
        self.velocity = np.random.normal(1, 0.05)
        self.direction = np.random.uniform(-1, 1, 2)
        self.direction /= np.linalg.norm(self.direction)
                
    def step(self): # this iterates at every step
        if not self.dead:  # the agent moves on ev ery step   
            self.move()
        if not self.dead: # have to repeat because they might have died through cognition and move (e.g., combat)
            self.tire_die()
        if not self.dead:
            self.reproduce()
            self.age+=1
            self.model.age.append(self.age)
            self.model.cog0.append(self.cognition[0])
            self.model.cog1.append(self.cognition[1])
            
    def introduce(self, x, y, energy, cog):
        a = A2(self.model.unique_id, self.model, start_energy = energy, cognition = cog)
        self.model.unique_id += 1
        self.model.grid.place_agent(a, (x,y))
        self.model.schedule.add(a)
        
    def kill(self):
        self.dead=True
        x,y = self.pos
        self.model.grid.remove_agent(self) 
        self.model.schedule.remove(self)
        self.model.death += 1
                
    def eat(self, coord, eaten_energy = 0):
        avail_food = [agent for agent in self.model.grid.get_neighbors([coord], radius = 0.5) if agent.identity==1]
        if len(avail_food)==0:
            return
        hungry_energy = self.eat_energy - eaten_energy
        food = random.choice(avail_food)
        deplete = self.model.deplete
        if food.energy > (hungry_energy * deplete):
            food.energy -= hungry_energy * deplete
            self.energy += hungry_energy
            # print("fooood")
            return
        if food.energy == (hungry_energy * deplete):
            self.energy += hungry_energy
            food.dead = True
            self.model.grid.remove_agent(food)
            self.model.schedule.remove(food)   
            #print("fooood")
            return
        if food.energy < (hungry_energy * deplete):
            food_energy = food.energy/deplete
            self.energy += food_energy    
            food.dead = True
            self.model.grid.remove_agent(food)
            self.model.schedule.remove(food)
            self.eat(coord, eaten_energy = eaten_energy+food_energy)
            #print("fooood")

    def reproduce(self): # reproduce function
        coin = random.random()
        if self.energy >= self.reproduction_energy:
            self.model.reprod += 1
            if self.model.disp_rate == 7:
                x = random.random() * self.grid.width
                y = random.random() * self.grid.width
                new_position = (x,y)
            elif self.model.disp_rate == 0:
                new_position = self.model.get_radius_reprod(0.3, self.pos)
            #change to 0.4? 
            energy_own = math.ceil(self.energy/2)
            energy_off = self.energy - energy_own
            self.energy = energy_own
            
            cog = [min(1, max(-1, random.normalvariate(self.cognition[0], 0.025))), min(1, max(0, random.normalvariate(self.cognition[1], 0.1) ))]
            # add mutuation function
                                    
            x,y = new_position                
            self.introduce(x,y, energy_off, cog)
            nA2s = np.argwhere(self.model.grid._agent_ids==2)
            if len(nA2s)>10000:
                nA2_id = np.random.choice(nA2s[:,0], 1)[0]
                killed = self.model.schedule.agents[nA2_id]
                killed.kill()

            
    def tire_die(self): 
        x,y = self.pos
        self.energy-=self.tire_energy # + (self.cognition[0]/10)
        if self.energy<=0:
            self.kill()
        
    def cogdecision(self):
      #  radius = 10
      #  food, locs, dists = self.model.grid.get_neighbors_locs(self.pos, radius)
      #  dists = dists/radius
        if self.model.cognition==1:
            food_ls = self.model.food_ls
            comp_ls = self.model.comp_ls
            
            weight = self.cognition[0]
            if self.model.social != 100.:
                weight = self.model.social
                
            exploration = self.cognition[1] * 80
            if self.model.exploration != 100:
                exploration = self.model.exploration
        
            total_ls = food_ls + weight*comp_ls
            
            if self.model.circum == True:
                cir = np.array(self.model.points_on_circumference(self.pos, self.velocity))%100  # %100 to accomodate for toroid
                in_ = np.round(cir*4).astype(int)%400  # attempts to find circum position on the grid
            else:
                curr = np.around(np.array(self.pos)*4)
                in_ = []
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        in_.append([curr[0]+i, curr[1]+j])
                
                for i in [-1, 0, 1]:
                    for j in [-2, 2]:
                        in_.append([curr[0]+i, curr[1]+j])
                        in_.append([curr[0]+j, curr[1]+i])
                in_ = np.array(in_)
                in_ = in_%400
                cir = in_/4
                in_ = in_.astype(int)
                    
            out = total_ls[in_[:,0], in_[:,1] ]
            weighted_out = exploration*out
            wtexp = np.exp(weighted_out)
            
            inf_check = np.argwhere(np.isinf(wtexp))
            if len(inf_check)==1:
                idx = int(inf_check[0])
                print(idx)
                return(cir[idx])
            if len(inf_check)>1:
                wtexp = weighted_out

            wtfinal = wtexp/np.sum(wtexp)
            index = np.arange(len(wtfinal))
            move_index = random.choices( index, k=1, weights = wtfinal )[0]
            move = cir[move_index]
            while len([agent for agent in self.model.grid.get_neighbors(move, radius = 0.4) if agent.identity==2])>0:
                if len(index) == 1:
                    return self.pos
                wtfinal = np.delete(wtfinal, move_index)
                cir = np.delete(cir, move_index, axis = 0)
                index = np.arange(len(wtfinal))
                move_index = random.choices( index, k=1, weights = wtfinal )[0]
                move = cir[move_index]
            return(tuple(move))
            
        if self.model.cognition==0:
            self.direction = np.random.uniform(-1, 1, 2)
            self.direction /= np.linalg.norm(self.direction)
            move = self.model.grid.torus_adj(np.array(self.pos) + self.direction * self.velocity)
            return(move)
    
    def move(self):  
        self.energy-=self.cognition_energy  
        newx, newy = self.cogdecision()
        x,y = self.pos
        self.model.grid.move_agent(self, (newx, newy) )
        self.eat((newx, newy)) 
    
class A1(Agent):
    
    """ plants agent functions
    """
    
    def __init__(self, unique_id, model, start_energy, og_loc):
        super().__init__(unique_id, model)
        
        self.energy = start_energy # agent starts at energy level 10
        self.eat_energy =  self.model.eat_energy
        self.tire_energy = self.model.tire_energy
        self.reproduction_energy = self.model.reproduction_energy
        self.dead = False
        self.identity = 1
        self.og_loc = og_loc
        
    def step(self): # this iterates at every step
        self.eat()
        self.tire_die()
        if not self.dead:
            self.reproduce() 
            
    def reproduce(self):
        if self.energy >= self.reproduction_energy and self.model.og_locs[self.og_loc]=="alive":
            if self.model.disp_rate == 1:
                x = random.random() * self.grid.width
                y = random.random() * self.grid.width
                new_position = (x,y)
            elif self.model.disp_rate == 0:
                new_position = self.model.get_radius_reprod(self.model.patchiness, self.og_loc)
            
            self.energy -= 10
            energy_own = math.ceil(self.energy/2)
            energy_off = self.energy - energy_own
            self.energy = energy_own
            
            x,y = new_position
            if np.size(self.model.grid.get_neighbors([(x, y)], radius = self.model.crowding))==0:                
                a = A1(self.model.unique_id, self.model, energy_off, self.og_loc)
                self.model.unique_id += 1
                self.model.grid.place_agent(a, new_position)
                self.model.schedule.add(a)
            
    def eat(self): # agent eats at every step and thus depeletes resources          
        self.energy += self.eat_energy # nutrition is added to agent's nutrition
            
    def tire_die(self): # agent loses energy at every step. if it fails to eat regularly, it dies due to energy loss
        x,y = self.pos
        self.energy-=self.tire_energy
        if self.energy<=0:
            self.dead=True
            self.model.grid.remove_agent(self) 
            self.model.schedule.remove(self)
            
class model(Model):
    
    def __init__(self, introduce_time, cognition, deplete, exploration, social, spread, circum):
        
        # initializations of 
        self.start_energy = 10
        self.eat_energy = 5
        self.tire_energy = 3
        self.reproduction_energy = 20
        self.cognition_energy = 1
        (self.a1num, self.a2num) = (20, 50)
        self.skip_300 = True
        self.intro_time = 40
        self.schedule = RandomActivation(self) # agents take a step in random order 
        self.cognition = 1
        self.deplete = 1
        self.exploration = exploration
        self.social = social
        self.spread = spread
        self.circum = True
        self.og_locs = {}
        self.patchiness = deplete
        self.anim_spread=circum
        if cognition==1.0:
            self.anim_reprod = 2
            self.crowding = 0.3
        elif cognition==2.0:
            self.anim_reprod = 1
            self.crowding = 0.2
        if cognition==3.0:
            self.anim_reprod = 2
            self.crowding = 0.2
        
        self.grid = space.ContinuousSpace(100, 100, True) # the world is a grid with specified height and width
        self.disp_rate = 0
        # data storage initialization
        self.age = []
        self.cog0 = []
        self.cog1 = []
        (self.nstep, self.unique_id, self.reprod, self.food, self.death, self.combat) = (0, ) * 6
        self.history = pd.DataFrame(columns = ["nA1", "nA2", "age", "cognition0","cognition1","cognition0sd", "cognition1sd", "neigh5", "neigh10", "neighanim5","neighanim10","reprod", "food", "death", "combat"])
        
        # initializations for calculating patchiness of world
        self.distances1 = np.array([])
        self.distances2 = np.array([])
        self.expect_NN = []
        self.neigh = [5, 10]
        for i in self.neigh:
            self.expect_NN.append((math.factorial(2*i) * i)/(2**i * math.factorial(i))**2)

        # initialize resource agent
        for i in range(self.a1num):
            self.introduce_agents("A1")
    
    def let_plants_grow_geez(self, loc, size=200):
        size = size
        for s in range(size):
            new_position = self.get_radius_reprod(self.patchiness, loc)
            a = A1(self.unique_id, self, self.start_energy, loc)
            self.unique_id += 1
            self.grid.place_agent(a, new_position)
            self.schedule.add(a)
            
    def introduce_agents(self, which_agent):
        x = random.random() * self.grid.width
        y = random.random() * self.grid.height
            
        if which_agent == "A1":
            self.og_locs[(x,y)] = "alive"
            a = A1(self.unique_id, self, self.start_energy, (x,y))
            self.unique_id += 1
            self.grid.place_agent(a, (x, y) )
            self.schedule.add(a)
            if self.nstep>30:
                self.let_plants_grow_geez((x,y), 100)
            else:
                self.let_plants_grow_geez((x,y), 100)

              
        elif which_agent == "A2":
            c = [random.uniform(-1, 1), random.random()]
            a = A2(self.unique_id, self, self.start_energy, cognition = c)
            self.unique_id += 1 
            self.grid.place_agent(a, (x,y))
            self.schedule.add(a)    
            
    def return_zero(self, num, denom, sd=False):
        if self.nstep == 1:
            return(0)
        if denom == "old_nA2":
            denom = self.history["nA2"][self.nstep-2]
        if denom == 0.0:
            return 0
        if sd==True:
            return np.std(num)
        return(num/denom)
        
    def get_radius_reprod(self, exp, center):
        radius = np.random.exponential(exp, 1)[0]
        angle = random.random() * math.pi * 2
        x = math.cos(angle) * radius + center[0]
        y = math.sin(angle) * radius + center[1]
        x = x%100
        y = y%100
        return(x, y)
        
    def nearest_neighbor(self, agent):
        if agent == "a1":
            x = self.grid._agent_points[self.grid._agent_ids==1]
            if len(x)<=10:
                return(-1)
            if len(x)>3500:
                return(0.97)
            if self.nstep<300 and self.skip_300:
                return(-1 )
        else:
            x = self.grid._agent_points[self.grid._agent_ids==2]
            if len(x)<=10:
                return(-1)
        density = len(x)/ (self.grid.width)**2
        expect_neigh_ = self.expect_NN
        expect_dist = np.array(expect_neigh_) /(density ** 0.5)
        distances = 0
        for i in x:   # calculates pairwise distances in a toroid
            distx = abs(x[:,0]-i[0])
            distx[distx>50] = 100-distx[distx>50]
            disty = abs(x[:,1]-i[1])
            disty[disty>50] = 100-disty[disty>50]
            dist = (distx**2+disty**2)**0.5
            distances += (np.partition(dist, 5)[5])
        mean_dist = np.array(distances)/len(x)
        out = mean_dist/expect_dist[0]
        return(out)
    
    def collect_hist(self):
        neigh_calc = [0,0]#[self.nearest_neighbor("a1"), 0] 
        neigh_animcalc = [0,0]#self.nearest_neighbor("a2")
        dat = { "nA1" : self.nA1, "nA2" : self.nA2,
               "age" : self.return_zero(sum(self.age), self.nA2),
               "cognition0" : self.return_zero(sum(self.cog0), self.nA2),
               "cognition1" : self.return_zero(sum(self.cog1), self.nA2),
               "cognition0sd" : self.return_zero(self.cog0, self.nA2, sd=True),
               "cognition1sd" : self.return_zero(self.cog1, self.nA2, sd=True),
               "neigh5": neigh_calc[0],"neigh10": neigh_calc[1],
               "neighanim5": neigh_animcalc[0],"neighanim10": neigh_animcalc[1],
               "reprod" : self.return_zero(self.reprod, "old_nA2" ), "food": self.return_zero(self.food, self.nA2),
               "death" : self.return_zero(self.death, "old_nA2"), "combat" : self.return_zero(self.combat, "old_nA2")}
        self.history = self.history.append(dat, ignore_index = True)
        self.age = []
        self.cog0 = []
        self.cog1 = []
        (self.reprod, self.food, self.death, self.combat) = (0, ) * 4
  
    def step(self):
        self.nstep +=1 # step counter
        self.food_ls = self.get_ls(1)
        self.comp_ls = self.get_ls(2)
        if self.nstep == self.intro_time:
            for i in range(self.a2num):
                self.introduce_agents("A2")  
        self.schedule.step()  
        self.nA1 = np.sum(self.grid._agent_ids==1)            
        self.nA2 = np.sum(self.grid._agent_ids==2)
        self.collect_hist()
        if self.nstep%40 == 0:
            dead = False
            while dead == False:
                loc = random.choice(list(self.og_locs.keys()) )
                if self.og_locs[loc]=="alive":
                    dead=True
                    self.og_locs[loc] = "dead"
                    self.introduce_agents("A1")
        if self.nstep%10 == 0:
            sys.stdout.write( str(self.nstep) +" "  +str(self.nA1) + " " + str(self.nA2) + " " + str(self.history.cognition0[self.nstep-2]) +"\n") 
        
    def animate(self):
        colors = ['midnightblue', 'mediumseagreen', 'white']
        plot_c = [colors[aid] for aid in self.grid._agent_ids]
        n = str(self.nstep)
        fig = plt.scatter(self.grid._agent_points[:, 0], self.grid._agent_points[:, 1], \
                          c = plot_c, s = self.grid._agent_ids*0.8)
        ax = plt.gca()
        ax.set_facecolor(colors[0])
        plt.title("Step #" + n, loc = "right")
       # plt.axis("off")
        return(fig)
        
    def get_ls(self, agent, gd=400):  # code taken from:       
        data = self.grid._agent_points[self.grid._agent_ids == agent]
        # Generate 2D data.
        x_data, y_data = data[:, 0], data[:, 1]
        xmin, xmax = (0,self.grid.width) #min(x_data), max(x_data)
        ymin, ymax = (0,self.grid.width) #min(y_data), max(y_data)
        
        # Define grid density.
        #gd = 400
        # Define bandwidth
        bw = 1.5 * self.spread
        if agent == 2:
            bw*=self.anim_spread
        
        # Using gaussian_filter
        # Obtain 2D histogram.
        rang = [[xmin, xmax], [ymin, ymax]]
        binsxy = [gd, gd]
        hist1, xedges, yedges = np.histogram2d(x_data, y_data, range=rang, bins=binsxy)
        # Gaussian filtered histogram.
        h_g = gaussian_filter(hist1, bw)
        return h_g
    
    def visualize_ls(self, a1 = 1, a2 = 0):
        h_g = a1*self.get_ls(1) + a2*self.get_ls(2)
        # Make plots.
        fig, ax1 = plt.subplots(1, 1)
        # Gaussian filtered 2D histograms.
        ax1.imshow(h_g.transpose(), origin='lower')
        
    def points_on_circumference(self, center_tup, r, n=20):
        return [
            [center_tup[0]+(math.cos(2 * math.pi / n * x) * r),  # x
             center_tup[1] + (math.sin(2 * math.pi / n * x) * r)  # y
            ] for x in np.arange(0, n + 1)]
            
    def visualize(self):
        #f, ax = plt.subplots(1)
        plt.figure()
        fig = plt.scatter(self.grid._agent_points[:, 0], self.grid._agent_points[:, 1], \
                          c = self.grid._agent_ids, s = self.grid._agent_ids*0.8)
       # plt.axis("off")
        return(fig)