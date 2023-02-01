import numpy as np
import pandas as pd 
from dataclasses import dataclass
import os
#from sklearn.cluster import KMeans 
import seaborn as sns 
import gym
from gym import spaces
import matplotlib.pyplot as plt
from time import sleep
gym.logger.set_level(40)
import random 
#from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from iteration_utilities import deepflatten
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from utils import *

class agent():
  def __init__(self, data_dim):
    super(agent, self).__init__()
    self.state = []
    self.obs = []
    self.id = 0
    self.state_mean = np.array([])
    self.state_cov = np.array([])
    self.state_n_data = np.array([])
    self.data_dim = data_dim
    self.action = []
    self.low_space_act = list(deepflatten([ -100 * np.ones(self.data_dim), -100 * np.ones(self.data_dim**2), np.zeros(self.data_dim), [-0.5]]))
    self.high_space_act = list(deepflatten([ 100* np.ones(self.data_dim), 100 * np.ones(self.data_dim**2), 1000 * np.ones(self.data_dim), [0.5]]))
    
    self.low_space_obs = list(deepflatten([ -100 * np.ones(self.data_dim), -20 * np.ones(self.data_dim**2), [0], [0]])) 
    self.high_space_obs = list(deepflatten([ 100 * np.ones(self.data_dim), 20 * np.ones(self.data_dim**2), [1],[30]])) 

  def move(self, action):
  
    self.state_mean = self.state[:self.data_dim] + action[self.id][:self.data_dim] 
    
    act_cov = action[self.id][self.data_dim: self.data_dim**2+self.data_dim].reshape(self.data_dim, self.data_dim)
    eigen_values = np.abs(action[self.id][self.data_dim**2+self.data_dim: self.data_dim**2+2*self.data_dim])
    
    self.state_cov = np.array(self.state[self.data_dim: self.data_dim**2+self.data_dim]).reshape(self.data_dim, self.data_dim)
    + spectral_composition(act_cov,eigen_values)  
    
    self.state_n_data =  self.state[ self.data_dim**2 + self.data_dim:] + action[self.id][self.data_dim**2 + 2*self.data_dim:] 
    
    self.action = list(deepflatten([action[self.id][:self.data_dim],spectral_composition(act_cov,eigen_values), self.state[ self.data_dim**2 + self.data_dim:] ]))
    
  def reset(self):
  
    self.state_mean = self.state[:self.data_dim] 
    

    self.state_cov = np.array(self.state[self.data_dim: self.data_dim**2+self.data_dim]).reshape(self.data_dim, self.data_dim)
 
    
    self.state_n_data =  self.state[ self.data_dim**2 + self.data_dim:] 

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, df, n_agents, history_times):
    super(CustomEnv, self).__init__()

    self.df = df
    self.step_data = df

    self.obs = []#np.array([], dtype=np.float16)


    self.agents = np.zeros(n_agents)
    self.n_agents = n_agents
    self.history_times = history_times
    self.data_dim = df.shape[1] - 1
    self.n_actions = self.data_dim + (1+self.data_dim)*self.data_dim + 1 
    
    self.means = []
    self.covariances = []
    self.current_step = 0
    self.best = [] 
    self.agents = []
    
    # Define action and observation space

    
    # configure spaces
    self.action_space = []
    self.observation_space = []
    total_action_space = []
    total_observation_space = []
    for idx in range(self.n_agents):
      self.agents.append(agent(self.data_dim))  
      agent_action_space = spaces.Box(low=np.array(self.agents[idx].low_space_act), high=np.array(self.agents[idx].high_space_act), dtype=np.float128) #shape=
                      #(self.data_dim + self.data_dim**2 + 1, ), dtype=np.float128)
      total_action_space.append(agent_action_space)

      agent_observation_space = spaces.Box(low=np.array(self.agents[idx].low_space_obs * self.n_agents  * self.history_times), high=np.array(self.agents[idx].high_space_obs * self.n_agents * self.history_times), dtype=np.float128)
      total_observation_space.append(agent_observation_space)
    
    self.action_space = total_action_space
    self.observation_space = total_observation_space
       
  def take_action(self,action):
    
    action = self.normalize_action(action)
    for agent in range(self.n_agents):
      self.agents[agent].move(action)
    
  def step(self, action):

    
    done = [self.current_step > 99] * self.n_agents
    self.take_action(action)

    self.current_step += 1
    self.best = self.best_posible() 
    
    reward = [self.reward()/ self.n_agents] * self.n_agents
    _obs = []
    obs_ = []
 
    obs, obs1 = np.array(self._next_observation(), dtype=np.float16)
    #reward = [self.rewards/ self.n_agents] * self.n_agents
    

    for agent in range(self.n_agents):
      _obs.append(obs)
      obs_.append(obs1)
    

    return _obs, reward, done, {}, obs_

  def _next_observation(self):

    self.step_data = self.df[self.df["current_step"] == self.current_step].loc[:,self.df.columns!="current_step"]
    gm = GaussianMixture(n_components= self.n_agents, covariance_type= "full", random_state=0).fit(self.step_data)
    
    self.ind = tidy_obs( self.means, gm.means_, )
    self.means = gm.means_
    self.covariances = gm.covariances_
    self.weights = gm.weights_
    new_obs = []
    old = []
    new =[]
    _obs = self.obs.copy()
    r = []
    for id in range(self.n_agents):
      old.append([self.means[id], self.covariances[id], self.n_data[id]])
      
      mean = gm.means_[self.ind[id]]
      cov = gm.covariances_[self.ind[id]] 
      n_data = gm.weights_[self.ind[id]] 
      
      new.append(list(deepflatten([mean, cov, n_data])))
      if self.agents[id].action == []:
        self.agents[id].action = np.zeros(len(self.agents[id].low_space_obs))
      
      _obs.append(self.agents[id].action)
      r.append(self.agents[id].action)
     
    
    old = list(deepflatten(old))
    new = list(deepflatten(new))
    
    new_obs.append(np.array(new) - np.array(old))
    new_obs.append([self.current_step] * self.n_agents)
    _obs.append([self.current_step] * self.n_agents)
    self.means = gm.means_
    self.covariances = gm.covariances_
    self.weights = gm.weights_
    r=np.array(list(deepflatten(r)))
    new_obs = list(deepflatten(new_obs))

   
    self.obs.append(new_obs)

    
    self.obs = list(deepflatten(self.obs))
    _obs = list(deepflatten(_obs))
    
    for i in range(self.n_agents * (self.data_dim + self.data_dim**2 + 2)):
      self.obs = np.delete(self.obs, 0)
      _obs = np.delete(_obs, 0)
    agents = []

    for agent1 in range(self.n_agents):
      #agents.append(agent(self.data_dim))
      self.agents[agent1].state = (list(deepflatten([ self.means[self.ind[agent1]], self.covariances[self.ind[agent1]], self.weights[self.ind[agent1]]])))  
      #agents[agent1].reset()
    
    
    
    self.obs = list(deepflatten(self.obs))

    return self.obs, _obs

  def reset(self):
    
    # Reset the state of the environment to an initial state   
    self.current_step = 0
    
    #charge dataframe
    self.df = self.df
    self.obs = []

    self.step_data = self.df[self.df["current_step"] == 0].loc[:,self.df.columns!="current_step"]
    gm = GaussianMixture(n_components= self.n_agents, covariance_type= "full", random_state=0).fit(self.step_data)
    self.means = gm.means_
    self.covariances = gm.covariances_
    self.n_data = gm.weights_
    #history of the first history_times observations
    for time in range(1,self.history_times+1):
      new_obs = []
      self.step_data = self.df[self.df["current_step"] == time].loc[:,self.df.columns!="current_step"]
      gm = GaussianMixture(n_components= self.n_agents, covariance_type= "full", random_state=0).fit(self.step_data)
      self.means = gm.means_
      self.ind = tidy_obs(self.means, gm.means_)
      old = []
      new =[]
      for id in range(self.n_agents):
        old.append([self.means[id], self.covariances[id], self.n_data[id]])
        
        mean = gm.means_[self.ind[id]]
        cov = gm.covariances_[self.ind[id]] 
        n_data = gm.weights_[self.ind[id]] 
        
        new.append(list(deepflatten([mean, cov, n_data])))
        
      old = list(deepflatten(old))
      new = list(deepflatten(new))
      new_obs.append(np.array(new) - np.array(old))
      new_obs.append([time] * self.n_agents)
      #new_obs = list(deepflatten(new))
      self.means = gm.means_
      self.covariances = gm.covariances_
      self.weights = gm.weights_
      
      self.obs.append(new_obs)
      
      self.current_step += 1
   
    self.obs = list(deepflatten(self.obs))
    
    
    #initializiate the agents
    agents = []
    for agent1 in range(self.n_agents):
      agents.append(agent(self.data_dim))
      agents[agent1].state = (list(deepflatten([self.means[self.ind[agent1]], 
                     self.covariances[self.ind[agent1]], self.weights[self.ind[agent1]]])))
      agents[agent1].id = agent1  
      agents[agent1].reset()
      
    self.agents = agents
    
    return self._next_observation()[0]
  
  def reward(self):
    
    self.step_data = self.df[self.df["current_step"] == self.current_step ].loc[:,self.df.columns!="current_step"]
    pos_matrix = np.zeros(shape=(len(self.step_data), self.n_agents))
    gm = GaussianMixture(n_components= self.n_agents, covariance_type= "full", random_state=0).fit(self.step_data)

    for i in range(self.n_agents):
      mean = self.agents[i].state_mean
        
      cov = self.agents[i].state_cov

   
      if self.agents[i].state_n_data < 0:
        self.agents[i].state_n_data = 0 
      if self.agents[i].state_n_data >1:
        self.agents[i].state_n_data = 1

      
      pos_matrix[:, i] = self.agents[i].state_n_data * multivariate_normal.pdf(self.step_data, mean=mean, cov=cov)

      #normalize pos_matrix to have a valid probability
    norm_pos_matrix = np.sum(pos_matrix, axis=1)[:,np.newaxis] + 0.000001

    
    pos_matrix /= norm_pos_matrix
      

    loss_function = 0
    for x in range(len(self.step_data)):
      x_loss_function = 0
      for i in range(self.n_agents):
        mean = self.agents[i].state_mean
        
        cov = self.agents[i].state_cov  
        cov = gm.covariances_[i]
        
        x_loss_function += multivariate_normal.pdf(self.step_data.iloc[x], mean=mean, cov=cov) * np.mean(pos_matrix[i], axis=0)

      loss_function += np.log(x_loss_function + 0.0000000000001)
      
    return loss_function

  def best_posible(self):
    
    self.step_data = self.df[self.df["current_step"] == self.current_step ].loc[:,self.df.columns!="current_step"]
    pos_matrix = np.zeros(shape=(len(self.step_data), self.n_agents))
    gm = GaussianMixture(n_components= self.n_agents, covariance_type= "full", random_state=0).fit(self.step_data)

    for i in range(self.n_agents):
      mean = gm.means_[i]
      cov = gm.covariances_[i]
      pos_matrix[:, i] = gm.weights_[i] * multivariate_normal.pdf(self.step_data, mean=mean, cov=cov)

      #normalize pos_matrix to have a valid probability
    norm_pos_matrix = np.sum(pos_matrix, axis=1)[:,np.newaxis]
    pos_matrix /= norm_pos_matrix
      

    loss_function = 0
    for x in range(len(self.step_data)):
      x_loss_function = 0
      for i in range(self.n_agents):
        mean = gm.means_[i]
        
        cov = gm.covariances_[i]
        
        x_loss_function += multivariate_normal.pdf(self.step_data.iloc[x], mean=mean, cov=cov) * np.mean(pos_matrix[i], axis=0)

      loss_function += np.log(x_loss_function + 0.0000000000000000000000001)
      
   
    return loss_function

  def render(self, mode='human', close=False):
    self.step_data = self.df[self.df["current_step"] == self.current_step ].loc[:,self.df.columns!="current_step"]
    pos_matrix = np.zeros(shape=(len(self.step_data), self.n_agents))
    gm = GaussianMixture(n_components= self.n_agents, covariance_type= "full", random_state=0).fit(self.step_data)
    
    num_c = 100
    
    for agent in range(self.n_agents):
      mean = self.agents[agent].state_mean
      cov = self.agents[agent].state_cov
          
      a = np.random.multivariate_normal(mean, cov, size=num_c, check_valid='warn', tol=1e-8)
      
      # create data
      x = a[:,0]
      y = a[:,1]

      # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
      nbins=300
      k = gaussian_kde([x,y])
      xi, yi = np.mgrid[-20:100:nbins*1j, -200:100:nbins*1j]
      zi = k(np.vstack([xi.flatten(), yi.flatten()]))**0.2
      plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', alpha= 1 - agent/self.n_agents)
      
      plt.scatter(mean[0], mean[1], s= 10, color = "blue", alpha=1)

    x = self.step_data.iloc[:,0]
    y = self.step_data.iloc[:,1]

    plt.scatter(x,y, s= 7, color="red", alpha=0.3)  
    plt.xlim(-20,100)#x_0-50, x_0+50)
    plt.ylim(-20,100)
    plt.pause(0.5)
    plt.clf()
    plt.show(block = False)
    if self.current_step >= 21 + self.history_times:
        plt.close()
    
  def normalize_action(self, action):
    act_k = (self.action_space[0].high - self.action_space[0].low)/ 2.
    act_b = (self.action_space[0].high + self.action_space[0].low)/ 2.
    return act_k * action + act_b

  def normalize_reverse_action(self, action):
    act_k_inv = 2./(self.action_space[0].high - self.action_space[0].low)
    act_b = (self.action_space[0].high + self.action_space[0].low)/ 2.
    return act_k_inv * (action - act_b)


directorio = os.getcwd()

directorioclases= directorio + '/simulated_data/'

