import numpy as np
import pandas as pd
import os
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
import  matplotlib.pyplot as plt
from utils import *
from em_enviroment2 import CustomEnv


def obs_list_to_state_vector(observation):
    state = np.array([])
   
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    scenario = 'em'
    directorio = os.getcwd()
    directorioclases= directorio + '/simulated_data/'
    #print(directorioclases+choose_data(0))
    df = pd.read_excel(directorioclases+choose_data(8))
    
    env = CustomEnv(df ,3 ,12)
    env.reset()
    
    n_agents = env.n_agents
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    critic_dims = sum(actor_dims) 
    
    # action space is a list of arrays, assume each agent has same action space
    n_actions = env.n_actions
    #print(actor_dims, critic_dims, n_agents  , n_actions)
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents  , n_actions, 
                           fc1=32, fc2=32,  
                           alpha=0.001, beta=0.001, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(100000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 50
    N_GAMES = 100000
    MAX_STEPS = 20
    total_steps = 0
    score_history = []
    best_history = []
    avg_score_history = []
    avg_best_history = []
    evaluate = False
    best_score = -130000

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
       
        obs1 = np.array(env.reset())
        obs = []
        for j in range(n_agents):
            obs.append(obs1)
        
        score = 0
        best = 0
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                env.render()
                #time.sleep(0.1) # to slow down the action for the video
            
            actions = maddpg_agents.choose_action(obs)
            
            obs_, reward, done, info, obs1 = env.step(actions)

            if i % 50 == 0:
                env.render()
            
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs1)
            
            if episode_step >= MAX_STEPS:
                done = [True]*n_agents
            
            memory.store_transition(obs, state, actions, reward, obs1, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)
                

            obs = obs_
            
            score += sum(reward)
            
            best += env.best 
            total_steps += 1
            episode_step += 1
        
        score_history.append(score)
        best_history.append(best)
        avg_score = np.mean(score_history[-100:])
        avg_best = np.mean(best_history[-100:])
        avg_score_history.append(avg_score)
        avg_best_history.append(avg_best)
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
               
        if i % PRINT_INTERVAL == 0 :
            print('episode', i, 'average score {:.1f}'.format(avg_score), 'best posible {:.1f}'.format(avg_best))
        
        if i > 0 and i % 500 == 0:
           plt.plot(score_history, ".")
           plt.plot(avg_score_history, "-")
           plt.plot(avg_best_history, "-")
           plt.xlabel('Episode')
           plt.ylabel('Reward')
           
           plt.savefig(directorio + "/plots3/ "+str(i / 500)+" .png")


