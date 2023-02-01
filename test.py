import numpy as np
from make_env import make_env

env = make_env("simple_adversary")

print("n_actors ", env.n)
print("obs_space ", env.observation_space)
print("act_space ", env.action_space)
print("n_action" , env.action_space[0].n)

observation = env.reset()

no_act = [1,0,0,0,0]
action = [no_act, no_act, no_act]
#print(observation)
obs_, rewards, done, _ = env.step(action)

              
              
#print(done)
               
