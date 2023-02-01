import torch as T
import torch.nn.functional as F
import numpy as np
from agent import Agent

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.001, beta=0.001, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario 
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        #print(raw_obs)
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(np.array(actions), dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)
        

        all_agents_new_actions = []
        old_agents_actions = []

        
        for agent_idx, agent in enumerate(self.agents):

            new_states = T.tensor(actor_new_states[agent_idx], 
                                dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            with T.no_grad():
                critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
                target = rewards[:,agent_idx] + (1-dones[:,0].int())*agent.gamma*critic_value_

            critic_value = agent.critic.forward(states, old_actions).flatten()
            
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            oa = old_actions.clone()
            oa[:,agent_idx*self.n_actions:agent_idx*self.n_actions+self.n_actions] = agent.actor.forward(mu_states)            
            actor_loss = -T.mean(agent.critic.forward(states, oa).flatten())
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            
        for agent in self.agents:    
            agent.update_network_parameters()
