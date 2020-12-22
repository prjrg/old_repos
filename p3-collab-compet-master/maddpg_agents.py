# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network
from ddpg_agent import DDPGAgent 
from replay_buffer import ReplayBuffer
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
UPDATE_EVERY = 10       # Update the networks after UPDATE_EVERY timesteps
N_LEARNING_PASSES = 10  # Number of times we update the network

class MADDPG:
    def __init__(self, state_size, action_size, random_seed, n_agents=2):
        super(MADDPG, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.t_step = 0
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed, n_agents)
        
        # critic input
        self.maddpg_agents = [DDPGAgent(index=i, state_size=state_size, action_size=action_size, random_seed=random_seed, memory=self.memory, n_agents=n_agents) for i in range(n_agents)]
        
    def act(self, states):
        """Each agent acts to choose an action corresponding to the policy"""
        actions = np.zeros([self.n_agents, self.action_size])
        for agent, state in zip(self.maddpg_agents, states):
            actions[agent.index, :] = agent.act(state)
        return actions
        
    
    def step(self, states, actions, rewards, next_states, dones):
        """Perform a step on all agents and add previously the experience to the replay buffer"""
        
        # Add the experience to the replay buffer
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory and we have updated UPDATE_EVERY times the memory replay buffer
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
               # Learn several passes from the batch, which means utilize more collected experiences
               self.learn()
            
    def learn(self):
        for _ in range(N_LEARNING_PASSES):
            for maddpg_agent in self.maddpg_agents:
                experiences = self.memory.sample()
                maddpg_agent.learn(experiences)
                
    def reset(self):
        for maddpg_agent in self.maddpg_agents:
            maddpg_agent.reset()
            
    def get(self, i):
        """Return the agent given by the index"""
        return self.maddpg_agents[i]
    