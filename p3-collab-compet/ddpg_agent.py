from model import Actor, Critic
from ounoise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
NOISE_FACTOR = 0.999
NOISE_DECAY = 0.9999



class DDPGAgent:
    """Interacts with and learns from the environment."""
    
    def __init__(self, index, state_size, action_size, random_seed, memory, n_agents=2):
        """Initialize an Agent object.
        
        Params
        ======
            index (int): index of the agent
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            memory (ReplayBuffer): shared Replay Buffer in between agents
            n_agents (int): number of agents
        """
        self.index = index
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size,  n_agents, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, n_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        self.memory = memory
        
        self.n_agents = n_agents
        
        self.noise_factor = NOISE_FACTOR
        
    
    def act(self, states, add_noise=True):
        """Returns actions of both agents to take for given states as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        # Add a noise factor when considering noise
        # This factor also decays over time
        if add_noise:
            actions += self.noise.sample() * self.noise_factor
            self.noise_factor *= NOISE_DECAY
        # The Action must be inbetween -1 and 1
        return np.clip(actions, -1, 1)
    
    
    def learn(self, experiences, gamma=GAMMA):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states_full, actions_full, rewards, next_states_full, dones = experiences
        
        # Compute actions next as in the DDPG scenario
        actions_next = torch.cat([self.actor_target(s) for s in states_full], dim=1).to(device)
        
        next_states = torch.cat(next_states_full, dim=1).to(device)
        states = torch.cat(states_full, dim=1).to(device)
        actions = torch.cat(actions_full, dim=1).to(device)
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # Gradient clipping as in the benchmark implementation
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = torch.cat([self.actor_local(s) for s in states_full], dim=1).to(device)
        
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU) 


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def reset(self):
        self.noise.reset()

   
