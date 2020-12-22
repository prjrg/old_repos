from collections import deque, namedtuple
import random
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, batch_steps, state_shape, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.batch_steps = batch_steps
        self.pad = np.zeros(state_shape)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = np.random.choice(len(self.memory), self.batch_size)

        s = []
        a = []
        r = []
        ns = []
        ds = []
        for i in range(len(self.batch_size)):
            am = min([experiences[i] + self.batch_steps, len(self.memory)])
            xss = [e.state for e in self.memory[experiences[i]:am]]
            xsa = [e.action for e in self.memory[experiences[i]:am]]
            xsr = [e.reward for e in self.memory[experiences[i]:am]]
            xsns = [e.next_state for e in self.memory[experiences[i]:am]]
            xsd = [e.done for e in self.memory[experiences[i]:am]]

            if am == len(self.memory):
                state = self.memory[-1].state
                for k in range(experiences[i] + self.batch_steps - len(self.memory)):
                    xss.append(state)
                    xsns.append(state)
                    xsa.append(self.memory[-1].action)
                    xsr.append(0.0)
                    xsd.append(self.memory[-1].done)

            s = np.hstack(xss)
            a = np.hstack(xsa)
            r = np.hstack(xsr)
            ns = np.hstack(xsns)
            ds = np.hstack(xsd)

        states = torch.from_numpy(np.vstack([state for state in s])).float().to(device)
        actions = torch.from_numpy(np.vstack([action for action in a])).long().to(device)
        rewards = torch.from_numpy(np.vstack([reward for reward in r])).float().to(device)
        next_states = torch.from_numpy(np.vstack([next_state for next_state in ns])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([done for done in ds]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)