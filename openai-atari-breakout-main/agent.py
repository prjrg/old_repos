import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from model import QNetwork
from optimizer_ranger_with_gc import Ranger

BUFFER_SIZE = int(3e5)
BATCH_SIZE = 8
NUM_EPISODES_PER_BATCH = 5
GAMMA = 0.99
TAU = 1e-3
LR = 1e-3
UPDATE_EVERY = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def PIL2array(img):
    return np.array(img.getdata(),np.uint8).reshape(img.size[1], img.size[0], 4)

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

def processScreen(screen):
    s = [210,160]
    image = array2PIL(screen,s)
    newImage = image.resize((140, 140))
    xtt = PIL2array(newImage)
    xtt = xtt.reshape(xtt.shape[2],xtt.shape[0],xtt.shape[1])
    img = torch.from_numpy(np.array(xtt))
    img = img.type('torch.FloatTensor')
    return img/255.0

def addEpisode(ind,prev,curr,reward,act, store):
    if len(store[ind]) ==0:
        store[ind][0]={'prev':prev,'curr':curr,'reward':reward,'action':act}
    else:
        store[ind].append({'prev':prev,'curr':curr,'reward':reward,'action':act})

def add_to_tensors(start, length, store, inp, target, rew, actions, ep):
    for i in range(start, length, 1):
        inp.append((store[ep][i]).get('prev'))
        target.append((store[ep][i]).get('curr'))
        rew[0][i - start] = store[ep][i].get('reward')
        actions[0][i - start] = store[ep][i].get('action')

# TODO
class Agent:
    def __init__(self, state_size, action_size, seed, env):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.main_network = QNetwork(state_size[0], state_size=state_size[2], action_size=action_size, seed=seed).to(device)
        self.target_network = QNetwork(state_size[0], state_size=state_size[2], action_size=action_size, seed=seed).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

        self.optimizer = Ranger(self.main_network.parameters(), lr=1e-4)
        self.criterion = F.smooth_l1_loss

        self.memory = 100000
        self.store = [[dict()] for i in range(self.memory)]
        self.env = env

    def train(self, total_episodes):
        store = self.store
        if total_episodes == 0:
            return
        ep = random.randint(0, total_episodes -1)
        if len(self.store[ep]) < 8:
            return
        else:
            start = random.randint(1, len(store) - 1)
            length = len(store[ep])
            inp = []
            target = []
            rew = torch.Tensor(1, length - start)
            actions = torch.Tensor(1, length - start)
            add_to_tensors(start, length, store, inp, target, rew, actions, ep)
            targets = torch.Tensor(target[0].shape[0], target[0].shape[1], target[0].shape[2]).to(device)
            torch.cat(target, out=targets)
            ccs = torch.Tensor(inp[0].shape[0], inp[0].shape[1], inp[0].shape[2]).to(device)
            torch.cat(inp, out=ccs)
            hidden = self.main_network.init_hidden(length-start)
            expected_actions = self.main_network(targets, hidden).detach().max(1)[1].unsqueeze(1)
            hidden = self.main_network.init_hidden(length - start)
            qvals_target = self.target_network(targets, hidden).gather(1, expected_actions)
            hidden = self.main_network.init_hidden(length - start)
            q_expected = self.main_network(ccs, hidden).gather(1, actions)
            targ = torch.Tensor(1, qvals_target.shape[0])
            for num in range(start, length, 1):
                if num==len(store[ep])-1:
                    targ[0][num-start] = rew[0][num-start]
                else:
                    targ[0][num-start] = rew[0][num-start]+GAMMA*qvals_target[num-start]
            self.optimizer.zero_grad()
            q_expected = q_expected.reshape(1, length-start)
            loss = self.criterion(q_expected, targ)
            loss.backward()
            for param in self.main_network.parameters():
                param.grad.data.clamp(-1, 1)
            self.optimizer.step()





