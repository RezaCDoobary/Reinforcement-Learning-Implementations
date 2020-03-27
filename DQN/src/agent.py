from collections import namedtuple, deque
import random
from model import QModelNet,device
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from replaybuffer import ReplayBuffer

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, memory , local_model, target_model, optimiser, policy, UPDATE_EVERY, GAMMA, BATCH_SIZE,LR, TAU, isDDQN = False):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.local_model = local_model
        self.target_model = target_model
        self.optimizer = optimiser
        self.optimizer = self.optimizer(self.local_model.parameters(), lr = LR)
        
        self.memory = memory
        self.t_step = 0

        self.policy = policy
        self.isDDQN = isDDQN

        self.UPDATE_EVERY = UPDATE_EVERY
        self.GAMMA = GAMMA 
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
    
    def step(self, state, action, reward, next_state, done):
        """Takes a step and with each time step sample from buffer and learn"""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.local_model.eval()
        with torch.no_grad():
            action_values = self.local_model(state)
        self.local_model.train()

        # Epsilon-greedy action selection
        return self.policy.get_action(action_values.cpu().data.numpy(),np.arange(self.action_size))


    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        #need to torchify this data
        states = torch.from_numpy(states).float().to(device)
        actons = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        if self.isDDQN:
            # Get optimal action from local model and feed forward next_states on target network
            best_local_actions = self.local_model(states).max(1)[1].unsqueeze(1)
            double_dqn_targets = self.target_model(next_states)
            # Get value of the target dqn vialocal optimal action
            Q_targets_next = torch.gather(double_dqn_targets, 1, best_local_actions)
        else:
            # Get max predicted Q values (for next states) from target model (without ddqn)
            Q_targets_next = self.target_model(next_states)
            Q_targets_next = Q_targets_next.detach().max(1)[0].unsqueeze(1)

            #Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)).double()

        # Get expected Q values from local model
        Q_expected = self.local_model(states).gather(1, torch.tensor(actions)).double()
        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_model, self.target_model, self.TAU)                     

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)