import environments
import pandas as pd
import numpy as np
import sys
from collections import defaultdict
import matplotlib.pyplot as plt


from collections import deque

class epsilon(object):
    def __init__(self, eps_start = 1.0, eps_decay = 0.999, eps_min = 0.0):
        self.eps_start = eps_start
        self.eps = self.eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        
    def update(self):
        self.eps = max(self.eps_start*self.eps_decay, self.eps_min)
        
    def get(self):
        return self.eps


class policy(object):
    
    def get_action(self, Q_state):
        pass
    
    def update(self):
        pass
    
class random(policy):
    def get_action(self, Q_state):
        return np.random.randint(0,len(Q_state))
    
class maximum(policy):
    def get_action(self, Q_state):
        return np.argmax(Q_state)
    
class epsilon_greedy(policy):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        
    def get_action(self,Q_state):
        eps = self.epsilon.get()
        rv = np.random.uniform(0,1)
        if rv > eps:
            return np.argmax(Q_state)
        else:
            return np.random.choice(np.arange(len(Q_state)))
    
    def update(self):
        self.epsilon.update()
        
        
def play(env, Q, n_episodes, policy):
    scores = []
    for i_epsiodes in range(0,n_episodes):
        done = False
        next_state = env.reset()
        score = 0
        while not done:
            next_action = policy.get_action(Q[next_state])
            next_state, reward, done, _ = env.step(next_action)
            score+=reward
        scores.append(score)
    return np.mean(scores), scores

def generate_episode(env, policy, Q):
    episode = []
    state = env.reset()
    while True:
        action = policy.get_action(Q[state])
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

class update_Q:
    def __init__(self, n_actions, gamma):
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        
    def update(self):
        pass
    
class update_vanilla_Q(update_Q):
    def __init__(self, n_actions,gamma):
        super(update_vanilla_Q, self).__init__(n_actions,gamma)
        self.N = defaultdict(lambda: np.zeros(n_actions))
        self.returns_sum = defaultdict(lambda: np.zeros(n_actions))
        
    def update(self, env, episode):
        score = 0
        states, actions, rewards = zip(*episode)
        discounts = np.array([self.gamma**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            self.returns_sum[state][actions[i]] += sum(rewards[i:]*discounts[:-(1+i)])
            self.N[state][actions[i]] += 1.0
            self.Q[state][actions[i]] = \
            self.returns_sum[state][actions[i]] / self.N[state][actions[i]]
            score+=rewards[i]
        return score

class update_control_Q(update_Q):
    def __init__(self, n_actions, alpha, gamma):
        super(update_control_Q, self).__init__(n_actions, gamma)
        self.alpha = alpha
        
    def update(self, env, episode):
        score = 0
        states, actions, rewards = zip(*episode)
        discounts = np.array([self.gamma**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            old_Q = self.Q[state][actions[i]]
            self.Q[state][actions[i]] = old_Q + self.alpha*(sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)
            score+=rewards[i]
        return score
    
def MC(env, policy, Q_class ,num_episodes, generate_episode, stopping = None):
    mean_rewards = deque(maxlen = 100)
    for i_episode in range(1, num_episodes+1):
        episode = generate_episode(env, policy, Q_class.Q)
        score = Q_class.update(env, episode)
        mean_rewards.append(score)
        policy.update()
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{} with mean reward {}".format(i_episode, num_episodes, np.mean(mean_rewards)), end="")
            sys.stdout.flush()
        if stopping:
            if np.mean(mean_rewards) > stopping:
                return Q_class,mean_rewards
    return Q_class,mean_rewards