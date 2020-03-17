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
        self.eps = max(self.eps*self.eps_decay, self.eps_min)
        
    def get(self):
        return self.eps


class policy(object):
    
    def get_action(self, Q_state):
        pass
    
    def update(self, episode_counter):
        pass
    
class random(policy):
    def get_action(self, Q_state):
        return np.random.randint(0,len(Q_state))
    
class maximum(policy):
    def get_action(self, Q_state):
        return np.argmax(Q_state)
    
class epsilon_greedy(policy):
    def __init__(self, epsilon, schedule):
        self.epsilon = epsilon
        self.schedule = schedule
        

    def get_action(self,Q_state, else_take = None):
        eps = self.epsilon.get()
        rv = np.random.random()
        if rv > eps:
            return np.argmax(Q_state)
        else:
            if else_take is None:
                return np.random.choice(np.arange(len(Q_state)))
            else:
                return np.random.choice(else_take)
    
    
    def update(self, episode_counter):
        if episode_counter%self.schedule==0:
            self.epsilon.update()
        else:
            pass
        
        
def play(env, Q, n_episodes, policy, window = 100):
    scores = []
    moving_scores = deque(maxlen = window)
    moving_average_scores = []
    for i_epsiodes in range(0,n_episodes):
        print('\r','Episode [{}/{}]'.format(i_epsiodes, n_episodes),end='')
        done = False
        next_state = env.reset()
        score = 0
        while not done:
            next_action = policy.get_action(Q[next_state])
            next_state, reward, done, _ = env.step(next_action)
            score+=reward
        scores.append(score)
        moving_scores.append(score)
        moving_average_scores.append(np.mean(moving_scores))
    return np.mean(scores), scores, moving_average_scores

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
    
def MC(env, policy, Q_class ,num_episodes, generate_episode, stopping = None, print_every = 1000):
    mean_rewards = deque(maxlen = 100)
    for i_episode in range(1, num_episodes+1):
        episode = generate_episode(env, policy, Q_class.Q)
        score = Q_class.update(env, episode)
        mean_rewards.append(score)
        policy.update(i_episode)
        if i_episode % print_every == 0:
            print("\rEpisode [{}/{}] with mean reward {}".format(i_episode, num_episodes, np.mean(mean_rewards)), end="")
            sys.stdout.flush()
        if stopping:
            if np.mean(mean_rewards) > stopping:
                return Q_class,mean_rewards
    return Q_class,mean_rewards

## TD

class update_sarsa_Q(update_Q):
    def __init__(self, n_actions, alpha, gamma):
        super(update_sarsa_Q, self).__init__(n_actions, gamma)
        self.alpha = alpha
        
    def update(self,  state, action, reward, next_state=None, next_action=None):
        current = self.Q[state][action]
        Qsa_next = self.Q[next_state][next_action] if next_state is not None else 0    
        target = reward + (self.gamma * Qsa_next)               # construct TD target
        new_value = current + (self.alpha * (target - current)) # get updated value
        self.Q[state][action] = new_value
        
class update_sarsamax_Q(update_Q):
    def __init__(self, n_actions, alpha, gamma):
        super(update_sarsamax_Q, self).__init__(n_actions, gamma)
        self.alpha = alpha
        
    def update(self,  state, action, reward, next_state=None, next_action=None):
        current = self.Q[state][action]
        Qsa_next = max(self.Q[next_state]) if next_state is not None else 0    
        target = reward + (self.gamma * Qsa_next)               # construct TD target
        new_value = current + (self.alpha * (target - current)) # get updated value
        self.Q[state][action] = new_value
        
class update_expectedsarsamax_Q(update_Q):
    def __init__(self, n_actions, alpha, gamma, eps):
        super(update_expectedsarsamax_Q, self).__init__(n_actions, gamma)
        self.alpha = alpha
        self.eps = eps
        
    def update(self,  state, action, reward, next_state=None, next_action=None):
        current = self.Q[state][action]         
        policy_s = np.ones(self.n_actions) * self.eps / self.n_actions  
        policy_s[np.argmax(self.Q[next_state])] = 1 - self.eps + (self.eps / self.n_actions) 
        Qsa_next = np.dot(self.Q[next_state], policy_s)         
        target = reward + (self.gamma * Qsa_next)               
        new_value = current + (self.alpha * (target - current))  
        self.Q[state][action] = new_value
        
def TD(env, policy, Q_class, num_episodes, stopping = None, print_every = 1000):
    tmp_scores = deque(maxlen=100) 
    means = []
    for i_episode in range(1, num_episodes+1):  
                                           # initialize score
        state = env.reset()                                   # start episode
        action = policy.get_action(Q_class.Q[state])   
        score = 0
        while True:
            next_state, reward, done, info = env.step(action) # take action A, observe R, S'
            score += reward                                   # add reward to agent's score
            if not done:
                next_action = policy.get_action(Q_class.Q[next_state]) # epsilon-greedy action
                Q_class.update(state, action, reward, next_state, next_action)
                                                                  
                state = next_state 
                action = next_action   
            if done:
                Q_class.update(state, action, reward)
                tmp_scores.append(score)
                break
        if i_episode % print_every == 0:
            means.append(np.mean(tmp_scores))
            print("\rEpisode [{}/{}] with mean reward {}".format(i_episode, num_episodes, np.mean(tmp_scores)), end="")
            sys.stdout.flush()
            if stopping and np.mean(tmp_scores) > stopping:
                print(' solved in ', i_episode,'with average reward : ',np.mean(tmp_scores))
                return means
        policy.update(i_episode)

    return means
