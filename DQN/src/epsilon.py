import numpy as np

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