import numpy as np
from library import *

class environment_wrapper_base(object):
    def __init__(self):
        pass
    
    def step(self, action):
        pass
        
    def action_space(self):
        pass
        
    def action_space_sample(self):
        pass
        
    def observation_space(self):
        pass
        
    def observation_space_sample(self):
        pass
    
    def reward_range(self):
        pass
        
    def reset(self):
        pass
        
    def render(self):
        pass
        
    def close(self):
        pass
    
class environment_gym(environment_wrapper_base):
    """
    Used for gym, pybullet_gym and pybullet_envs
    """
    def __init__(self,name):
        super(environment_gym).__init__()
        import gym  # open ai gym
        import pybullet_envs
        #import pybulletgym
        self.gym = gym
        self.name = name
        self.env = gym.make(self.name)
    
    def step(self, action):
        return self.env.step(action)
        
    def action_space(self):
        return self.env.action_space
        
    def action_space_sample(self):
        return self.env.action_space.sample()
        
    def observation_space(self):
        return self.env.observation_space
        
    def observation_space_sample(self):
        return self.env.observation_space.sample()
    
    def reward_range(self):
        return self.env.reward_range
        
    def reset(self):
        return self.env.reset()
        
    def render(self):
        self.env.render()
        
    def close(self):
        self.env.close()
        
class environment_unity(environment_wrapper_base):
    """A wrapper for the unity environment."""
    def __init__(self, name):
        super(environment_unity).__init__()
        from unityagents import UnityEnvironment
        self.name = name
        self.env = UnityEnvironment(file_name=self.name)
        self.brain_name = self.env.brain_names[0]
        
    def get_brain(self):
        return self.env.brains[self.brain_name]
    
    def action_space(self):
        return self.get_brain().vector_action_space_size, self.get_brain().vector_action_space_type
        
    def observation_space(self):
        return self.get_brain().vector_observation_space_size, self.get_brain().vector_observation_space_type
        
    def step(self, action):
        res = self.env.step(action)[self.brain_name]
        next_state = res.vector_observations
        reward = res.rewards
        done = res.local_done
        return next_state, reward, done
    
    def reset(self, is_train_mode=True):
        env_info = self.env.reset(train_mode=is_train_mode)[self.brain_name]
        state = env_info.vector_observations
        return state

    def close(self):
        self.env.close()

class env_wrapper:
    def __init__(self, env, is_discretised = False):
        import gym
        self.gym = gym
        self.env = env
        self.state_maps = {}
        self.action_maps = {}
        self.inverse_action_maps = {}
        self.inverse_state_maps = {}
        self.is_discretised = is_discretised
        self.create_mappings()
 
           
    def find_coordinates(self, space):
        #print(space.n)
        
        
        #note will need to add other cases to handle different gym.space/action data types
        
        if isinstance(space, self.gym.spaces.tuple_space.Tuple):
            space_description = [np.arange(coordinate_space.n) for coordinate_space in space.spaces]
            import itertools
            r = ''
            for i in range(0,len(space_description)):
                r+='space_description['+str(i)+'],'
            r = 'list(itertools.product(' + r[:-1] + '))'
            return eval(r)
        elif isinstance(space, self.gym.spaces.discrete.Discrete):
            return np.arange(space.n)
        elif isinstance(space, self.gym.spaces.box.Box):
            if self.is_discretised:
                #just on observation space for now
                self.discretised = discretise(self.env)
                return self.discretised.get_coordinates()
        return None
    
    def create_mappings(self):
        action_coordinates = self.find_coordinates(self.env.action_space())
        state_coordinates = self.find_coordinates(self.env.observation_space())
        
        
        #print(type(tuple(state_coordinates[0])))
        #print(type(action_coordinates[0]))
              
        for i in range(0,len(action_coordinates)):
            self.action_maps[action_coordinates[i]] = i
            self.inverse_action_maps[i] = action_coordinates[i]
            
        if state_coordinates:
            for i in range(0,len(state_coordinates)):
                #if len(state_coordinates[i]) > 1:
                
                self.state_maps[state_coordinates[i]] = i
                self.inverse_state_maps[i] = state_coordinates[i] 
    
    def step(self, action):
        if self.is_discretised:
            next_state, reward, is_done, info = self.env.step(self.inverse_action_maps[action])
            next_state = self.state_maps[tuple(self.discretised.mapper(next_state))]
            return next_state, reward, is_done, info
        else:
            return self.env.step(self.inverse_action_maps[action])
        
    def reset(self):
        if self.is_discretised and len(self.state_maps) != 0:
            return self.state_maps[tuple(self.discretised.mapper(self.env.reset()))]
        elif not self.is_discretised and len(self.state_maps) != 0: 
            return self.state_maps[self.env.reset()]
        else:
            return self.env.reset()
        
    def render(self):
        self.env.render()
        
    def action_size(self):
        return len(self.action_maps)
        
    def close(self):
        self.env.close()
        
class discretise(object):
    def __init__(self, env):
        self.env = env
        self.grid = None
        self.lower_discrete = None
        self.upper_discrete = None
        self.all_coordinates = []
        self.get_all_arangements()
        
    def _create_uniform_grid(self, low, high, bins):
        self.grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]

    def _discretise(self, sample):
        return list(int(np.digitize(s, g)) for s, g in zip(sample, self.grid)) 
    
    def get_discretisation(self, bins=None):
        upper_bounds = self.env.observation_space().high
        lower_bounds = self.env.observation_space().low
        
        if not bins:
            bins = tuple([5]*len(upper_bounds))

        self._create_uniform_grid(lower_bounds,upper_bounds, bins)
        self.lower_discrete = self._discretise(lower_bounds)
        self.upper_discrete = self._discretise(upper_bounds)
        
    def get_all_arangements(self):
        #TO DO SET UP BINS CHOICE FOR DISCRETISATION
        self.get_discretisation()
        arangements = []
        for i in range(0,len(self.lower_discrete)):
            arangements.append(np.arange(self.lower_discrete[i]-1,self.upper_discrete[i])+1)

        import itertools
        listOLists = arangements
        for coords in itertools.product(*listOLists):
            self.all_coordinates.append(tuple(coords))
            
        self.all_coordinates = self.all_coordinates
        
    def get_coordinates(self):
        return self.all_coordinates
    
    def mapper(self, observation):
        return self._discretise(observation)