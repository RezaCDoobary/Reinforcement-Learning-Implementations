import numpy as np

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
        import pybulletgym
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
    def __init__(self, env):
        self.env = env
        self.state_maps = {}
        self.action_maps = {}
        self.inverse_action_maps = {}
        self.inverse_state_maps = {}
        self.create_mappings()
           
    def find_coordinates(self, space):
        if isinstance(space, self.env.gym.spaces.tuple_space.Tuple):
            space_description = [np.arange(coordinate_space.n) for coordinate_space in space.spaces]
            import itertools
            r = ''
            for i in range(0,len(space_description)):
                r+='space_description['+str(i)+'],'
            r = 'list(itertools.product(' + r[:-1] + '))'
            coordinates = eval(r)
        else:
            coordinates = np.arange(space.n)
        return coordinates
    
    def create_mappings(self):
        action_coordinates = self.find_coordinates(self.env.action_space())
        state_coordinates = self.find_coordinates(self.env.observation_space())
        for i in range(0,len(action_coordinates)):
            self.action_maps[action_coordinates[i]] = i
            self.inverse_action_maps[i] = action_coordinates[i]
            
        for i in range(0,len(state_coordinates)):
            self.state_maps[state_coordinates[i]] = i
            self.inverse_state_maps[i] = state_coordinates[i] 
    
    def step(self, action):
        return self.env.step(self.inverse_action_maps[action])
        
    def reset(self):
        return self.state_maps[self.env.reset()]
        
    def render(self):
        self.env.render()
        
    def action_size(self):
        return len(self.action_maps)
        
    def close(self):
        self.env.close()