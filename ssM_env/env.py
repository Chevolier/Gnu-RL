import numpy as np

class ssModel():
    def __init__(self, ssMatrix, x_init = 24, x_target = 24, eta = 10):
        self.A = ssMatrix['A']
        self.Bd = ssMatrix['Bd']
        self.Bu = ssMatrix['Bu'] # n_state x n_ctrl
        self.C = ssMatrix['C'] # n_obs x n_state
        self.d = ssMatrix['d']
        
        ## Reshape for 1I1O system
        if len(self.Bu.shape)==1:
            self.Bu = self.Bu.reshape(-1, 1)
        
        if len(self.C.shape)==1:
            self.C = self.C.reshape(1, -1)
        
        self.n_state = self.A.shape[0]
        self.n_observation = self.C.shape[0]
        self.n_action = self.Bu.shape[1]
        self.n_disturbance = self.d.shape[0]
        
        self.max_steps = self.d.shape[1]
        self.timestep = 0
        
        self.x_init = np.ones(self.n_state) * x_init
        self.state = self.x_init
        self.x_target = x_target
        self.eta = [0.1, eta]
        
    def reset(self, t=0):
        # Initialize State
        self.state = self.x_init
        self.timestep = t
        obs = self.C.dot(self.state)
        return np.concatenate([obs, self.d[:, self.timestep]])
        
    def step(self, u, occupied = 1, obs_state=False):
        next_state = self.A.dot(self.state)+self.Bd.dot(self.d[:, self.timestep]) + self.Bu.dot(u)
        if self.timestep < self.max_steps-1:
            self.timestep += 1
        else:
            self.timestep = 0
        next_obs = self.C.dot(next_state)
        self.state = next_state
        
        reward = - self.eta[int(occupied)] * np.mean((next_obs-self.x_target)**2) - np.mean(u**2)
    
        ## Concatenate disturbance from the next time-step as part of the observation
        next_obs = np.concatenate([next_obs, self.d[:, self.timestep]])
        
        if obs_state:
            return next_obs, reward, next_state
        else:
            return next_obs, reward
