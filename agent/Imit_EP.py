import os
import sys

import gym
import eplus_env

# Assign mpc_path to be the file path where mpc.torch is located.
mpc_path = os.path.abspath(os.path.join(__file__, '..', '..' ))
sys.path.insert(0, mpc_path)

import argparse
from numpy import genfromtxt
import numpy as np
import pickle
import pandas as pd

from diff_mpc import mpc
from diff_mpc.mpc import QuadCost, LinDx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import make_dict, R_func

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

parser = argparse.ArgumentParser(description='GruRL-Imitation Learning')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--lr', type=float, default=5e-4, metavar='G',
                    help='Learning Rate')
parser.add_argument('--T', type=int, default=12, metavar='N',
                    help='Planning Horizon (default: 12)')
parser.add_argument('--step', type=int, default=900, metavar='N',
                    help='Time Step in Simulation, Unit in Seconds (default: 900)') # 15 Minutes Now!
parser.add_argument('--eta', type=int, default=5,
                    help='Hyper Parameter for Balancing Comfort and Energy')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Size of Mini-batch')
parser.add_argument('--save_name', type=str, default='rl',
                    help='save name')
args = parser.parse_args()

torch.manual_seed(args.seed)

# Modify here: Outputs from EnergyPlus; Match the variables.cfg file.
obs_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Diff. Solar Rad.", "Direct Solar Rad.", "Htg SP", "Clg SP", "Indoor Temp.", "Indoor Temp. Setpoint", "PPD", "Occupancy Flag", "Coil Power", "HVAC Power", "Sys In Temp.", "Sys In Mdot", "OA Temp.", "OA Mdot", "MA Temp.", "MA Mdot", "Sys Out Temp.", "Sys Out Mdot"]

# Modify here: Change based on the specific control problem
state_name = ["Indoor Temp."]
dist_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Diff. Solar Rad.", "Direct Solar Rad.", "Occupancy Flag"]
# Caveat: The RL agent controls the difference between Supply Air Temp. and Mixed Air Temp., i.e. the amount of heating from the heating coil. But, the E+ expects Supply Air Temp. Setpoint.
ctrl_name = ["Delta T"]
target_name = ["Indoor Temp. Setpoint"]

n_state = len(state_name)
n_ctrl = len(ctrl_name)
n_dist = len(dist_name)

eta = [0.1, args.eta] # eta: Weight for comfort during unoccupied and occupied mode
step = args.step # step: Timestep; Unit in seconds
T = args.T # T: Number of timesteps in the planning horizon
tol_eps = 90 # tol_eps: Total number of episodes; Each episode is a natural day

# Read Historical Data
dataset = pd.read_pickle("results/Sim-TMY2.pkl")
target = dataset[target_name]
disturbance = dataset[dist_name]
# Min-Max Normalization
disturbance = (disturbance-disturbance.min())/(disturbance.max()-disturbance.min())

dataset["Delta T"] = dataset["Sys Out Temp."]-dataset["MA Temp."]

# Train-Test Split
n_samples = len(dataset)
n_train = int(0.7*n_samples)
n_test = n_samples - n_train
train_set = dataset[:n_train]
test_set = dataset[n_train:]

class Learner():
    def __init__(self, n_state, n_ctrl, n_dist, disturbance, target, u_upper, u_lower):
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.n_dist = n_dist
        self.disturbance = disturbance
        self.target = target
        
        # My Initial Guess
        self.F_hat = torch.ones((self.n_state, self.n_state+self.n_ctrl))
        self.F_hat[0, 0] = 0.9
        self.F_hat[0, 1] = 0.3
        self.F_hat = self.F_hat.double().requires_grad_()
        
        self.Bd_hat = np.random.rand(self.n_state, self.n_dist)
        self.Bd_hat = torch.tensor(self.Bd_hat).requires_grad_()
        
        self.optimizer = optim.Adam([self.F_hat, self.Bd_hat], lr=args.lr)
    
        self.u_lower = u_lower * torch.ones(T, 1, n_ctrl).double()
        self.u_upper = u_upper * torch.ones(T, 1, n_ctrl).double()
    
    def Cost_function(self, cur_time):
        diag = torch.zeros(T, self.n_state + self.n_ctrl)
        occupied = self.disturbance["Occupancy Flag"][cur_time:cur_time + pd.Timedelta(seconds = (T-1) * step)]
        occupied = np.array(occupied)
        if len(occupied)<T:
            occupied = np.pad(occupied, ((0, T-len(occupied)), ), 'edge')
        eta_w_flag = torch.tensor([eta[int(flag)] for flag in occupied]).unsqueeze(1).double() # Tx1
        diag[:, :n_state] = eta_w_flag
        diag[:, n_state:] = 0.001
        
        C = []
        for i in range(T):
            C.append(torch.diag(diag[i]))
        C = torch.stack(C).unsqueeze(1) # T x 1 x (m+n) x (m+n)
        
        x_target = self.target[cur_time : cur_time + pd.Timedelta(seconds = (T-1) * step)] # in pd.Series
        x_target = np.array(x_target)
        if len(x_target)<T:
            x_target = np.pad(x_target, ((0, T-len(x_target)), (0, 0)), 'edge')
        x_target = torch.tensor(x_target)
        
        c = torch.zeros(T, self.n_state+self.n_ctrl) # T x (m+n)
        c[:, :n_state] = -eta_w_flag*x_target
        c[:, n_state:] = 1 # L1-norm now! Check
        c = c.unsqueeze(1) # T x 1 x (m+n)
        return C, c
    
    def forward(self, x_init, C, c, cur_time):
        dt = np.array(self.disturbance[cur_time : cur_time + pd.Timedelta(seconds = (T-2) * step)]) # T-1 x n_dist
        if len(dt)<T-1:
            dt = np.pad(dt, ((0, T-1-len(dt)), (0, 0)), 'edge')
        dt = torch.tensor(dt).transpose(0, 1) # n_dist x T-1
        
        ft = torch.mm(self.Bd_hat, dt).transpose(0, 1) # T-1 x n_state
        ft = ft.unsqueeze(1) # T-1 x 1 x n_state
        
        x_pred, u_pred, _ = mpc.MPC(n_state=self.n_state, 
                                    n_ctrl=self.n_ctrl,
                                    T=T,
                                    u_lower = self.u_lower,
                                    u_upper = self.u_upper,
                                    lqr_iter=20,
                                    verbose=0,
                                    exit_unconverged=False,
                                    )(x_init, QuadCost(C.double(), c.double()),
                                      LinDx(self.F_hat.repeat(T-1, 1, 1, 1),  ft))
        
        return x_pred[1, 0, :], u_pred[0, 0, :] # Dim.
    
    def predict(self, x_init, action, cur_time):
        dt = np.array(self.disturbance.loc[cur_time]) # n_dist
        dt = torch.tensor(dt).unsqueeze(1) # n_dist x 1
        ft = torch.mm(self.Bd_hat, dt) # n_state x 1
        tau = torch.stack([x_init, action]) # (n_state + n_ctrl) x 1
        next_state  = torch.mm(self.F_hat, tau) + ft # n_state x 1
        return next_state
                                    
    def update_parameters(self, x_true, u_true, x_pred, u_pred):
        # Every thing in T x Dim.
        state_loss = torch.mean((x_true.double() - x_pred)**2)
        action_loss = torch.mean((u_true.double() - u_pred)**2)
        
        # Note: args.eta balances the importance between predicting states and predicting actions
        traj_loss = args.eta*state_loss + action_loss
        print("From state {}, From action {}".format(state_loss, action_loss))
        self.optimizer.zero_grad()
        traj_loss.backward()
        self.optimizer.step()
        print(self.F_hat)
        print(self.Bd_hat)
        return state_loss.detach(), action_loss.detach()
        
def evaluate_performance(x_true, u_true, x_pred, u_pred):
    state_loss = torch.mean((x_true.double() - x_pred)**2)
    action_loss = torch.mean((u_true.double() - u_pred)**2)
    return state_loss, action_loss

def main():
    dir = 'results'
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    perf = []
    n_step = 96 # n_step: Number of Steps per Day
    numOfEpoches = 20
    
    timeStamp = []
    record_name =["Learner nState", "Expert nState", "Learner action", "Expert action"]
    losses = []
    losses_name = ["train_state_loss", "train_action_loss", "val_state_loss", "val_action_loss"]
    
    # Initialize the learner
    u_upper = 5
    u_lower = 0
    learner = Learner(n_state, n_ctrl, n_dist, disturbance, target, u_upper, u_lower)
     
    for epoch in range(numOfEpoches):
        x_true = []
        u_true = []
        x_pred = []
        u_pred = []
        
        train_state_loss = []
        train_action_loss = []
        for i in range(n_train): # By number of entries in the historical data
            idx = np.random.randint(n_train)
            cur_time = train_set.index[idx]
           
            expert_moves = train_set[cur_time:cur_time+pd.Timedelta(seconds = step)]
            if len(expert_moves)<2:
                print(cur_time)
                continue
            
            expert_state = torch.tensor(expert_moves[state_name].values).reshape(-1, n_state) # 2 x n_state
            expert_action = torch.tensor(expert_moves[ctrl_name].values).reshape(-1, n_ctrl) # 2 x n_ctrl
            x_true.append(expert_state[-1])
            u_true.append(expert_action[0])

            obs = train_set.loc[cur_time]
            x_init = torch.tensor(np.array([obs[name] for name in state_name])).unsqueeze(0) # n_batch x n_state, i.e. 1 x n_state
            C, c = learner.Cost_function(cur_time)
            learner_state, learner_action = learner.forward(x_init, C, c, cur_time)
                
            # Predict next state based on expert's action
            next_state = learner.predict(x_init.squeeze(0), expert_action[0], cur_time)
            x_pred.append(next_state)
            u_pred.append(learner_action)
            
            if (i % args.batch_size == 0) & (i>0):
                x_true = torch.stack(x_true).reshape(-1, n_state)
                u_true = torch.stack(u_true).reshape(-1, n_ctrl)
                x_pred = torch.stack(x_pred).reshape(-1, n_state)
                u_pred = torch.stack(u_pred).reshape(-1, n_ctrl)
                b_state_loss, b_action_loss = learner.update_parameters(x_true, u_true, x_pred, u_pred)
                train_state_loss.append(b_state_loss)
                train_action_loss.append(b_action_loss)
                x_true = []
                u_true = []
                x_pred = []
                u_pred = []

        # Evaluate performance at the end of each epoch
        x_true = []
        u_true = []
        x_pred = []
        u_pred = []
        timeStamp = []
        for idx in range(n_test):
            cur_time = test_set.index[idx]
            expert_moves = test_set[cur_time:cur_time+pd.Timedelta(seconds = step)]
            if len(expert_moves)<2:
                print(cur_time)
                continue
            expert_state = torch.tensor(expert_moves[state_name].values).reshape(-1, n_state) # 2 x n_state
            expert_action = torch.tensor(expert_moves[ctrl_name].values).reshape(-1, n_ctrl) # 2 x n_ctrl
            x_true.append(expert_state[-1])
            u_true.append(expert_action[0])
            
            timeStamp.append(cur_time+pd.Timedelta(seconds = step))
            
            obs = test_set.loc[cur_time]
            x_init = torch.tensor(np.array([obs[name] for name in state_name])).unsqueeze(0) # 1 x n_state
            C, c = learner.Cost_function(cur_time)
            learner_state, learner_action = learner.forward(x_init, C, c, cur_time)
            next_state = learner.predict(x_init.squeeze(0), expert_action[0], cur_time)
            x_pred.append(next_state.detach())
            u_pred.append(learner_action.detach())
            
        x_true = torch.stack(x_true).reshape(-1, n_state)
        u_true = torch.stack(u_true).reshape(-1, n_ctrl)
        x_pred = torch.stack(x_pred).reshape(-1, n_state)
        u_pred = torch.stack(u_pred).reshape(-1, n_ctrl)
        val_state_loss, val_action_loss = evaluate_performance(x_true, u_true, x_pred, u_pred)
        print("At Epoch {0}, the loss from the state is {1} and from the action is {2}".format(epoch, val_state_loss, val_action_loss))
        losses.append((np.mean(train_state_loss), np.mean(train_action_loss), val_state_loss, val_action_loss))
       
        record = pd.DataFrame(torch.cat((x_pred, x_true, u_pred, u_true), dim = 1).numpy(), index = np.array(timeStamp), columns = record_name)
        record_df = pd.DataFrame(np.array(record), index = np.array(timeStamp), columns = record_name)
        record_df.to_pickle("results/Imit_{}_{}.pkl".format(args.save_name, epoch))
        
        # Save weights
        F_hat = learner.F_hat.detach().numpy()
        Bd_hat = learner.Bd_hat.detach().numpy()
        np.save("results/weights/F-{}.npy".format(epoch), F_hat)
        np.save("results/weights/Bd-{}.npy".format(epoch), Bd_hat)
        
    # Save losses at each epoch
    losses_df = pd.DataFrame(np.array(losses), index = np.arange(numOfEpoches), columns = losses_name)
    losses_df.to_pickle("results/Imit_loss_"+args.save_name+".pkl")
    
if __name__ == '__main__':
    main()

