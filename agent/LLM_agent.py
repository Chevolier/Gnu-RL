import sys
# sys.path.append("./")

import gym
import eplus_env

import pandas as pd
import pickle
import numpy as np

from utils import make_dict

import boto3
import json
bedrock = boto3.client(service_name='bedrock-runtime')

def request_claude(prompt, system_prompt, version='3-sonnet-20240229-v1:0'):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512, 
        # "temperature": 0.1,
        # "top_p": 0.9,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
    })

    modelId = f'anthropic.claude-{version}'
    accept = 'application/json'
    contentType = 'application/json'

    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    # print(f"response_body: {response_body}")
    
    return response_body.get('content')[0]['text']

def request_claude2(prompt, version='v2'):
    body = json.dumps({
        "prompt": f"Human:{prompt}Assistant:",
        "max_tokens_to_sample": 512,
        "temperature": 0.1,
        "top_p": 0.9,
    })

    modelId = f'anthropic.claude-{version}'
    accept = 'application/json'
    contentType = 'application/json'

    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    
    return response_body.get('completion')

# test Claude 3 model
# prompt = "Who are you?"
# response = request_claude(prompt)
# print(f"prompt: {prompt}, response: {response}")

system_prompt = """
You are an HVAC controller agent, you would interacte with the EnergyPlus environment to control the indoor room temperature. Your main purpose
is to keep the indoor room temperature (St) as close to the indoor room set temperature (St_setpoint) as possible by adjusting the supply air temperature setpoint 
while obtaining the highest cumulative rewards. Mixed air temperature at time t is defined as MAt

1. State is defined as the indoor room temperature St. 
2. Action is defined as the supply air temperature setpoint At. 
3. Reward is defined as Rt = -0.5 * (St - St_setpoint) ** 2 - abs(At - MAt)

You will be given a history of the form [[S0, A0, R0, S0_setpoint, MA0], [S1, A1, R1, S1_setpoint, MA1], ..., [St-1, At-1, Rt-1, St-1_setpoint, MAt-1], [St, None, None, St_setpoint, MAt]], try to find the pattern in the history and take the best action At based on the history and the current state, setpoint, etc., to maximize the reward. Besides, please respond to the abrupt change of St_setpoint as quickly as possible.

For instance, you are given the following history:

History: [[15.1, 24.5, -5.3, 22.2, 16.7], [16.2, 24.8, -4.2, 22.1, 17.5], [18.2, None, None, 22.7, 19.9]]
Action: 

Do not return any analysis, only return the value of Action

23.2

"""

# Create Environment. Follow the documentation of 'Gym-Eplus' to set up additional EnergyPlus simulation environment.
# env = gym.make('5Zone-sim_TMY2-v0');
# env = gym.make('5Zone-sim_TMY3-v0');
env = gym.make('5Zone-control_TMY2-v0')

# Modify here: Outputs from EnergyPlus; Match the variables.cfg file.
obs_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Diff. Solar Rad.", "Direct Solar Rad.", "Htg SP", "Clg SP", "Indoor Temp.", "Indoor Temp. Setpoint", "PPD", "Occupancy Flag", "Coil Power", "HVAC Power", "Sys In Temp.", "Sys In Mdot", "OA Temp.", "OA Mdot", "MA Temp.", "MA Mdot", "Sys Out Temp.", "Sys Out Mdot"]
dist_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Diff. Solar Rad.", "Direct Solar Rad.", "Indoor Temp. Setpoint", "Occupancy Flag"]

# Reset the env (creat the EnergyPlus subprocess)
timeStep, obs, isTerminal = env.reset();
obs_dict = make_dict(obs_name, obs)
start_time = pd.datetime(year = env.start_year, month = 3, day = 5)
print(start_time)

timeStamp = [start_time]
observations = [obs]
actions = []

claude_version = 3
num_hist_steps = 1 # only use the latest num_steps in the history
num_steps = 9*96
history = [[round(obs_dict['Indoor Temp.'], 1), None, None, round(obs_dict['Indoor Temp. Setpoint'], 1), round(obs_dict['MA Temp.'], 1)]]
for i in range(num_steps):
    # Using EnergyPlus default control strategy;
    
    history = history[-num_hist_steps:]
    prompt = f"""
        History: {history}
        Action: 
    """
    
    # print(f"prompt: {prompt}")
    try:
        if claude_version == 2:
            response = request_claude2(system_prompt+prompt)
        elif claude_version == 3:
            response = request_claude(prompt, system_prompt)
            
        # print(f"response: {response}")
        action = float(response.strip())
        # print(f"action_value: {action}, type: {type(action)}")
    except Exception as e:
        print(f"Error: {e}")
        continue
    
    timeStep, obs, isTerminal = env.step([action])
    obs_dict = make_dict(obs_name, obs)
    cur_time = start_time + pd.Timedelta(seconds = timeStep)
    reward = -0.5 * (obs_dict['Indoor Temp.']-obs_dict['Indoor Temp. Setpoint'])**2 - abs(action-obs_dict['MA Temp.'])
    history[-1][1] = round(action, 1)
    history[-1][2] = round(reward, 1)
    states_new = [round(obs_dict['Indoor Temp.'], 1), None, None, round(obs_dict['Indoor Temp. Setpoint'], 1), round(obs_dict['MA Temp.'], 1)]
    history.append(states_new)
    
    print("{}:  Sys Out: {:.2f}({:.2f})-> OA: {:.2f}({:.2f})-> MA: {:.2f}({:.2f})-> Sys Out: {:.2f}({:.2f})-> Zone Temp: {:.2f}-> Indoor Temp. Setpoint: {:.2f}-> Action: {:.2f}, Reward: {:.2f}".format(cur_time, obs_dict["Sys In Temp."], obs_dict["Sys In Mdot"],obs_dict["OA Temp."], obs_dict["OA Mdot"],
                                                    obs_dict["MA Temp."], obs_dict["MA Mdot"], obs_dict["Sys Out Temp."], obs_dict["Sys Out Mdot"],
                                                    obs_dict["Indoor Temp."], obs_dict['Indoor Temp. Setpoint'], action, reward))

    timeStamp.append(cur_time)
    observations.append(obs)
    #actions.append(action)

# Save Observations
obs_df = pd.DataFrame(np.array(observations), index = np.array(timeStamp), columns = obs_name)
dist_df = obs_df[dist_name]
# obs_df.to_csv("results/Sim-TMY2.csv")
obs_df.to_pickle(f"results/Sim-TMY2-llm-{num_hist_steps}-claude{claude_version}.pkl")
# dist_df.to_csv("results/Dist-TMY2.csv")
dist_df.to_pickle(f"results/Dist-TMY2-llm-{num_hist_steps}-claude{claude_version}.pkl")
print("Saved!")

env.end_env() # Safe termination of the environment after use.
