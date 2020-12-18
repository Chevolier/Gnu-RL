# Gnu-RL

This is the companion code repository for [Gnu-RL: A Precocial Reinforcement Learning Solution for Building HVAC Control Using a Differentiable MPC Policy](https://dl.acm.org/citation.cfm?id=3360849). If our paper is helpful to your research, cite it using the following reference:

```
@inproceedings{chen2019gnu,
  title={Gnu-RL: A Precocial Reinforcement Learning Solution for Building HVAC Control Using a Differentiable MPC Policy},
  author={Chen, Bingqing and Cai, Zicheng and Berg{\'e}s, Mario},
  booktitle={Proceedings of the 6th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation},
  pages={316--325},
  year={2019},
  organization={ACM}
}
```

### Description
Gnu-RL is a novel approach that enables practical deployment of reinforcement learning (RL) for heating, ventilation, and air conditioning (HVAC) control and requires no prior information other than historical data from existing HVAC controllers. 

Prior to any interaction with the environment, a Gnu-RL agent is pre-trained on historical data using imitation learning, which enables it to match the behavior of the existing controller. Once it is put in charge of controlling the environment, the agent continues to improve its policy end-to-end, using a policy gradient algorithm.

![Framework](imgs/framework.png)

Specifically, Gnu-RL adopts a recently-developed [Differentiable Model Predictive Control (MPC)](http://papers.nips.cc/paper/8050-differentiable-mpc-for-end-to-end-planning-and-control.pdf) policy, which encodes domain knowledge on planning and system dynamics, making it both sample-efficient and interpretable. 

![policy](imgs/policy.png)

### Install Required Packages
The following two packages were used.    
- [Gym-Eplus](https://github.com/zhangzhizza/Gym-Eplus)
    - This package is an OpenGym AI wrapper for EnergyPlus. 
    - Install the package following its [documentation](https://github.com/zhangzhizza/Gym-Eplus).
    - While the documentation of this repo specifies EnergyPlus version 8.6, but the Gym-plus package is applicable to any EnergyPlus version 8.x. 
- [mpc.torch](https://github.com/locuslab/mpc.pytorch)
    - This package is a fast and differentiable model predictive control solver for PyTorch.
    - The required files are already placed under ./diff_mpc. No installation is necessary.

Install other packages by, 
```
$ pip install -r requirements.txt
``` 

### Register Simulation Environments
We demonstrate Gnu-RL in an EnergyPlus model. Check [here](Demo.ipynb) for details.

To set up the co-simulation environment with EnergyPlus: 
- Read the documentation of [Gym-Eplus](https://github.com/zhangzhizza/Gym-Eplus) on registering simulation environments. 
- All the EnergyPlus model files used in the demo are placed under ./eplus_env.
- Register the environments following this table. A  *\_\_init\_\_.py* file for registration is included. Place it under *Gym-Eplus/eplus_env/*. 
- Place the model and weather files under *Gym-Eplus/eplus_env/envs/* at the corresponding location.  
- Check that the placement of model files and weather files match *\_\_init\_\_.py*.
- Finally, change line 24 in *Gym-Eplus/eplus_env/envs/eplus8_6.py* to  

  ```
  YEAR = 2017 # Non leap year
  ```


| **Environment Name** |**Model File (\*.idf)**|**Configuration File (\*.cfg)**|**Weather File (\*.epw)**| 
|:----------------|:---------------|:--------|:-----------|
|**5Zone-sim_TMY2-v0**|5Zone_Default.idf|variables_Default.cfg|pittsburgh_TMY2.epw|
|**5Zone-control_TMY3-v0**|5Zone_Control.idf|variables_Control.cfg|pittsburgh_TMY3.epw|
| **5Zone-sim_TMY3-v0**   | 5Zone_Default.idf|variables_Default.cfg|pittsburgh_TMY3.epw|


### Files

To generate "historical" data from baseline controller, 

```
$ python simulate.py
```

For **Offline Pretraining**, 
```
$ python Imit_EP.py
```

For **Online Learning**, 
```
$ python PPO_MPC_EP.py
``` 

### Contact
- [Bingqing Chen](mailto:bingqinc@andrew.cmu.edu), PhD Candidate at Carnegie Mellon University, Department of Civil and Environmental Engineering, Intelligent Infrastructure Research Laboratory (INFERLab).
- [Mario Berges](mailto:marioberges@cmu.edu), Professor at Carnegie Mellon University, Department of Civil and Environmental Engineering, INFERLab

### License

The MIT License (MIT) Copyright (c) 2019, Bingqing Chen. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
