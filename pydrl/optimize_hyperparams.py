#!/usr/bin/env python

import os
import time
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import yaml
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import optuna
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner
from optuna.samplers import RandomSampler, TPESampler
from optuna.integration.skopt import SkoptSampler


# Stable Baselines3 
#  https://github.com/DLR-RM/stable-baselines3
from stable_baselines3 import PPO, SAC, TD3

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList


# Setup Environment
ENV_LIST = [
    "LunarLanderContinuous-v2",
    "MountainCarContinuous-v0",    
    "Pendulum-v0",    
    "BipedalWalker-v3",
    "BipedalWalkerHardcore-v3",   
    "CarRacing-v0",
]
# all_envs = [i for i in gym.envs.registry.all()]

tmp_dict = {}
for i in ENV_LIST:
    tmp_env = gym.make(i)
    tmp_dict[i] = {
        "env_name":i,
        "action_space":tmp_env.action_space,
        "action_space_high":tmp_env.action_space.high[0],
        "action_space_low":tmp_env.action_space.low[0],
        "observation_space":tmp_env.observation_space,
        "max_episode_steps":gym.envs.registry.env_specs[i].max_episode_steps,
        "reward_threshold":gym.envs.registry.env_specs[i].reward_threshold,
    }
ENV_DF = pd.DataFrame(tmp_dict)
print(ENV_DF.T)
# print(ENV_DF["BipedalWalker-v3"])

ALGORITHMS = {
    'PPO': PPO,
    'SAC': SAC,
    'TD3': TD3,
}



parser = argparse.ArgumentParser()
parser.add_argument('-a','--algorithm_name', help='RL Algorithm', default='SAC',
                    type=str, required=False,) # choices=list(ALGORITHMS.keys()))
parser.add_argument('-e','--env_name', type=str, default="LunarLanderContinuous-v2", help='environment ID')
parser.add_argument('--eval_freq', help='Evaluate the agent every n steps (if negative, no evaluation)',
                    default=1000, type=int)
parser.add_argument('-ne','--n_eval_episodes', help='Number of episodes to use for evaluation',
                    default=5, type=int)
parser.add_argument('-nt','--n_trials', help='Number of trials for optimizing hyperparameters', type=int, default=10)
parser.add_argument('-nj','--n_jobs', help='Number of parallel jobs when optimizing hyperparameters', type=int, default=1)
parser.add_argument('-s','--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('-sq','--sqlite_file', type=str, default=None, help='sqlite file')
#parser.add_argument('--sampler', help='Sampler to use when optimizing hyperparameters', type=str,
#                    default='tpe', choices=['random', 'tpe', 'skopt'])
#parser.add_argument('--pruner', help='Pruner to use when optimizing hyperparameters', type=str,
#                    default='median', choices=['halving', 'median', 'none'])
#parser.add_argument('--n-startup-trials', help='Number of trials before using optuna sampler',
#                    type=int, default=10)
parser.add_argument('-t','--total_timesteps', help='Number of evaluations for hyperparameter optimization',
                    type=int, default=2000)
args = parser.parse_args()

#
# Inputs
#
# env_name       = ENV_DF["BipedalWalkerHardcore-v3"]["env_name"]
env_name       = args.env_name #"LunarLanderContinuous-v2"
algorithm_name = args.algorithm_name

eval_freq= args.eval_freq
n_eval_episodes = args.n_eval_episodes
n_jobs= args.n_jobs
n_trials= args.n_trials
total_timesteps= args.total_timesteps
seed=args.seed
sqlite_file = args.sqlite_file

timestr = time.strftime("%Y%m%d-%H%M")
name_prefix = f"{algorithm_name}_{env_name}"
log_dir = f"./runs/{env_name}/{timestr}"
optim_hyperparam_file = f"{log_dir}/{name_prefix}_optimTrials.yaml"
optim_csv_file = f"{log_dir}/{name_prefix}_optimTrials.csv"

if sqlite_file is None:
    sqlite_file = f"sqlite:///{log_dir}/{name_prefix}_sqlite.db"

print(f"SQLITE File: {sqlite_file} ")

os.makedirs(log_dir, exist_ok=True)

env = gym.make(env_name)
eval_env = gym.make(env_name)


set_random_seed(seed)



def sample_sac_params(trial):
    """
    Sampler for SAC hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical('gamma', [0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512])
    buffer_size = trial.suggest_categorical('buffer_size', [int(1e4), int(1e5), int(1e6)])
    learning_starts = trial.suggest_categorical('learning_starts', [0, 1000, 10000, 20000])
    # train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
    train_freq = trial.suggest_categorical('train_freq', [8, 16, 32, 64, 128, 256, 512])
    # Polyak coeff
    tau = trial.suggest_categorical('tau', [0.001, 0.005, 0.01, 0.02])
    # gradient_steps takes too much time
    # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
    gradient_steps = train_freq
    # ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
    ent_coef = 'auto'
    log_std_init = trial.suggest_uniform('log_std_init', -4, 1)
    net_arch = trial.suggest_categorical('net_arch', [64, 128])
    #net_arch = trial.suggest_categorical('net_arch', [64, 128, 256])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])
    target_entropy = 'auto'
    # if ent_coef == 'auto':
    #     # target_entropy = trial.suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
    #     target_entropy = trial.suggest_uniform('target_entropy', -10, 10)

    return {
        'gamma': gamma,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'learning_starts': learning_starts,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps,
        'ent_coef': ent_coef,
        'tau': tau,
        'target_entropy': target_entropy,
        'policy_kwargs': dict(log_std_init=log_std_init, net_arch=[net_arch,net_arch])
    }

HYPERPARAMS_SAMPLER = {
#     'ppo': sample_ppo_params,
    'SAC': sample_sac_params,
#     'a2c': sample_a2c_params,
#     'td3': sample_td3_params
}


class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """
    def __init__(self, trial, eval_env=None, n_eval_episodes=5,
                 eval_freq=10000, deterministic=True, verbose=0):

        super(TrialEvalCallback, self).__init__(eval_env=eval_env, n_eval_episodes=n_eval_episodes,
                                                eval_freq=eval_freq,
                                                deterministic=deterministic,
                                                verbose=verbose)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super(TrialEvalCallback, self)._on_step()
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elasped time ?
            self.trial.report(-1 * self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune(self.eval_idx):
                self.is_pruned = True
                return False
        return True

# sampler = RandomSampler(seed=0)
sampler = TPESampler(n_startup_trials=10, seed=0)
# pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=20 // 3)




study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        #storage=sqlite_file,
        #load_if_exists=True,
        )

algo_sampler = HYPERPARAMS_SAMPLER[algorithm_name]
def objective(trial):
    #total_timesteps = 2000
    
    
    trial.model_class = None
    hyperparams_dict_opt = {} 
    hyperparams_dict_opt.update(algo_sampler(trial))
    
    

    env = gym.make(env_name)
    eval_env = gym.make(env_name)

    alogrithm = ALGORITHMS[algorithm_name]
    model = alogrithm('MlpPolicy', 
                      env,
                      **hyperparams_dict_opt)    
    model.trial = trial
    
    eval_optuna_callback = TrialEvalCallback(trial, eval_env=eval_env, n_eval_episodes=n_eval_episodes,
                                             eval_freq=eval_freq, deterministic=True)

    try:
        model.learn(total_timesteps=total_timesteps, 
                    callback=eval_optuna_callback)
        model.env.close()
        eval_env.close()
    except AssertionError as e:
        model.env.close()
        eval_env.close()
        print(e)
        raise optuna.exceptions.TrialPruned()
        
    is_pruned = eval_optuna_callback.is_pruned
    cost = eval_optuna_callback.last_mean_reward
    if is_pruned:
       raise optuna.exceptions.TrialPruned()
    return cost


#n_trials = 2
#n_jobs   = 1
study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

print('Number of finished trials: ', len(study.trials))

trial = study.best_trial
print('Best trial:', trial.number)

print('Value: ', trial.value)

print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

study_df = study.trials_dataframe()
study_df.to_csv(optim_csv_file)


new_hyperparams = trial.params
log_std_init = new_hyperparams["log_std_init"]
net_arch = new_hyperparams["net_arch"]
new_hyperparams["policy_kwargs"] = dict(
        log_std_init=log_std_init,
        net_arch=[net_arch,net_arch]
        )
del new_hyperparams["log_std_init"], new_hyperparams["net_arch"]

with open(optim_hyperparam_file, 'w') as f:
    print(f"Saving hyperparams: {optim_hyperparam_file}")
    f.write(f"#Environment: {env_name}\n")
    f.write(f"#Reward: {trial.value}\n")
    yaml.dump(trial.params, f)    

# Value:  -11.795852661132812
# Params:
#     gamma: 0.98
#     lr: 0.0038994519179077905
#     batch_size: 128
#     buffer_size: 1000000
#     learning_starts: 20000
#     train_freq: 8
#     tau: 0.005
#     log_std_init: -2.1291500983288723
#     net_arch: small



# Best trial: 14
# Value:  79.02980041503906
# Params: 
#     gamma: 0.95
#     lr: 0.00028407696725135895
#     batch_size: 64
#     buffer_size: 1000000
#     learning_starts: 0
#     train_freq: 64
#     tau: 0.02
#     log_std_init: 0.3062792340519811
#     net_arch: small





