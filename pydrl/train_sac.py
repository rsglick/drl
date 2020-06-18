import time
import os
import sys
import math
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC, TD3


from onedof import onedof
from onedof import random_agent
#random_agent(render=True, ctrl=1)

#all_envs = [i for i in gym.envs.registry.all()]
env      = onedof()
eval_env = onedof()
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)

# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps): # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps
        
    def __enter__(self): # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)
            
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb): # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=100, 
                                                          verbose=1)        
eval_callback = EvalCallback(eval_env, 
                             render=False,
                             best_model_save_path="./",
                             deterministic=True,
                             callback_on_new_best=reward_threshold_callback,
                             #log_path=log_dir, 
                             verbose=1,
                             n_eval_episodes=5,
                             eval_freq=1000)

hyperparams_dict = {
    'learning_rate': 3.0e-4,
    'buffer_size': 1000000,
    'gamma': 0.99,
    'batch_size':10000,  #256,
    'tau': 0.005,
    'device':'cuda',
    'seed':0,
    'target_entropy':"auto",
    'policy_kwargs':dict(net_arch=[8, 8]),
}

model = SAC('MlpPolicy', 
            env, 
            tensorboard_log='.',
            **hyperparams_dict,
            )

# Train
model = model.load("./modelSAC")
total_timesteps = 100000
with ProgressBarManager(total_timesteps) as progress_callback:
    callback = CallbackList([progress_callback,
                             eval_callback])
    model.learn(total_timesteps=total_timesteps, 
                callback=callback,
               )
model.save("./modelSAC")

#
# Evaluate Model
#
saved_acts = []
saved_obs = []
saved_pos = []
saved_vel = []
saved_acc = []
saved_steps = []
saved_epRollingReward = []
saved_ep_num = []
saved_epTotalrewards = []
saved_eplen = []
epMissDistance = []

renderFlag = 0
eplen = 0
epRollingReward = 0
num_episodes = 0
max_eps = 100
obs = env.reset()
timesteps = 100000
hit_counter = 0
pbar = tqdm(total=timesteps)

for i in range(timesteps):
    pbar.update()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = eval_env.step(action)

    if renderFlag:
        eval_env.render()
    
    epRollingReward += rewards
    eplen += 1

    saved_acts.append(action * eval_env.ctrl_limit) 
    saved_obs.append(obs) 
    saved_pos.append(obs[0]) 
    saved_vel.append(obs[1]) 
    saved_acc.append(obs[2]) 
    saved_steps.append(i)
    saved_epRollingReward.append(epRollingReward)

    if dones:
        epMissDistance.append(info["delta_position"])
        if abs(info["delta_position"]) < 0.05:
            hit_counter += 1

        pbar.write(f"Ep{num_episodes} EpReward = {epRollingReward:.2f},\
                EpLen = {eplen}, MissDis = {info['delta_position']}, Hits = {hit_counter}",)


        saved_ep_num.append(num_episodes)
        num_episodes += 1 

        saved_eplen.append(eplen)
        saved_epTotalrewards.append(epRollingReward)
        eplen = 0
        epRollingReward = 0
        eval_env.reset()

        if num_episodes == max_eps:
            pbar.write("Reaching Max Eps")
            break

eval_env.close()
mean_reward = np.mean(saved_epTotalrewards)
std_reward  = np.std(saved_epTotalrewards)
mean_miss   = np.mean(epMissDistance)
hit_ratio   = hit_counter / len(epMissDistance)
print(f"Total Eps({num_episodes}): mean_reward = {mean_reward:.2f} +/- {std_reward:.2f}, Hit%:{hit_ratio}")

fig, ax = plt.subplots(4, sharex=True)
ax[0].plot( saved_steps, [ob[0] for ob in saved_obs] )
ax[0].set_title("Position")
ax[1].plot( saved_steps, [ob[1] for ob in saved_obs] )
ax[1].set_title("Velocity")
ax[2].plot( saved_steps, [ob[2] for ob in saved_obs] )
ax[2].set_title("Acceleration")
ax[3].plot( saved_steps, saved_acts )
ax[3].set_title("Cmd Accel")
fig.savefig("states.png")


fig, ax = plt.subplots(3, sharex=True)
ax[0].plot( saved_ep_num, saved_epTotalrewards)
ax[0].set_title("epRollingrewards")
ax[1].plot( saved_ep_num, saved_eplen )
ax[1].set_title("saved_eplen")
ax[2].plot( saved_ep_num, epMissDistance)
ax[2].set_title(f"epMissDistance (Hit Ratio: {hit_ratio})")
fig.savefig("rewards.png")


obs_in_ep = np.split(np.array(saved_obs), max_eps)
pos_in_ep = np.split(np.array(saved_pos), max_eps)
vel_in_ep = np.split(np.array(saved_vel), max_eps)
acc_in_ep = np.split(np.array(saved_acc), max_eps)
act_in_ep = np.split(np.array(saved_acts), max_eps)
rew_in_ep = np.split(np.array(saved_epRollingReward), max_eps)
steps     = np.linspace(0, 1, 100)

fig, ax = plt.subplots(1, sharex=True)
for i in range(len(pos_in_ep)):
    ax.plot( steps, pos_in_ep[i] )
    ax.plot( steps, - 0.05 * np.ones_like(steps), '--r' )
    ax.plot( steps,   0.05 * np.ones_like(steps), '--r' )
ax.set_title("Position")
fig.savefig("pos_by_ep.png")

fig, ax = plt.subplots(1, sharex=True)
for i in range(len(acc_in_ep)):
    ax.plot( steps, acc_in_ep[i] )
ax.set_title("Acceleration")
fig.savefig("acc_by_ep.png")


fig, ax = plt.subplots(4, sharex=True)
for i in range(len(pos_in_ep)):
    ax[0].plot( steps, pos_in_ep[i] )
    ax[1].plot( steps, vel_in_ep[i] )
    ax[2].plot( steps, acc_in_ep[i] )
    ax[3].plot( steps, act_in_ep[i] )
ax[0].set_title("Position")
ax[1].set_title("Velocity")
ax[2].set_title("Acceleration")
fig.savefig("states_by_ep.png")

fig, ax = plt.subplots(1, sharex=True)
for i in range(len(pos_in_ep)):
    ax.plot( steps, rew_in_ep[i] )
ax.set_title("Ep Reward")
fig.savefig("reward_by_ep.png")


#if __name__ == "__main__":
#    main()
