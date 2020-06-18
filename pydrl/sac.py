import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import gym
import numpy as np
import random

import time
import os

import pickle
import yaml

from onedof import onedof
from replay_buffer import replay_buffer
from actor_critic import actor
from actor_critic import critic

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = False

# Env Seeds
#env = gym.make("MountainCarContinuous-v0")
#env = gym.make("Pendulum-v0")
env = onedof()
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

input_shape  = np.prod(env.observation_space.shape)
output_shape = np.prod(env.action_space.shape)


##############################################################################
#
# Hyperparams
# 
hyperparams = {
"buffer_size"     : int(1e6),
"batch_size"      : 256,
"learning_rate"   : 3.0e-4,
"learning_starts" : 1000,
"tau"             : 0.005,
"gamma"           : 0.99,
"hidden_size"     : 8,
}

total_timesteps = 100000
try:
    episode_length  = int(env.max_episode_steps)
except:
    episode_length  = env.spec.max_episode_steps



#writer = SummaryWriter(f"testing")
#writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
#        '\n'.join([f"|{key}|{value}|" for key, value in hyperparams.items()])))

# Replay Buffer
replay_buffer = replay_buffer(hyperparams["buffer_size"])

# Actor Network
actor = actor(env, input_shape, output_shape, hyperparams["hidden_size"], device ).to(device)

# Critic Networks
qf1 = critic(input_shape, output_shape, hyperparams["hidden_size"], device  ).to(device)
qf2 = critic(input_shape, output_shape, hyperparams["hidden_size"], device  ).to(device)
qf1_target = critic(input_shape, output_shape, hyperparams["hidden_size"], device  ).to(device)
qf2_target = critic(input_shape, output_shape, hyperparams["hidden_size"], device  ).to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())

# Entropy
init_alpha   = torch.Tensor([1.0]).to(device)
target_alpha = - np.prod(env.action_space.shape).astype(np.float32)
log_alpha    = torch.log(torch.ones(1, device=device) * init_alpha).requires_grad_(True)

values_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=hyperparams["learning_rate"])
policy_optimizer = optim.Adam(list(actor.parameters()), lr=hyperparams["learning_rate"])
alpha_optimizer  = optim.Adam([log_alpha], lr=hyperparams["learning_rate"])

loss_fn = nn.MSELoss()

def save():
    timestr = time.strftime("%Y%m%d_1")
    save_dir = f"./sac_trained_{timestr}"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(actor.state_dict(),            f'{save_dir}/actor.pth')
    torch.save(qf1.state_dict(),              f'{save_dir}/qf1.pth')
    torch.save(qf2.state_dict(),              f'{save_dir}/qf2.pth')
    torch.save(values_optimizer.state_dict(), f'{save_dir}/values_optimizer.pth')
    torch.save(policy_optimizer.state_dict(), f'{save_dir}/policy_optimizer.pth')
    torch.save(alpha_optimizer.state_dict(),  f'{save_dir}/alpha_optimizer.pth')

    with open( f'{save_dir}/replay_buffer.pth', 'wb') as f:
        pickle.dump(replay_buffer, f)

    with open( f'{save_dir}/hyperparams.yaml', 'w' ) as f:
        yaml.dump(hyperparams, f ) 


def load():
    #save_list = [actor,qf1,qf2,values_optimizer,policy_optimizer,alpha_optimizer]
    #for i in save_list:
    #    i.load_state_dict(torch.load(f"./{i}.pth"))
    #    i.eval()
    actor.load_state_dict(torch.load( f'{save_dir}/actor.pth' ) )
    qf1.load_state_dict(torch.load( f'{save_dir}/qf1.pth' ) )
    qf2.load_state_dict(torch.load( f'{save_dir}/qf2.pth' ) )
    values_optimizer.load_state_dict(torch.load( f'{save_dir}/values_optimizer.pth' ) )
    policy_optimizer.load_state_dict(torch.load( f'{save_dir}/policy_optimizer.pth' ) )
    alpha_optimizer.load_state_dict(torch.load( f'{save_dir}/alpha_optimizer.pth' ) )

    with open( f'{save_dir}/replay_buffer.pth', 'rb') as f:
        replay_buffer = pickle.load( f )  

    with open( f'{save_dir}/hyperparams.yaml', 'r' ) as f:
        hyperparams = yaml.load( f ) 

    actor.eval()
    qf1.eval()
    qf2.eval()
    values_optimizer.eval()
    policy_optimizer.eval()
    alpha_optimizer.eval()

def train():
    global_step = 0
    while global_step < total_timesteps:
        next_obs = np.array(env.reset())
        actions = np.empty((episode_length,), dtype=object)
        rewards, dones = np.zeros((2, episode_length))
        qf1_losses, qf2_losses, policy_losses = np.zeros((3, episode_length))
        qf_losses, = np.zeros((1, episode_length))
        obs = np.empty((episode_length,) + env.observation_space.shape)
        
        # ALGO LOGIC: put other storage logic here
        entropys = torch.zeros((episode_length,), device=device)
        alpha_losses = []
        alphas = []
        
        # TRY NOT TO MODIFY: prepare the execution of the game.
        for step in range(episode_length):
            global_step += 1
            obs[step] = next_obs.copy()
            
            # ALGO LOGIC: put action logic here
            action, log_prop, _ = actor.get_action(obs[step:step+1])
            actions[step] = action.tolist()[0]

            # Alpha Entropy Optimize
            alpha_loss = None
            alpha = torch.exp( log_alpha.detach())
            alpha_loss = - ( log_alpha * ( log_prop + target_alpha).detach()).mean()
            
            alpha_losses.append(alpha_loss.item())
            alphas.append(alpha.item())
            entropys[step] = alpha

            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

        
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards[step], dones[step], info = env.step(action.tolist()[0][0])
            replay_buffer.put((obs[step], actions[step], rewards[step], next_obs, dones[step]))
            next_obs = np.array(next_obs)
            # ALGO LOGIC: training.
            if len(replay_buffer.buffer) > hyperparams["learning_starts"]:
                s_obs, s_actions, s_rewards, s_next_obses, s_dones = replay_buffer.sample(hyperparams["batch_size"])
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(s_next_obses)
                    qf1_next_target = qf1_target.forward(s_next_obses, next_state_actions)
                    qf2_next_target = qf2_target.forward(s_next_obses, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * hyperparams["gamma"] * (min_qf_next_target).view(-1)

                qf1_a_values = qf1.forward(s_obs, torch.Tensor(s_actions).to(device)).view(-1)
                qf2_a_values = qf2.forward(s_obs, torch.Tensor(s_actions).to(device)).view(-1)
                # qf1, qf2 = critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
                #qf1_loss = loss_fn(qf1_a_values, next_q_value)
                #qf2_loss = loss_fn(qf2_a_values, next_q_value)
                qf_loss   = 0.5 *( loss_fn(qf1_a_values, next_q_value) + loss_fn(qf2_a_values, next_q_value) )

                values_optimizer.zero_grad()
                #qf1_loss.backward()
                qf_loss.backward()
                values_optimizer.step()

                #values_optimizer.zero_grad()
                #qf2_loss.backward()
                #values_optimizer.step()

                pi, log_pi, _ = actor.get_action(s_obs)
                qf1_pi = qf1.forward(s_obs, pi)
                qf2_pi = qf2.forward(s_obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)

                policy_loss = ((alpha * log_pi) - min_qf_pi).mean()
                
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # update the target network
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(hyperparams["tau"] * param.data + (1 - hyperparams["tau"]) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(hyperparams["tau"]* param.data + (1 - hyperparams["tau"]) * target_param.data)
                #qf1_losses[step] = qf1_loss.item()
                #qf2_losses[step] = qf2_loss.item()
                qf_losses[step] = qf_loss.item()
                policy_losses[step] = policy_loss.item()

            if dones[step]:
                break

        print(f"global_step={global_step},\t" 
              f"episode_reward={rewards.sum():.2f},\t" 
              f"Miss={info['delta_position']:.2f},\t" 
              f"Act_sum={np.array([a[0] for a in actions]).sum():.2f}")
        #writer.add_scalar("charts/episode_reward", rewards.sum(), global_step)
        #writer.add_scalar("losses/entropy", entropys[:step+1].mean().item(), global_step)
        ##writer.add_scalar("losses/soft_q_value_1_loss", qf1_losses[:step+1].mean(), global_step)
        ##writer.add_scalar("losses/soft_q_value_2_loss", qf2_losses[:step+1].mean(), global_step)
        #writer.add_scalar("losses/soft_q_loss", qf_losses[:step+1].mean(), global_step)
        #writer.add_scalar("losses/policy_loss", policy_losses[:step+1].mean(), global_step)
    #writer.close()
    env.close()

# TODO:
#  Refine load and save of parameters
#  Tensorboard stuff
#  Train function
#  Test function
#  callbacks for each phase of training
#       evaluate 
#       end train on reward threshold
#       save every so many iterations
#       etc
#  eventually do hyperparam optimization (way later)
#  Misc: Look at rllib/stable-baselines for ideas



def test():
    import matplotlib.pyplot as plt

    #
    # Evaluate Model
    #
    eval_env = onedof()

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
    max_eps = 5
    obs = env.reset()
    timesteps = 100000
    hit_counter = 0
    #pbar = tqdm(total=timesteps)

    for i in range(timesteps):
        #pbar.update()
        with torch.no_grad():
            action, _ , _ = actor.get_action(obs.reshape(1,3))
        #action, _states = model.predict(obs)
        obs, rewards, dones, info = eval_env.step(action.tolist()[0][0])

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

            print(f"Ep{num_episodes} EpReward = {epRollingReward:.2f},\
                    EpLen = {eplen}, MissDis = {info['delta_position']}, Hits = {hit_counter}",)

            saved_ep_num.append(num_episodes)
            num_episodes += 1 

            saved_eplen.append(eplen)
            saved_epTotalrewards.append(epRollingReward)
            eplen = 0
            epRollingReward = 0
            eval_env.reset()

            if num_episodes == max_eps:
                #pbar.write("Reaching Max Eps")
                print("Reaching Max Eps")
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
    #fig.savefig("states.png")


    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot( saved_ep_num, saved_epTotalrewards)
    ax[0].set_title("epRollingrewards")
    ax[1].plot( saved_ep_num, saved_eplen )
    ax[1].set_title("saved_eplen")
    ax[2].plot( saved_ep_num, epMissDistance)
    ax[2].set_title(f"epMissDistance (Hit Ratio: {hit_ratio})")
    #fig.savefig("rewards.png")


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
    #fig.savefig("pos_by_ep.png")


    fig, ax = plt.subplots(1, sharex=True)
    for i in range(len(acc_in_ep)):
        ax.plot( steps, acc_in_ep[i] )
    ax.set_title("Acceleration")
    #fig.savefig("acc_by_ep.png")



    fig, ax = plt.subplots(1, sharex=True)
    for i in range(len(act_in_ep)):
        ax.plot( steps, act_in_ep[i] )
    ax.set_title("Action")
    #fig.savefig("act_by_ep.png")


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


