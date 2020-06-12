import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import registry, register

class onedof(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['console', 'rgb_array', 'human']}

    def __init__(self, max_cmd_accel=2, act_space=1, obs_space=1, seed=None, total_time=1, total_steps=100, tau=0.1, aimpoint=0 ):
        self.seed(seed)
        self.viewer = None

        self.max_cmd_accel =  max_cmd_accel
        self.min_cmd_accel = -max_cmd_accel

        self.total_time    =  total_time
        self.total_steps   =  total_steps
        self.dt            =  self.total_time/self.total_steps
        self.tau           =  tau
        self.aimpoint      =  aimpoint
        self.episode_step  =  0

        self.pos = np.random.uniform(-1, 1)
        self.vel = 0
        self.acc = 0
        self.cmd = 0
        self.state = np.array([self.pos, self.vel, self.acc] )

        self.action_space = spaces.Box(
            low=-act_space,
            high=act_space, 
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-obs_space,
            high=obs_space,
            shape=(3,),
            dtype=np.float32
        )

        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        self.episode_step += 1

        previous_step_error = np.abs(self.pos - self.aimpoint)
        previous_cmd = self.cmd 


        # Scale up action by max_cmd_accel
        #action     = np.clip( action, self.min_cmd_accel, self.max_cmd_accel)
        action     = action * self.max_cmd_accel
        self.cmd   = action # To store action

        # Calculate updated states
        self.pos   = self.pos + self.dt * self.vel
        self.vel   = self.vel + self.dt * self.acc
        self.acc   = (self.acc + self.dt/self.tau * (action - self.acc))[0]
        self.state = np.array( [self.pos, self.vel, self.acc] )

        # Capture current position error to aimpoint
        error = np.abs(self.pos - self.aimpoint)
        change_in_error = previous_step_error - error 

        #
        # Reward structure
        #   Reward Shaping
        #       Shape rewards such that over the duration of the episode
        #       the ideal actions are rewarded that lead to the end goal.
        #       Without shaping, only having a terminal reward can lead 
        #       to significantly longer training iterations.
        #   Reward Terminal
        #       This reward should give large bonus for achieving goal
        #   If pos is getting closer to aimpoint, give a +reward
        #   If pos is getting farther, give a -reward
        #   If cmd is dramatically different than previous cmd, -reward
        #   If at end of the episode, the pos is at aimpoint, give +++reward
        #   If at end of the episode, the pos is not at aimpoint, give ---reward
        # 
        reward  = 0
        #k = 0.5
        #if change_in_error < 0:
        #    # Getting further from aimpoint, -reward
        #    reward += -10 
        #else:
        #    # Getting closer to aimpoint, +reward
        #    reward +=  10

        # -Reward for dramatic change in cmd
        # Not sure this even really does that. 
        #change_in_cmd = np.abs( previous_cmd - self.cmd )[0]
        #reward += -10 * change_in_cmd

        # To reduce control effort?? 
        reward += -0.1 * self.cmd[0] * self.cmd[0]

        term_reward = 0
        if self.episode_step == self.total_steps:
            done = True
            term_reward = -10 * error
        else:
            done = False

        reward += term_reward
        
        info = {"error":error, "term_reward":term_reward}

        return self.state, reward, done, info

    def reset(self):
        self.episode_step = 0
        self.pos = np.random.uniform(-1, 1)
        self.vel = 0
        self.acc = 0
        self.state = np.array([self.pos, self.vel, self.acc] )

        if self.viewer is not None:
            plt.close( self.viewer[0] )
            self.viewer[0].clf()
            self.viewer = None

        return self.state

    def render(self, mode='human'):
        if self.viewer is None:
            fig, ax = plt.subplots(2, sharex=True)
            plt.show(block=False)
            self.viewer = (fig, ax)
            self.viewer[1][0].set_xlabel("Steps")
            self.viewer[1][0].set_ylabel("Pos")
            self.viewer[1][1].set_ylabel("Cmd")
            self.viewer[1][0].plot( self.total_steps, self.aimpoint, 'om')
        self.viewer[1][0].plot( self.episode_step, self.pos, 'xb' )
        self.viewer[1][1].plot( self.episode_step, self.cmd, 'xb' )
        #self.viewer[0].canvas.draw()
        plt.pause(0.001)
        #pass

    def close(self):
        if self.viewer is not None:
            plt.close( self.viewer[0] )
            self.viewer = None

register(
    id='Onedof-v0',
    entry_point=onedof,
    #max_episode_steps=100,
    #reward_threshold=100,
)



#
# Random Agent
#
def random_agent(n_steps=100, render=False):
    env = onedof()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())
    print(gym.envs.registry.env_specs["Onedof-v0"].max_episode_steps)
    print(gym.envs.registry.env_specs["Onedof-v0"].reward_threshold)

    obs = env.reset()
    ep_rwd = 0
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        ep_rwd += reward

        if render:
            env.render()

        print(f"Pos: {obs[0]:.2f}, Error={info['error']:.2f}, Step_Reward={reward:.2f}, Ep_Reward={ep_rwd:.2f}")
        if done:
            print(f"\tTerm_reward={info['term_reward']:.2f}")
            ep_rwd = 0
            obs = env.reset()

    env.close()


