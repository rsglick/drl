import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import registry, register

from integrators import rk4
from integrators import euler

class onedof(gym.Env):
    """
    Custom Environment that follows gym interface.
    """

    metadata = {'render.modes': ['console', 'rgb_array', 'human']}
    reward_range = (-float('inf'), float('inf'))

    def __init__(self, ctrl_limit=10, seed=None, n_axis=1,
                 total_time=1, dt=0.01, aimpoint=0,
                 integration="euler", custom_reward_fn=None):
        self.seed(seed)
        self.viewer = None

        self.total_time        = total_time
        self.dt                = dt
        self.max_episode_steps = self.total_time/self.dt
        self.integration       = integration
        self.custom_reward_fn  = custom_reward_fn

        self.aimpoint          = aimpoint
        self.episode_step      = 0

        #
        # Initialize state
        #   states: Position, Velocity, Acceleration
        #
        self.n_axis = n_axis
        pos = np.random.uniform(-1, 1, self.n_axis)
        vel = np.zeros_like(pos)
        acc = np.zeros_like(pos)
        self.init_states = np.array( [pos, vel, acc] )
        self.state = self.init_states.copy()

        # Initialilze contrl params 
        self.ctrl          = np.zeros( (self.n_axis,) )
        self.ctrl_limit    = ctrl_limit

        act_space = 1
        obs_space = 1
        self.action_space = spaces.Box(
            low=-act_space,
            high=act_space, 
            shape=self.ctrl.shape,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-obs_space,
            high=obs_space,
            shape=self.state.shape,
            dtype=np.float32
        )

        self.reset()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]


        
        
    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """        

        def basic_reward_function():
            """
            Reward Shaping
                Shape rewards such that over the duration of the episode
                the ideal actions are rewarded that lead to the end goal.
                Without shaping, only having a terminal reward can lead 
                to significantly longer training iterations.
            Reward Terminal
                This reward should give large bonus for achieving goal
            If at end of the episode, the pos is at aimpoint, give +++reward
            If at end of the episode, the pos is not at aimpoint, give ---reward
            """
            reward = 0.0

            # Reward shaping to reduce overall ctrl usage
            reward_gain = 0.001
            reward     += - reward_gain * self.ctrl[0] * self.ctrl[0]

            # Reward shaping to reduce dramatic changes in ctrl 
            #reward_gain = 0.10
            #reward += - reward_gain * np.abs( previous_ctrl - self.ctrl[0] )

            # Terminal Rewards
            term_reward =  0.0
            reward_gain = 1000.0
            if self.episode_step == self.max_episode_steps:
                term_reward = (- reward_gain * error * error).item()

            reward += term_reward
            
            info = {"error":error, "term_reward":term_reward}
            
            return reward, info


        #from IPython import embed; embed()
        self.episode_step += 1

        # Scale up action
        previous_ctrl       = self.ctrl[0]
        self.ctrl[0]        = action * self.ctrl_limit
        

        last_state = self.state.copy()
        # RK4 Integration to calculate updated states
        if self.integration is "rk4":
            self.state = rk4(last_state, self.ctrl, self.dt)

        # Euler Integration to calculate updated states
        elif self.integration is "euler":
            self.state = euler(last_state, self.ctrl, self.dt)

        # Capture current position error to aimpoint
        error = np.abs(self.state[0] - self.aimpoint)

        # Reward Function
        if self.custom_reward_fn is None:
            reward, info = basic_reward_function()
        else:
            reward, info = self.custom_reward_fn()

        # Check if episode is over
        if self.episode_step == self.max_episode_steps:
            done = True
        else:
            done = False

        return self.state, reward, done, info

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """

        self.episode_step = 0

        #
        # Initialize state
        #   states: Position, Velocity, Acceleration
        #
        pos = np.random.uniform(-1, 1, self.n_axis)
        vel = np.zeros_like(pos)
        acc = np.zeros_like(pos)
        self.init_states = np.array( [pos, vel, acc] )
        self.state = self.init_states.copy()

        # Initialilze contrl params 
        self.ctrl = np.zeros( (self.n_axis,) )


        if self.viewer is not None:
            plt.close( self.viewer[0] )
            self.viewer[0].clf()
            self.viewer = None

        return self.state

    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """

        if self.viewer is None:
            fig, ax = plt.subplots(2, sharex=True)
            plt.show(block=False)
            self.viewer = (fig, ax)
            self.viewer[1][0].set_xlabel("Steps")
            self.viewer[1][0].set_ylabel("Pos")
            self.viewer[1][1].set_ylabel("Ctrl")
            self.viewer[1][0].plot( self.max_episode_steps, self.aimpoint, 'om')
        self.viewer[1][0].plot( self.episode_step, self.state[0], 'xb' )
        self.viewer[1][1].plot( self.episode_step, self.ctrl[0], 'xb' )
        #self.viewer[0].canvas.draw()
        plt.pause(0.001)
        #pass

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """        
        if self.viewer is not None:
            plt.close( self.viewer[0] )
            self.viewer = None

register(
    id='Onedof-v0',
    entry_point=onedof,
    reward_threshold=0,
)



#
# Random Agent
#
def random_agent(n_steps=100, render=False, ctrl=None):
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
        if ctrl:
            action = ctrl
        else:
            action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        ep_rwd += reward

        if render:
            env.render()

        #from IPython import embed; embed()
        print(f"Pos: {obs[0].item():.2f}, Error={info['error'].item():.2f}, Step_Reward={reward:.2f}, Ep_Reward={ep_rwd:.2f}")
        if done:
            print(f"\tTerm_reward={info['term_reward']:.2f}")
            ep_rwd = 0
            #input()
            obs = env.reset()
    
    env.close()

if __name__ == "__main__":
    random_agent(render=True, ctrl=1)

