import gym
import numpy
import pathlib
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plot
import seaborn as sns
import random
import copy
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_WEST = 2
MOVE_EAST = 3

ROOMS_ACTIONS = [MOVE_NORTH,MOVE_SOUTH,MOVE_WEST,MOVE_EAST]

AGENT_CHANNEL = 0
GOAL_CHANNEL = 1
OBSTACLE_CHANNEL = 2
NR_CHANNELS = len([AGENT_CHANNEL,GOAL_CHANNEL,OBSTACLE_CHANNEL])

class RoomsEnv(gym.Env):

    def __init__(self, width, height, obstacles, time_limit, stochastic=False, movie_filename=None):
        self.seed()
        self.movie_filename = movie_filename
        self.action_space = spaces.Discrete(len(ROOMS_ACTIONS))
        self.observation_space = spaces.Box(-numpy.inf, numpy.inf, shape=(NR_CHANNELS,width,height))
        self.agent_position = None
        self.done = False
        self.goal_position = (width-2,height-2)
        self.obstacles = obstacles
        self.time_limit = time_limit
        self.time = 0
        self.width = width
        self.height = height
        self.stochastic = stochastic
        self.undiscounted_return = 0
        self.state_history = []
        self.reset()
        
    def is_subgoal(self, state):
        is_at_goal = self.agent_position == self.goal_position
        x,y = self.agent_position
        is_at_door_vertical = state[y-1][x][OBSTACLE_CHANNEL] == 1 and state[y+1][x][OBSTACLE_CHANNEL] == 1
        is_at_door_horizontal = state[y][x-1][OBSTACLE_CHANNEL] == 1 and state[y][x+1][OBSTACLE_CHANNEL] == 1
        return is_at_goal or is_at_door_vertical or is_at_door_horizontal
        
    def state(self):
        state = numpy.zeros((NR_CHANNELS,self.width,self.height))
        x_agent,y_agent = self.agent_position
        state[AGENT_CHANNEL][x_agent][y_agent] = 1
        x_goal, y_goal = self.goal_position
        state[GOAL_CHANNEL][x_goal][y_goal] = 1
        for obstacle in self.obstacles:
            x,y = obstacle
            state[OBSTACLE_CHANNEL][x][y] = 1
        return numpy.swapaxes(state, 0, 2)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self, action):
        if self.stochastic and numpy.random.rand() < 0.2:
            action = random.choice(ROOMS_ACTIONS)
        return self.step_with_action(action)
        
    def step_with_action(self, action):
        if self.done:
            return self.agent_position, 0, self.done, {}
        self.time += 1
        self.state_history.append(self.state())
        x,y = self.agent_position
        reward = 0
        if action == MOVE_NORTH and y+1 < self.height:
            self.set_position_if_no_obstacle((x, y+1))
        elif action == MOVE_SOUTH and y-1 >= 0:
            self.set_position_if_no_obstacle((x, y-1))
        if action == MOVE_WEST and x-1 >= 0:
            self.set_position_if_no_obstacle((x-1, y))
        elif action == MOVE_EAST and x+1 < self.width:
            self.set_position_if_no_obstacle((x+1, y))
        goal_reached = self.agent_position == self.goal_position
        if goal_reached:
            reward = 1
        self.undiscounted_return += reward
        self.done = goal_reached or self.time >= self.time_limit
        return self.agent_position, reward, self.done, {}
        
    def set_position_if_no_obstacle(self, new_position):
        if new_position not in self.obstacles:
            self.agent_position = new_position

    def reset(self):
        self.done = False
        self.agent_position = (1,1)
        self.time = 0
        self.state_history.clear()
        return self.agent_position
        
    def state_summary(self, state):
        return {
            "agent_x": self.agent_position[0],
            "agent_y": self.agent_position[0],
            "goal_x": self.goal_position[0],
            "goal_y": self.goal_position[0],
            "is_subgoal": self.is_subgoal(state),
            "time_step": self.time,
            "score": self.undiscounted_return
        }
        
    def save_video(self):
        if self.movie_filename is not None:
            history_of_states = self.state_history
            duration_in_seconds = len(history_of_states)/4
            fig, ax = plot.subplots()
            frames_per_second = len(history_of_states)/duration_in_seconds
            def make_frame(t):
                ax.clear()
                ax.grid(False)
                ax.imshow(history_of_states[int(t*frames_per_second)])
                ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)
                return mplfig_to_npimage(fig)
            animation = VideoClip(make_frame, duration=duration_in_seconds)
            animation.write_videofile(self.movie_filename, fps=frames_per_second)
        
def read_map_file(path):
    file = pathlib.Path(path)
    assert file.is_file()
    with open(path) as f:
        content = f.readlines()
    obstacles = []
    width = 0
    height = 0
    for y,line in enumerate(content):
        for x,cell in enumerate(line.strip().split()):
            if cell == '#':
                obstacles.append((x,y))
            width = x
        height = y
    width += 1
    height += 1
    return width, height, obstacles

def load_env(path, movie_filename, time_limit=100, stochastic=False):
    width, height, obstacles = read_map_file(path)
    return RoomsEnv(width, height, obstacles, time_limit, stochastic, movie_filename)
