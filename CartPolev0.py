import gym
from gym import wrappers
from math import pi
import numpy as np


class Learner:
    def __init__(self):
        self.moves = [[[0, 0] for spped in np.arange(-5, 5.1, 0.4)] 
            for angle in range(-20, 20, 4)]
        self.angles = [[angle, [speed for speed in np.arange(-5, 5.1, 0.4) ]] 
            for angle in range(-20, 20, 4)]
        self.indexes_moves = []

    def find_index(self, observation):
        angle = (observation[2] * 180 / pi)
        for i in range(len(self.angles)-1):
            if self.angles[i][0] < angle < self.angles[i+1][0]:
                for j in range(len(self.angles[0][1])):
                    if self.angles[i][1][j] < observation[1] < self.angles[i][1][j+1]:
                        return i, j

    def q_action(self, index):
        if self.moves[index[0]][index[1]][0] >= self.moves[index[0]][index[1]][1]:
            return 0
        else:
            return 1

    def add_reward(self, index, action, done, next_index, t):
        if done and t != 199:
            for ind in self.indexes_moves:
                self.moves[ind[1][0]][ind[1][1]][ind[0]] += -0.003
                self.moves[index[0]][index[1]][action] += -1    
        elif done and t == 199:
            for ind in self.indexes_moves:
                self.moves[ind[1][0]][ind[1][1]][ind[0]] += 1000
        else:
            self.moves[index[0]][index[1]][action] += 0.002
            self.moves[index[0]][index[1]][action] =  \
                max(self.moves[next_index[0]][next_index[1]]) * 0.8 + \
                0.2 * self.moves[index[0]][index[1]][action]


class Environment:
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.env = wrappers.Monitor(self.env, './tmp4/cartpole-experiment-1', force=True)
        self.learner = Learner()

        self.scores = []

    def update(self, t):
        action = self.learner.q_action(self.index)
        self.observation, reward, self.done, info = self.env.step(action)
        self.learner.indexes_moves.append([action, self.index])

        next_index = self.learner.find_index(self.observation)
        self.learner.add_reward(self.index, action, self.done, next_index, t)

        self.index = next_index

    def run(self):
        for _ in range(1000):
            self.observation = self.env.reset()
            self.index = self.learner.find_index(self.observation)
            for t in range(300):
                self.update(t)
                if self.done:
                    self.scores.append(t)
                    break

        print(max(self.scores))
        print(sum(self.scores)/len(self.scores))


if __name__ == "__main__":
    environment = Environment()
    environment.run()
    environment.env.close()
