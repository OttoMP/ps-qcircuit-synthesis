import numpy as np
import csv
from collections import namedtuple

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards",
                                   "episode_depths"])

class Simulation:
    def __init__(self, env, agent):

        self.env = env
        self.agent = agent

        self.episode_length = np.array([0])
        self.episode_reward = np.array([0])

    def save_csv(self, stats):
        with open('csvfiles/length.csv', 'w') as csvfile:
            rewardwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i, e in enumerate(stats.episode_lengths):
                rewardwriter.writerow([i,e])
            csvfile.close()

        with open('csvfiles/depth.csv', 'w') as csvfile:
            rewardwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i, e in enumerate(stats.episode_depths):
                rewardwriter.writerow([i,e])
            csvfile.close()

        with open('csvfiles/reward.csv', 'w') as csvfile:
            rewardwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i, e in enumerate(stats.episode_rewards):
                rewardwriter.writerow([i,e])
            csvfile.close()

    def run_ps(self, max_number_of_episodes=100, display_frequency=1):
        circuit_depth = np.array([0])

        # repeat for each episode
        for episode_number in range(max_number_of_episodes):

            # initialize state
            percept = self.env.reset()

            done = False # used to indicate terminal state
            R = 0 # used to display accumulated rewards for an episode
            t = 0 # used to display accumulated steps for an episode i.e episode length

            # repeat for each step of episode, until state is terminal
            while not done:

                t += 1 # increase step counter - for display

                # Return action from PS Agent memory
                action = self.agent.act(percept)

                # take action, observe reward and next percept
                next_percept, reward, done, depth = self.env.step(action)

                # agent learn (ECM update)
                self.agent.learn(reward, done)

                # state <- next state
                percept = next_percept

                R += reward # accumulate reward - for display

            self.episode_length = np.append(self.episode_length,t) # keep episode length - for display
            self.episode_reward = np.append(self.episode_reward,R) # keep episode reward - for display
            circuit_depth = np.append(circuit_depth, depth)        # keep episode depth - for display

        # Make Graphics and save them in file
        #self.fig.clf()
        stats = EpisodeStats(
            episode_lengths=self.episode_length,
            episode_rewards=self.episode_reward,
            episode_depths=circuit_depth)

        self.save_csv(stats)
        self.agent.print_ECM()

