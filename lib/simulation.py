import numpy as np
import sys
import pandas as pd
import csv
from collections import namedtuple
from matplotlib import pyplot as plt
from matplotlib import pylab
import matplotlib.gridspec as gridspec

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards", "episode_depths", "episode_running_variance"])

class Simulation:
    def __init__(self, env, agent):

        self.env = env
        self.agent = agent

        self.episode_length = np.array([0])
        self.episode_reward = np.array([0])

        self.fig = pylab.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(2, 2)
        self.ax = pylab.subplot(gs[:, 0])
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)

        self.ax1 = pylab.subplot(gs[0, 1])
        self.ax1.yaxis.set_label_position("right")
        self.ax1.set_ylabel('Length')
        self.ax1.set_xlim(0, max(10, len(self.episode_length)+1))
        self.ax1.set_ylim(0, 51)

        self.ax2 = pylab.subplot(gs[1, 1])
        self.ax2.set_xlabel('Episode')
        self.ax2.yaxis.set_label_position("right")
        self.ax2.set_ylabel('Reward')
        self.ax2.set_xlim(0, max(10, len(self.episode_reward)+1))
        self.ax2.set_ylim(0, 2)

        self.line, = self.ax1.plot(range(len(self.episode_length)),self.episode_length)
        self.line2, = self.ax2.plot(range(len(self.episode_reward)),self.episode_reward)

    def save_csv(self, stats):
        with open('length.csv', 'w') as csvfile:
            rewardwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i, e in enumerate(stats.episode_lengths):
                rewardwriter.writerow([i,e])
            csvfile.close()

        with open('depth.csv', 'w') as csvfile:
            rewardwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i, e in enumerate(stats.episode_depths):
                rewardwriter.writerow([i,e])
            csvfile.close()

        with open('reward.csv', 'w') as csvfile:
            rewardwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i, e in enumerate(stats.episode_rewards):
                rewardwriter.writerow([i,e])
            csvfile.close()

    def plot_episode_stats(self, stats, smoothing_window=10, hideplot=False):
        # Plot the episode length over time
        fig1 = plt.figure(figsize=(10,5))
        plt.plot(stats.episode_lengths)
        plt.xlabel("Episode")
        plt.ylabel("Total Gates in circuit")
        plt.title("Total Gates over Time")
        if hideplot:
            plt.close(fig1)
        else:
            plt.show(fig1)

        #Plot the episode circuit depth over time
        fig2 = plt.figure(figsize=(10,5))
        plt.plot(stats.episode_depths)
        plt.xlabel("Episode")
        plt.ylabel("Circuit depth")
        plt.title("Circuit Depth over Time")
        if hideplot:
            plt.close(fig2)
        else:
            plt.show(fig2)

        # Plot the episode reward over time
        fig3 = plt.figure(figsize=(10,5))
        rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")
        plt.title("Episode Reward over Time")
        if hideplot:
            plt.close(fig3)
        else:
            plt.show(fig3)

        return fig1, fig2, fig3

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
        self.fig.clf()
        stats = EpisodeStats(
            episode_lengths=self.episode_length,
            episode_rewards=self.episode_reward,
            episode_depths=circuit_depth,
            episode_running_variance=np.zeros(max_number_of_episodes))
        self.save_csv(stats)
        '''
        lenght_plot, depth_plot, reward_plot = self.plot_episode_stats(stats, display_frequency)
        lenght_plot.savefig("total_gates.png")
        depth_plot.savefig("depth.png")
        reward_plot.savefig("reward.png")
        '''
