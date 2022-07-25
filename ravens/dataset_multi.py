#!/usr/bin/env python

"""
Dataset for training with multiple tasks.
This code works with simulation data.
"""

import os
import sys
import json
import argparse
import cv2
import pickle
import numpy as np
from ravens import utils as U
from ravens import tasks, cameras
from collections import defaultdict
import tensorflow as tf

# See transporter.py, regression.py, dummy.py, task.py, load.py, etc.
PIXEL_SIZE = 0.003125
CAMERA_CONFIG = cameras.RealSenseD415.CONFIG
BOUNDS = np.array([[0.25, 0.75], [-0.25, 0.25], [0, 0.28]])

# Task names as strings, REVERSE-sorted so longer (more specific) names come first.
TASK_NAMES = (tasks.names).keys()
TASK_NAMES = sorted(TASK_NAMES)[::-1]


def get_max_episode_len(path):
    """A somewhat more scalable way to get the max episode lengths."""
    path = path.replace('data/', '')
    path = path.replace('goals/', '')
    task = tasks.names[path]()
    max_steps = task.max_steps - 1  # Remember, subtract one!
    return max_steps


def process_depth(img, cutoff=10):
    # Turn to three channels and zero-out values beyond cutoff.
    w,h = img.shape
    d_img = np.zeros([w,h,3])
    img = img.flatten()
    img[img > cutoff] = 0.0
    img = img.reshape([w,h])
    for i in range(3):
        d_img[:,:,i] = img

    # Scale values into [0,255) and make type uint8.
    assert np.max(d_img) > 0.0
    d_img = 255.0 / np.max(d_img) * d_img
    d_img = np.array(d_img, dtype=np.uint8)
    for i in range(3):
        d_img[:,:,i] = cv2.equalizeHist(d_img[:,:,i])
    return d_img


def get_heightmap(obs):
    """Following same implementation as in transporter.py."""
    heightmaps, colormaps = U.reconstruct_heightmaps(
        obs['color'], obs['depth'], CAMERA_CONFIG, BOUNDS, PIXEL_SIZE)
    colormaps = np.float32(colormaps)
    heightmaps = np.float32(heightmaps)

    # Fuse maps from different views.
    valid = np.sum(colormaps, axis=3) > 0
    repeat = np.sum(valid, axis=0)
    repeat[repeat == 0] = 1
    colormap = np.sum(colormaps, axis=0) / repeat[..., None]
    colormap = np.uint8(np.round(colormap))
    heightmap = np.max(heightmaps, axis=0)
    return colormap, heightmap


class DatasetMulti:

    def __init__(self, path_list):
        """A simple RGB-D image dataset."""
        self.path_list = path_list
        self.dataset_num = len(self.path_list)
        self.sample_set_list = []
        for _ in range(len(self.path_list)):
            self.sample_set_list.append([])

        self.n_episodes_list = [0] * len(self.path_list)

        # Track existing dataset if it exists.
        for i, path in enumerate(self.path_list):
            color_path = os.path.join(path, 'color')
            max_seed = -1
            if os.path.exists(color_path):
                for fname in sorted(os.listdir(color_path)):
                    if '.pkl' in fname:
                        seed = int(fname[(fname.find('-') + 1):-4])
                        self.n_episodes_list[i] += 1
                        max_seed = max(max_seed, seed)
            print(f'[Dataset Loaded] Path: {path} N_Episodes: {self.n_episodes_list[i]}')

        self._cache = dict()

        # Only for goal-conditioned Transporters, if we want more goal images.
        self.subsample_goals = False

    def set(self, dataset_idx, episodes):
        """Limit random samples to specific fixed set."""
        
        self.sample_set_list[dataset_idx] = episodes
        print(f'Dataset: {self.path_list[dataset_idx]}')
        print(f'Dataset Episode: {self.sample_set_list[dataset_idx]}')

    def load(self, dataset_id, episode_id, images=True, cache=False):
        """Load data from a saved episode.

        Args:
        dataset_id: the ID of the dataset to be loaded.
        episode_id: the ID of the episode to be loaded.
        images: load image data if True.
        cache: load data from memory if True.

        Returns:
        episode: list of (obs, act, reward, info) tuples.
        seed: random seed used to initialize the episode.
        """
        
        def load_field(dataset_id, episode_id, field, fname):

            # Check if sample is in cache.
            if cache:
                if episode_id in self._cache:
                    if field in self._cache[episode_id]:
                        return self._cache[episode_id][field]
                else:
                    self._cache[episode_id] = {}

            # Load sample from files.
            path = os.path.join(self.path_list[dataset_id], field)
            with open(os.path.join(path, fname), 'rb') as f:
                try:
                    data = pickle.load(f)
                except EOFError:
                    data = None
            if cache:
                self._cache[episode_id][field] = data
            return data

        # Get filename and random seed used to initialize episode.
        seed = None
        path = os.path.join(self.path_list[dataset_id], 'action')

        for fname in sorted(tf.io.gfile.listdir(path)):
            if f'{episode_id:06d}' in fname:
                seed = int(fname[(fname.find('-') + 1):-4])

                # Load data.
                color = load_field(dataset_id, episode_id, 'color', fname)
                depth = load_field(dataset_id, episode_id, 'depth', fname)
                action = load_field(dataset_id, episode_id, 'action', fname)
                reward = load_field(dataset_id, episode_id, 'reward', fname)
                info = load_field(dataset_id, episode_id, 'info', fname)

                # Reconstruct episode.
                episode = []
                for i in range(len(action)):
                    obs = {'color': color[i], 'depth': depth[i]} if images else {}
                    episode.append((obs, action[i], reward[i], info[i]))
                return episode, seed

    def random_sample(self, goal_images=False):
        """Randomly sample from the dataset uniformly.

        Daniel: The 'cached_load' will use the load (from pickle file) to
        load the list, and then extract the time step `i` within it as the
        data point. I'm also adding a `goal_images` feature to load the last
        information. The last information isn't in a list, so we don't need
        to extract an index. That is, if loading a 1-length time step, we
        should see this:

        In [11]: data = pickle.load( open('last_color/000099-1.pkl', 'rb') )
        In [12]: data.shape
        Out[12]: (3, 480, 640, 3)

        In [13]: data = pickle.load( open('color/000099-1.pkl', 'rb') )
        In [14]: data.shape
        Out[14]: (1, 3, 480, 640, 3)

        Update: now using goal_images for gt_state, but here we should interpret
        `goal_images` as just giving the 'info' portion to the agent.
        """
        # Randomly select a dataset.
        dataset_id = np.random.choice(range(len(self.n_episodes_list)))
        
        # Randomly select an episode.
        if len(self.sample_set_list[dataset_id]) > 0:
            iepisode = np.random.choice(self.sample_set_list[dataset_id])
        else:
            iepisode = np.random.choice(range(self.n_episodes_list[dataset_id]))

        print(f'{self.path_list[dataset_id]} -- {iepisode}')
        
        # Load the episode.
        episode, _ = self.load(dataset_id, iepisode)
        
        # Pick a step in the episode that is not random action.
        while True:
            i = np.random.choice(range(len(episode)-1))
            random = episode[i][3]['random']
            if not random:
                break
        
        assert not episode[i][3]['random']
        assert i < (len(episode) - 1)
        obs = {}
        obs['color'] = episode[i][0]['color']
        obs['depth'] = episode[i][0]['depth']
        act = {}
        act['params'] = episode[i][1]
        act['camera_config'] = CAMERA_CONFIG
        act['primitive'] = 'pick_place'
        info = episode[i][3]
        assert obs['color'].shape == (3, 480, 640, 3), obs['color'].shape
        assert obs['depth'].shape == (3, 480, 640), obs['depth'].shape

        # Load goal images. Must be in the SAME episode! We might not pick final images.
        assert goal_images
        if goal_images:
            ep_len = len(episode)
            assert i < ep_len, f'{i} vs {ep_len}'
            goal = {}

            # Subsample the goal, from i+1 up to (and INCLUDING) the episode length.
            if self.subsample_goals:
                low = i + 1
                high = ep_len
                new_i = np.random.choice(range(low, high+1))  # NOTE: high+1
            else:
                new_i = ep_len

            assert new_i == ep_len
            if new_i < ep_len:
                # Load a list and index in it by `new_i`.
                goal['color'] = episode[new_i][0]['color']
                goal['depth'] = episode[new_i][0]['depth']
                goal['info']  = episode[new_i][3]
            else:
                # Stand-alone information about the final set of images.
                goal['color'] = episode[-1][0]['color']
                goal['depth'] = episode[-1][0]['depth']
                goal['info']  = episode[-1][3]

            assert goal['color'].shape == (3, 480, 640, 3), goal['color'].shape
            assert goal['depth'].shape == (3, 480, 640), goal['depth'].shape
            return obs, act, info, goal

        return obs, act, info