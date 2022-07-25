#!/usr/bin/env python

import os

import numpy as np
import pybullet as p

from ravens.tasks import Task
from ravens import utils


class Aligning(Task):

    def __init__(self):
        super().__init__()
        self.ee = 'suction'
        self.max_steps = 5
        self.metric = 'pose'
        self.primitive = 'pick_place'

    def reset(self, env):
        self.num_steps = 1
        self.goal = {'places': {}, 'steps': []}

        # Generate randomly shaped box.
        box_size = self.random_size(0.05, 0.15, 0.05, 0.15, 0.01, 0.06)

        # Add corner.
        dimx = (box_size[0] / 2 - 0.025 + 0.0025, box_size[0] / 2 + 0.0025)
        dimy = (box_size[1] / 2 + 0.0025, box_size[1] / 2 - 0.025 + 0.0025)
        corner_template = 'assets/corner/corner-template.urdf'
        replace = {'DIMX': dimx, 'DIMY': dimy}
        corner_urdf = self.fill_template(corner_template, replace)
        corner_size = (box_size[0], box_size[1], 0)
        corner_pose = self.random_pose(env, corner_size)
        env.add_object(corner_urdf, corner_pose, fixed=True)
        os.remove(corner_urdf)

        # Add possible placing poses.
        theta = p.getEulerFromQuaternion(corner_pose[1])[2]
        flipped_rotation = p.getQuaternionFromEuler((0, 0, theta + np.pi))
        flipped_pose = (corner_pose[0], flipped_rotation)
        alt_x = (box_size[0] / 2) - (box_size[1] / 2)
        alt_y = (box_size[1] / 2) - (box_size[0] / 2)
        alt_position = (alt_x, alt_y, 0)
        alt_rotation0 = p.getQuaternionFromEuler((0, 0, np.pi / 2))
        alt_rotation1 = p.getQuaternionFromEuler((0, 0, 3 * np.pi / 2))
        alt_pose0 = self.multiply(corner_pose, (alt_position, alt_rotation0))
        alt_pose1 = self.multiply(corner_pose, (alt_position, alt_rotation1))
        self.goal['places'] = {0: corner_pose,
                               1: flipped_pose,
                               2: alt_pose0,
                               3: alt_pose1}

        # Add box.
        box_template = 'assets/box/box-template.urdf'
        box_urdf = self.fill_template(box_template, {'DIM': box_size})
        box_pose = self.random_pose(env, box_size)
        box_id = env.add_object(box_urdf, box_pose)
        os.remove(box_urdf)
        self.color_random_brown(box_id)
        self.goal['steps'].append({box_id: (2 * np.pi, [0, 1, 2, 3])})
