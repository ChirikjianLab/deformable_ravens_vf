#!/usr/bin/env python

import os
import numpy as np

import pybullet as p

from ravens.tasks import Task
from ravens import utils


class Pushing(Task):

    def __init__(self):
        super().__init__()
        self.ee = 'stick'
        self.max_steps = 200
        self.metric = 'zone'
        self.primitive = 'push'

    def reset(self, env):
        self.total_rewards = 0

        # Add goal post.
        line_urdf = 'assets/line/line.urdf'
        self.zone_size = (0.5, 1, 0.2)
        self.zone_pose = ((0.5, -0.86, 0), p.getQuaternionFromEuler((0, 0, 0)))
        env.add_object(line_urdf, self.zone_pose, fixed=True)

        # Add box.
        box_size = self.random_size(0.05, 0.25, 0.05, 0.15, 0.05, 0.05)
        box_template = 'assets/box/box-template.urdf'
        box_urdf = self.fill_template(box_template, {'DIM': box_size})
        px = self.bounds[0, 0] + 0.1 + np.random.rand() * 0.3
        position = (px, 0.3, box_size[2] / 2)
        theta = np.random.rand() * np.pi / 4 - np.pi / 8
        rotation = p.getQuaternionFromEuler((0, 0, theta))
        box_pose = (position, rotation)
        box_id = env.add_object(box_urdf, box_pose)
        os.remove(box_urdf)
        self.color_random_brown(box_id)
        self.object_points = {box_id: self.get_object_points(box_id)}

        # Move end effector to start position next to box.
        box_dim = p.getVisualShapeData(box_id)[0][3]
        start_position = np.array([0, box_dim[1] / 2 + 0.01, 0])
        start_position = self.apply(box_pose, start_position)
        rotation = tuple(env.home_pose[3:])
        joints = env.solve_IK((tuple(start_position) + rotation))
        for i in range(len(env.joints)):
            p.resetJointState(env.ur5, env.joints[i], joints[i])
