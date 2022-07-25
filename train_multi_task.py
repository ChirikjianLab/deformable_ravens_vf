#!/usr/bin/env python

""" The main script for training the GCTN in the multi-modal action proposal module. """

import datetime
import os
import argparse
import numpy as np
import tensorflow as tf
from ravens import agents
from ravens.dataset_multi import DatasetMulti

task_list = [
  'stack-tower',
  'stack-pyramid',  
  'stack-square',
  'put-block-base',
  'stack-palace',
  'stack-t']

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',            default='0')
    parser.add_argument('--disp',           action='store_true')
    parser.add_argument('--data_dir',       default=None) # Directory containing the training data (ravens_visual_foresight/data_train)
    parser.add_argument('--models_dir',     default=None) # Directory containing the trained models (ravens_visual_foresight/gctn_models)
    parser.add_argument('--task',           default='put-block-base-mcts') # This is just a placeholder.
    parser.add_argument('--agent',          default='transporter-goal')
    parser.add_argument('--num_demos',      default='10')
    parser.add_argument('--num_runs',       default=1, type=int)
    parser.add_argument('--num_rots',       default=36, type=int)
    parser.add_argument('--gpu_mem_limit',  default=None)
    parser.add_argument('--subsamp_g',      action='store_true')
    
    args = parser.parse_args()

    # Configure which GPU to use.
    cfg = tf.config.experimental
    gpus = cfg.list_physical_devices('GPU')
    if len(gpus) == 0:
        print('No GPUs detected. Running with CPU.')
    else:
        cfg.set_visible_devices(gpus[int(args.gpu)], 'GPU')

    # Configure how much GPU to use.
    if args.gpu_mem_limit is not None:
        MEM_LIMIT = 1024 * int(args.gpu_mem_limit)
        print(args.gpu_mem_limit)
        dev_cfg = [cfg.VirtualDeviceConfiguration(memory_limit=MEM_LIMIT)]
        cfg.set_virtual_device_configuration(gpus[0], dev_cfg)

    if args.data_dir is None:
        raise ValueError("data_dir should be specified! python train_multi_task.py --data_dir=/dir/to/training/data")

    # Initialize task. Later, initialize Environment if necessary.
    dataset_dir_list = [os.path.join(args.data_dir, f'{task}-mcts-pp-train') for task in task_list]
    dataset = DatasetMulti(dataset_dir_list)
    if args.subsamp_g:
        dataset.subsample_goals = True  

    # Evaluate on increasing orders of magnitude of demonstrations.
    num_train_runs = args.num_runs  # to measure variance over random initialization
    num_train_iters = 40000
    test_interval = 4000

    # Check if it's goal-conditioned.
    goal_conditioned = True

    # Do multiple training runs from scratch with TensorFlow random initialization.
    for train_run in range(num_train_runs):

        # Set up tensorboard logger.
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join('logs', args.agent, current_time, 'train')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # Set the beginning of the agent name.
        name = f'GCTN-Multi-{args.agent}-{args.num_demos}-{train_run}'

        # Initialize agent and limit random dataset sampling to fixed set.
        tf.random.set_seed(train_run)
        
        assert 'transporter-goal' in args.agent
        assert goal_conditioned
        name = f'{name}-rots-{args.num_rots}'
        if args.subsamp_g:
            name += '-sub_g'
        else:
            name += '-fin_g'
        agent = agents.names[args.agent](
            name, args.task, num_rotations=args.num_rots, models_dir=args.models_dir)

        # Limit random data sampling to fixed set.
        np.random.seed(train_run)
        num_demos = int(args.num_demos)

        episodes_list = []
        for i in range(len(task_list)):
            max_demos = dataset.n_episodes_list[i]
            assert max_demos >= num_demos
            episodes = np.random.choice(range(max_demos), num_demos, False)
            episodes_list.append(episodes)
            dataset.set(i, episodes_list[i])

        performance = []
        while agent.total_iter < num_train_iters:
            # Train agent.
            tf.keras.backend.set_learning_phase(1)
            agent.train(dataset, num_iter=test_interval, writer=train_summary_writer)
            tf.keras.backend.set_learning_phase(0)