#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from past.utils import old_div
import os
import rospy
import rospkg
import pickle

import numpy as np
import rosbag
import roslaunch
from rosbag import ROSBagUnindexedException

import test_driver
import std_msgs.msg
import sys
import time
from nav_msgs.msg import Odometry
from kobuki_msgs.msg import BumperEvent
from gazebo_msgs.srv import GetModelState
import csv
import math


def find_results(dir):
    file = [dir + f for f in os.listdir(dir) if os.path.isfile(dir + f)]
    results = []
    # results = {}
    # for p in params.keys():
    # range_list = list(params[p]['range'])
    # l = len(range_list)
    # score = np.zeros(l)
    # success = 0.  # np.zeros(l)
    # pl = 0.  # np.zeros(l)
    # t = 0.  # np.zeros(l)
    # num_s = 0
    # num_pl = 0
    # num_t = 0
    # results = {}
    # for i in range(l):
    #     value = range_list[i]
    # file_pv = [f for f in file if p + str(value) in f]
    for f in file:
        try:
            bag = rosbag.Bag(f, 'r')
            for _, result, _ in bag.read_messages(topics=['result']):
                # num_s += 1
                if result.data == 'SUCCEEDED':
                    results.append(1.)
                else:
                    results.append(0.)
                    # success += 1.
                    # for _, path_length, _ in bag.read_messages(topics=['path_length']):
                    #     pl += path_length.data
                    #     num_pl += 1
                    # for _, rt, _ in bag.read_messages(topics=['time']):
                    #     t += rt.data.secs
                    #     num_t += 1

        except ROSBagUnindexedException:
            continue
    bag.close()
    # success /= num_s
    # pl /= num_pl
    # t /= num_t
    # results = {'success rate': success, 'path_length': pl, 'runtime': t}

    return np.array(results)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return old_div(ret[n - 1:], n)


def combine_models(params, models, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    '''
    for m in models:
        combined = {}
        for p in params:
            f = '/home/haoxin/data/training/' + p + '/' + m + '_models.pickle'
            combined[p] = pickle.load(open(f, 'rb'))[p]
        f = save_path + m + '_models.pickle'
        pickle.dump(combined, open(f, 'wb'))
    '''
    '''
    bc = {1.5:{}, 1.25: {}, 1.0:{}, 0.75:{}}
    for p in params:
        f = '/home/haoxin/data/training/' + p + '/best_configs.pickle'
        bcp = pickle.load(open(f, 'rb'))
        for key in bc.keys():
            bc[key][p] = bcp[key][p]
    f = save_path + '/best_configs.pickle'
    pickle.dump(bc, open(f, 'wb'))
    '''
    import torch
    for m in models:
        combined = {}
        for p in params:
            f = '/home/haoxin/data/training/' + p + '/' + m + '_models.pt'
            combined[p] = torch.load(f)
        f = save_path + m + '.pickle'
        pickle.dump(combined, open(f, 'wb'))


if __name__ == "__main__":
    '''
    models = ['deep_cnn_regressor']
    params = ['max_depth', 'planner_frequency']
    save_path = '/home/haoxin/data/training/depth_planner_freq/'
    combine_models(params, models, save_path)
    '''
    # '''
    dir = '/home/haoxin/data/rl/multiple/model/dqn/turtlebot_ego_teb/'
    a = find_results(dir)
    ma = moving_average(a, 100)
    print(ma)
    ma.tofile('pretrained.csv',sep="\n",format='%0.2f')
    # '''
