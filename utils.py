import os
import sys

import shutil
import logging

import torch.multiprocessing as mp
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class plot_scalers:
    def __init__(self, path):
        self.path = path

        self.xs = []
        self.ys = defaultdict(list)

    def __call__(self, x, ys):
        self.xs.append(x)

        plt.figure()
        for k,v in ys.items():
            self.ys[k].append(v)

            plt.plot(self.xs, self.ys[k], '-', label=k)

        plt.legend()

        plt.savefig(self.path)

        plt.close()






class printer:
    def __init__(self, args):
        self.args = args

    def __call__(self, msg):
        msg = time.strftime("%Y-%m-%d %H:%M:%S, ", time.localtime()) + msg
        with open(os.path.join(self.args.save_dir, 'log.txt'), 'a') as f:
            f.write(msg+ '\n')
        print(msg)


def create_exp_dir(args):
    if not os.path.exists('./runs'):
        os.mkdir('./runs')

    i = 0
    while os.path.exists(os.path.join('./runs', args.exp + str(i))):
        i += 1
    args.save_dir = os.path.join('./runs', args.exp + str(i))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    files = os.listdir('./')
    for file in files:
        if 'runs' not in file:
            try:
                shutil.copytree(file, os.path.join(args.save_dir, file))
            except:
                try:
                    os.mkdir(args.save_dir)
                except:
                    pass
                shutil.copyfile(file, os.path.join(args.save_dir, file))

    # print_ = printer(args)

    print('experiments dir is {}'.format(args.save_dir))

    # return print_




def loss_(x, mus, x_range):
    M = mus.shape[0] // 2
    interval = (x_range[1] - x_range[0]) / M
    y = 0
    for m in range(M):

        if m == M - 1:
            ind = x >= (x_range[0] + interval * m)
        else:
            ind = ((x >= (x_range[0] + interval * m)) * (x < (x_range[0] + interval * (m + 1))))
        y += (x  * mus[m] + mus[m + M]) * ind

    return y

def plot_loss_functions(mus, path, x_range=[-1.0, 1.0]):

    plt.figure()

    x = np.linspace(x_range[0], x_range[1], num=50)

    y_mean = loss_(x, mus[0], x_range)
    plt.plot(x, y_mean, 'b-',label='mu')

    plt.plot(x, x, 'r-',label='base')

    y_best = loss_(x, mus[1], x_range)
    plt.plot(x, y_best, 'k-',label='best')
    plt.legend()

    # plt.fill_between(x, np.min(ys,0), np.max(ys,0),color='b',
    #              alpha=0.2)

    plt.xlim(x_range[0], x_range[1])
    plt.legend()

    plt.savefig(path)
    plt.close()
    # plt.show()
