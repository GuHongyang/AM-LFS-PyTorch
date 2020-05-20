import argparse
import os
import torch.multiprocessing as mp
from utils import *
from dataset import get_datasets, Sampler
from model import resnet20
from torch.optim import SGD, Adam
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.distributions import MultivariateNormal
import numpy as np
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt



class Samples(mp.Process):
    def __init__(self, datasets, epoch, lock, population, finish_tasks, device, args):
        super(Samples, self).__init__()

        self.trainer = Trainer(datasets, lock, epoch, population, finish_tasks, device, args)
        self.epoch = epoch
        self.args = args
        self.population = population

    def run(self):

        while True:

            if self.epoch.value >= self.args.epochs:
                break

            if self.population.empty():
                continue


            self.trainer.get_task()
            self.trainer.load_model()
            self.trainer.train(self.args.interval)
            self.trainer.validate()


class Optimizer(mp.Process):
    def __init__(self, datasets, epoch, lock, population, finish_tasks, device, args):
        super(Optimizer, self).__init__()

        self.args = args
        self.mus = Variable(torch.from_numpy(np.concatenate([np.ones([args.M, ]), np.zeros([args.M, ])])).float(),
                            requires_grad=True)
        self.optim_mus = Adam([self.mus], lr=0.05)
        self.acc_mean = -1
        self.acc_var = -1
        self.population = population
        self.finish_tasks = finish_tasks
        self.epoch = epoch
        self.lock = lock

        # self.writer = SummaryWriter(args.save_dir)

        self.trainer = Trainer(datasets, lock, epoch, population, finish_tasks, device, args)
        self.trainer.save_model(True)

        dist = MultivariateNormal(self.mus.detach(), torch.eye(2 * args.M) * args.sigma2)
        thetas = dist.sample((args.B,))
        for i in range(args.B):
            population.put(dict(id=i, theta=thetas[i]))


        self.t1_slopes = plot_scalers(os.path.join(args.save_dir, 't1_slopes.jpg'))
        self.t1_inter = plot_scalers(os.path.join(args.save_dir, 't1_inter.jpg'))
        # self.t2_slopes = plot_scalers(os.path.join(args.save_dir, 't2_slopes.jpg'))
        # self.t2_inter = plot_scalers(os.path.join(args.save_dir, 't2_inter.jpg'))
        self.test_acc = plot_scalers(os.path.join(args.save_dir, 'accs.jpg'))

        self.t1_slopes(0, {f'a_{i}':self.mus[i].item() for i in range(args.M)})
        self.t1_inter(0, {f'b_{i}':self.mus[i + args.M].item() for i in range(args.M)})

        # self.t2_slopes(0, {f'a_{i}': self.mus[i + args.M * 2].item() for i in range(args.M)})
        # self.t2_inter(0, {f'b_{i}': self.mus[i + args.M * 3].item() for i in range(args.M)})


    def run(self):
        while True:
            if self.epoch.value >= self.args.epochs:
                break

            if self.finish_tasks.full() and self.population.empty():

                with self.epoch.get_lock():
                    self.epoch.value = self.epoch.value + self.args.interval

                epoch = self.epoch.value - 1

                task = []
                accs = []
                while not self.finish_tasks.empty():
                    task.append(self.finish_tasks.get())
                    accs.append(task[-1]['acc'])

                task = sorted(task, key=lambda x: x['acc'], reverse=True)
                self.trainer.print_('Epoch {} best score on {} is {}%'.format(epoch, task[0]['id'],
                                                                                       task[0]['acc'] * 100))
                shutil.copyfile(os.path.join(self.args.save_dir, 'ckpt_{}.pth'.format(task[0]['id'])), os.path.join(self.args.save_dir, 'ckpt_best.pth'))


                mus1 = [self.mus.detach().numpy()[:2*self.args.M]]
                mus1.append(task[0]['theta'].numpy()[:2*self.args.M])

                # mus2 = [self.mus.detach().numpy()[2 * self.args.M:]]
                # mus2.append(task[0]['theta'].numpy()[2 * self.args.M:])

                # self.writer.add_figure('sample', plot_loss_functions(mus), global_step=epoch)
                plot_loss_functions(mus1, os.path.join(self.args.save_dir, 'mus_{}_t1.jpg'.format(epoch)))
                # plot_loss_functions(mus2, os.path.join(self.args.save_dir, 'mus_{}_t2.jpg'.format(epoch)))

                if self.acc_mean == -1:
                    self.acc_mean = np.mean(accs)
                    self.acc_var = np.var(accs)

                self.acc_mean = self.acc_mean * 0.99 + np.mean(accs) * 0.01
                self.acc_var = self.acc_var * 0.9 + np.var(accs) * 0.1

                loss_mu = 0
                dist = MultivariateNormal(self.mus, torch.eye(2 * self.args.M) * self.args.sigma2)
                for i in range(len(task)):
                    loss_mu -= dist.log_prob(task[i]['theta']) * (task[i]['acc'] - np.mean(accs)) / (np.std(accs) + np.finfo(np.float32).eps.item())
                loss_mu /= self.args.B * 1.0

                # self.writer.add_scalars('as',{'a_{}'.format(i):self.mus[i].item() for i in range(self.args.M)},epoch)
                # self.writer.add_scalars('bs',{'b_{}'.format(i):self.mus[i+self.args.M].item() for i in range(self.args.M)},epoch)

                self.t1_slopes(epoch + 1, {f'a_{i}': self.mus[i].item() for i in range(self.args.M)})
                self.t1_inter(epoch + 1, {f'b_{i}': self.mus[i + self.args.M].item() for i in range(self.args.M)})

                # self.t2_slopes(epoch + 1, {f'a_{i}': self.mus[i + self.args.M * 2].item() for i in range(self.args.M)})
                # self.t2_inter(epoch + 1, {f'b_{i}': self.mus[i + self.args.M * 3].item() for i in range(self.args.M)})


                self.optim_mus.zero_grad()
                loss_mu.backward()
                self.optim_mus.step()

                self.trainer.print_('Epoch {} : Mu Loss = {:.4f} mus={}'.format(epoch, loss_mu.item(), self.mus))


                self.trainer.load_model()
                # self.writer.add_scalar('test_acc',self.trainer.test(epoch),epoch)
                self.test_acc(epoch+1,{'acc':self.trainer.test(epoch)})

                dist = MultivariateNormal(self.mus.detach(), torch.eye(2 * self.args.M) * self.args.sigma2)
                self.lock.acquire()
                for i in range(self.args.B):
                    self.population.put(dict(id=task[i]['id'], theta=dist.sample()))
                self.lock.release()







if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AM-LFS')
    parser.add_argument('--exp', type=str, default='new_exp_')

    parser.add_argument('--B', type=int, default=32)
    parser.add_argument('--M', type=int, default=6)
    parser.add_argument('--sigma2', type=float, default=0.2)
    parser.add_argument('--gpus', type=str, default='0,4,5,8,9')
    parser.add_argument('--num_per_gpu', type=int, default=8)

    parser.add_argument('--train_bs', type=int, default=128)
    parser.add_argument('--test_bs', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--interval', type=int, default=1)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    create_exp_dir(args)
    mp.set_start_method("spawn")

    population = mp.Queue(maxsize=args.B)
    finish_tasks = mp.Queue(maxsize=args.B)
    test_outputs = mp.Queue()
    epoch = mp.Value('i', 0)
    lock = mp.Lock()

    resources=[]
    print('Using resources:')
    for i in range(1,len(args.gpus.split(','))):
        for j in range(args.num_per_gpu):
            resources.append(f'cuda:{i}')
            print(f'cuda:{i}')

    datasets = get_datasets()

    Processes = [Samples(datasets, epoch, lock, population, finish_tasks, resources[i], args)
                 for i in range(len(resources))]
    Processes.append(Optimizer(datasets, epoch, lock, population, finish_tasks, 'cuda:0', args))

    [p.start() for p in Processes]
    [p.join() for p in Processes]
