import os
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from model import resnet20
from loss import loss
from utils import printer
from dataset import Sampler


class Trainer:
    def __init__(self, datasets, lock, epoch, population, finish_tasks, device, args):
        self.model = resnet20().to(device)
        self.optim = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        self.device = device
        self.args = args
        self.datasets = datasets
        self.lock = lock
        self.population = population
        self.finish_tasks = finish_tasks
        self.epoch_base = epoch
        self.printer = printer(args)

        self.log_dir = os.path.join(args.save_dir, 'log.txt')

    def get_task(self):

        task = self.population.get()
        self.task_id = task['id']
        self.theta = task['theta']
        self.epoch = self.epoch_base.value


    def adjust_lr(self, lr0=0.1, steps=[100, 150]):

        n = 1
        for step in steps:
            if step > self.epoch:
                break
            else:
                n += 1
        cur_lr = lr0 ** n

        for param_group in self.optim.param_groups:
            param_group['lr'] = cur_lr

        return cur_lr

    def train_on_epoch(self):
        self.model.train()
        lr = self.adjust_lr()

        train_loader = torch.utils.data.DataLoader(self.datasets['train'], batch_size=self.args.train_bs, shuffle=True,
                                                   num_workers=1, drop_last=True)
        # train_loader = torch.utils.data.DataLoader(self.datasets['train'], batch_size=self.args.train_bs, sampler = Sampler(self.datasets['train'],self.epoch),
        #                                                                                       num_workers=1, drop_last=True)
        p_bar = tqdm(train_loader,
                     desc='Train (task {}, lr {}, epoch {}, device {})'.format(self.task_id, lr, self.epoch, self.device), ncols=120,
                     leave=True)
        p_bar.L = 0
        for bi, (x, y) in enumerate(p_bar):
            x = x.to(self.device)
            y = y.to(self.device)

            # if bi == 0:
            #     self.print_('{}'.format(y))

            output = self.model(x)

            l = loss(output, y, self.args.M, self.theta)

            self.optim.zero_grad()
            l.backward()
            self.optim.step()

            p_bar.L = (p_bar.L * bi + l.item()) / (bi + 1)
            p_bar.set_postfix_str('loss={:.4f}'.format(p_bar.L))
        self.print_('Train (task {}, lr {}, epoch {}, device {}, loss {})'.format(self.task_id, lr, self.epoch, self.device,
                                                                                             p_bar.L))

    def train(self, epochs):
        epoch = self.epoch
        for self.epoch in range(epoch, epoch + epochs):
            self.train_on_epoch()

    def validate(self):
        self.model.eval()
        valid_loader = torch.utils.data.DataLoader(self.datasets['valid'], batch_size=self.args.test_bs, shuffle=False,
                                                   num_workers=1)

        with torch.no_grad():
            p_bar = tqdm(valid_loader, desc='Valid (task {}, epoch {})'.format(self.task_id, self.epoch), ncols=120,
                         leave=True)
            p_bar.N = 0
            p_bar.S = 0
            for x, y in p_bar:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x).argmax(axis=1)
                p_bar.N += (output == y).sum().item()
                p_bar.S += x.size(0)

                p_bar.set_postfix_str('acc={:.4f}%'.format(p_bar.N * 1.0 / p_bar.S * 100.0))

            acc = p_bar.N * 1.0 / p_bar.S
        self.print_(
            'Valid (task {}, epoch {}, acc {}%)'.format(self.task_id, self.epoch, acc * 100))

        self.save_model()

        self.finish_tasks.put(dict(id=self.task_id, acc=acc, theta=self.theta))

        return acc


    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            p_bar = tqdm(
                torch.utils.data.DataLoader(self.datasets['test'], batch_size=self.args.test_bs, shuffle=False,
                                            num_workers=1), desc='Test (epoch {})'.format(epoch),
                ncols=120, leave=True)
            p_bar.N = 0
            p_bar.S = 0
            for batch in p_bar:
                x, y = batch
                x = x.to('cuda:0')
                y = y.to('cuda:0')

                output = self.model(x).argmax(axis=1)
                p_bar.N += (output == y).sum().item()
                p_bar.S += x.size(0)

                p_bar.set_postfix_str('acc={:.4f}%'.format(p_bar.N * 1.0 / p_bar.S * 100.0))

            acc = p_bar.N * 1.0 / p_bar.S
        self.print_('Test (epoch {}, acc {}%)'.format(epoch, acc * 100))
        return acc

    def load_model(self):


        load_path = os.path.join(self.args.save_dir, 'ckpt_best.pth')

        ckpt = torch.load(load_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)

        # self.optim.load_state_dict(ckpt['optim_state_dict'])
        # for state in self.optim.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(self.device)

        self.optim = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    def save_model(self, best=False):
        model_sd = self.model.state_dict()
        for k, v in model_sd.items():
            model_sd[k] = v.cpu()

        for state in self.optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        optim_sd = self.optim.state_dict()

        checkpoint = dict(model_state_dict=model_sd, optim_state_dict=optim_sd)
        if best:
            torch.save(checkpoint, os.path.join(self.args.save_dir, 'ckpt_best.pth'))

        else:
            torch.save(checkpoint, os.path.join(self.args.save_dir, 'ckpt_{}.pth'.format(self.task_id)))


    def print_(self, msg):
        self.lock.acquire()
        self.printer(msg)
        self.lock.release()





