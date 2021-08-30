
import sys
# update your projecty root path before running


sys.path.insert(0, 'path/to/project')

import os
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
# import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from flops_counter import add_flops_counting_methods

import time
import utils
from data_process.calligraphy import get_data_transforms, calligraphy
from resnet import resnet50,resnet101

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./tensorboard_log')
device = 'cuda'


def main(save='train', expr_root='', seed=0, gpu=0):

    # ---- train logger ----------------- #
    save_pth = os.path.join(expr_root, '{}'.format(save))
    utils.create_exp_dir(save_pth)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    # fh = logging.FileHandler(os.path.join(save_pth, 'log.txt'))
    # fh.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(fh)

    # ---- parameter values setting ----- #

    NUM_CLASSES = 5
    learning_rate = 0.025
    momentum = 0.9
    weight_decay = 3e-4

    batch_size = 56
    report_freq = 50
    epochs = 100

    train_params = {
        'report_freq': report_freq,
    }

    model = resnet50(pretrained=False, progress=True, num_classes=NUM_CLASSES)
    #model = torch.hub.load('pytorch/vision:v0.8.2', 'resnet101', pretrained=True)

    # logging.info("Genome = %s", genome)

    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)

    n_params = (np.sum(np.prod(v.size()) for v in filter(lambda p: p.requires_grad, model.parameters())) / 1e6)
    model = model.to(device)

    logging.info("param size = %fMB", n_params)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    train_transform, valid_transform = get_data_transforms()

    train_data = calligraphy().get_train_data(transform=train_transform)
    valid_data = calligraphy().get_val_data(transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        shuffle=True,
        pin_memory=True, num_workers=16)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        shuffle=False,
        pin_memory=True, num_workers=16)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs))

    best_acc = 0

    save = time.strftime("%Y%m%d-%H%M%S")

    for epoch in range(epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])

        train_acc, train_obj = train(train_queue, model, criterion, optimizer, train_params)
        logging.info('train_acc %f', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('valid_acc', valid_acc, epoch)

        if valid_acc > best_acc:
            utils.save(model, os.path.join('./store_point', 'weights-{}.pt'.format(save)))
            best_acc = valid_acc

        print("best acc: ", best_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    # calculate for flops
    model = add_flops_counting_methods(model)
    model.eval()
    model.start_flops_count()
    random_data = torch.randn(1, 3, 32, 32)
    model(torch.autograd.Variable(random_data).to(device))
    n_flops = np.round(model.compute_average_flops_cost() / 1e6, 4)
    logging.info('flops = %f', n_flops)


    # logging.info("Architecture = %s", genotype))

    return {
        'valid_acc': valid_acc,
        'params': n_params,
        'flops': n_flops,
    }

# Training
def train(train_queue, net, criterion, optimizer, params):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for step, (inputs, targets) in enumerate(train_queue):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if step % params['report_freq'] == 0:
            logging.info('train %03d %e %f', step, train_loss/total, 100.*correct/total)

    logging.info('train acc %f', 100. * correct / total)

    return 100.*correct/total, train_loss/total



def infer(valid_queue, net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # if step % args.report_freq == 0:
            #     logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)

    acc = 100.*correct/total
    # logging.info('valid acc %f', 100. * correct / total)

    return acc, test_loss/total

if __name__ == '__main__':
    main()