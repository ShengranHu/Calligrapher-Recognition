import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils

import time
import utils
from data_process.calligraphy import get_data_transforms, calligraphy
from resnet import resnet50, resnet101
from PIL import Image

######change device according to the environment##########
device = 'cpu' # 'cuda' or 'cpu'

idx_to_class = {0: '张旭',
                1: '褚遂良',
                2: '赵孟頫',
                3: '钟绍京',
                4: '颜真卿'}

def load_model(model_name='resnet101', NUM_CLASSES=len(idx_to_class),
               ckpt_path='/store_point/weights-res101-balanced-97.19.pt'):

    if model_name == 'resnet101':
        model = resnet101(pretrained=False, progress=True, num_classes=NUM_CLASSES)
    else:
        raise NotImplementedError

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    model.to(device)

    return model

def infer(model, input_path):
    model.eval()

    img = Image.open(input_path)
    train_transform, valid_transform = get_data_transforms()
    img = valid_transform(img).unsqueeze(0)



    with torch.no_grad():
        start = time.time()

        model, img = model.to(device), img.to(device)
        pred = model(img)
        pred = F.softmax(pred, dim=1)
        _, predicted = pred.max(1)

        print("time cost: ", time.time() - start)

    if pred[0][predicted.item()].item() < 0.5:
        return "其它", pred[0][predicted.item()].item()
    else:
        return idx_to_class[predicted.item()], pred[0][predicted.item()].item()

def validate(valid_queue, net):

    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    # logging.info('valid acc %f', 100. * correct / total)

    return acc


if __name__ == '__main__':
    model = load_model()

    train_transform, valid_transform = get_data_transforms()

    train_data = calligraphy().get_train_data(transform=train_transform)
    valid_data = calligraphy().get_val_data(transform=valid_transform)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=64,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        shuffle=False,
        pin_memory=False, num_workers=16)

    acc = validate(valid_queue, model)
    print(acc)
    print(len(valid_data))