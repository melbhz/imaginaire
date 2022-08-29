"""
# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import argparse
import os
import sys
import random

import torch.autograd.profiler as profiler
import wandb

import imaginaire.config
from imaginaire.config import Config
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.dataset import get_train_and_val_dataloader
from imaginaire.utils.distributed import init_dist, is_master, get_world_size
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.utils.misc import slice_tensor
from imaginaire.utils.logging import init_logging, make_logging_dir
from imaginaire.utils.trainer import (get_model_optimizer_and_scheduler,
                                      get_trainer, set_random_seed)

sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config',
                        help='Path to the training config file.', required=True)
    parser.add_argument('--logdir', help='Dir for saving logs and models.')
    parser.add_argument('--checkpoint', default='', help='Checkpoint path.')
    parser.add_argument('--seed', type=int, default=2, help='Random seed.')
    parser.add_argument('--randomized_seed', action='store_true', help='Use a random seed between 0-10000.')
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_jit', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_name', default='default', type=str)
    parser.add_argument('--wandb_id', type=str)
    parser.add_argument('--resume', type=int)
    parser.add_argument('--num_workers', type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_affinity(args.local_rank)
    if args.randomized_seed:
        args.seed = random.randint(0, 10000)
    set_random_seed(args.seed, by_rank=True)
    cfg = Config(args.config)
    try:
        from userlib.auto_resume import AutoResume
        AutoResume.init()
    except:  # noqa
        pass

    # If args.single_gpu is set to True,
    # we will disable distributed data parallel
    if not args.single_gpu:
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank)
    print(f"Training with {get_world_size()} GPUs.")

    # Global arguments.
    imaginaire.config.DEBUG = args.debug
    imaginaire.config.USE_JIT = args.use_jit

    # Override the number of data loading workers if necessary
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    # Create log directory for storing training results.
    cfg.date_uid, cfg.logdir = init_logging(args.config, args.logdir)
    make_logging_dir(cfg.logdir)

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Initialize data loaders and models.
    batch_size = cfg.data.train.batch_size
    total_step = max(cfg.trainer.dis_step, cfg.trainer.gen_step)
    cfg.data.train.batch_size *= total_step
    train_data_loader, val_data_loader = get_train_and_val_dataloader(cfg, args.seed)
    net_G, net_D, opt_G, opt_D, sch_G, sch_D = \
        get_model_optimizer_and_scheduler(cfg, seed=args.seed)
    trainer = get_trainer(cfg, net_G, net_D,
                          opt_G, opt_D,
                          sch_G, sch_D,
                          train_data_loader, val_data_loader)
    resumed, current_epoch, current_iteration = trainer.load_checkpoint(cfg, args.checkpoint, args.resume)

    # Initialize Wandb.
    if is_master():
        if args.wandb_id is not None:
            wandb_id = args.wandb_id
        else:
            if resumed and os.path.exists(os.path.join(cfg.logdir, 'wandb_id.txt')):
                with open(os.path.join(cfg.logdir, 'wandb_id.txt'), 'r+') as f:
                    wandb_id = f.read()
            else:
                wandb_id = wandb.util.generate_id()
                with open(os.path.join(cfg.logdir, 'wandb_id.txt'), 'w+') as f:
                    f.write(wandb_id)
        wandb_mode = "disabled" if (args.debug or not args.wandb) else "online"
        wandb.init(id=wandb_id,
                   project=args.wandb_name,
                   config=cfg,
                   name=os.path.basename(cfg.logdir),
                   resume="allow",
                   settings=wandb.Settings(start_method="fork"),
                   mode=wandb_mode)
        wandb.config.update({'dataset': cfg.data.name})
        wandb.watch(trainer.net_G_module)
        wandb.watch(trainer.net_D.module)

    # Start training.
    for epoch in range(current_epoch, cfg.max_epoch):
        print('Epoch {} ...'.format(epoch))
        if not args.single_gpu:
            train_data_loader.sampler.set_epoch(current_epoch)
        trainer.start_of_epoch(current_epoch)
        for it, data in enumerate(train_data_loader):
            with profiler.profile(enabled=args.profile,
                                  use_cuda=True,
                                  profile_memory=True,
                                  record_shapes=True) as prof:
                data = trainer.start_of_iteration(data, current_iteration)

                for i in range(cfg.trainer.dis_step):
                    trainer.dis_update(
                        slice_tensor(data, i * batch_size,
                                     (i + 1) * batch_size))
                for i in range(cfg.trainer.gen_step):
                    trainer.gen_update(
                        slice_tensor(data, i * batch_size,
                                     (i + 1) * batch_size))

                current_iteration += 1
                trainer.end_of_iteration(data, current_epoch, current_iteration)
                if current_iteration >= cfg.max_iter:
                    print('Done with training!!!')
                    return
            if args.profile:
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                prof.export_chrome_trace(os.path.join(cfg.logdir, "trace.json"))
            try:
                if AutoResume.termination_requested():
                    trainer.save_checkpoint(current_epoch, current_iteration)
                    AutoResume.request_resume()
                    print("Training terminated. Returning")
                    return 0
            except:  # noqa
                pass

        current_epoch += 1
        trainer.end_of_epoch(data, current_epoch, current_iteration)
    print('Done with training!!!')
    return


if __name__ == "__main__":
    main()
"""

# !/usr/bin/env python
# coding: utf-8

# # Imaginaire Classifier from and Cats and Dogs
# reference:
# - https://www.kaggle.com/code/reukki/pytorch-cnn-tutorial-with-cats-and-dogs
# - https://wtfleming.github.io/blog/pytorch-cats-vs-dogs-part-3/
# - https://www.kaggle.com/code/basu369victor/pytorch-tutorial-the-classification/notebook
# 1. dataset
# 2. model
# 3. trainer
# 4. inference
# 5. visualisation

# In[1]:


'''
sinteractive -p shortgpgpu --time=1:00:00 --qos=gpgpuresplat
# module load gcccore/10.2.0
# module load python/3.8.6
# module load cudnn/8.0.4.30-cuda-11.1.1
module load fosscuda/2020b
module load jupyter/1.0.0-python-3.8.6
jupyter notebook
'''

import matplotlib.pyplot as plt

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import zipfile

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ## 1. Datasets
class dataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        label = img_path.split('/')[-2]
        if label == 'images_a':
            label = 0
        elif label == 'images_b':
            label = 1
        else:
            print(f'Error: wrong label found! label = {label}')
        return image, label, img_path


# ## 2. Model Nets
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(3 * 3 * 64, 10)
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = torch.flatten(x, 1) # torch.flatten(input, start_dim=0, end_dim=-1)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def print_architecture(self):
        model = Net()  # On CPU
        # model = Net().to(device)  # On GPU
        print(model)


def vis_train_dataset(train_loader, classes, batch_size, train_list):
    def img_display(img):
        # img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        return npimg

    def vis1():
        # get some random training images
        dataiter = iter(train_loader)
        images, labels, _ = dataiter.next()

        # show images
        # img_display(torchvision.utils.make_grid(images))
        plt.imshow(img_display(torchvision.utils.make_grid(images)))
        plt.show()
        # print labels
        print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    def vis2():
        # get some random training images
        dataiter = iter(train_loader)
        images, labels, _ = dataiter.next()
        classes = {0: 'Dog', 1: 'Cat'}
        # Viewing data examples used for training
        fig, axis = plt.subplots(4, 4, figsize=(8, 10))
        for i, ax in enumerate(axis.flat):
            with torch.no_grad():
                image, label = images[i], labels[i]
                ax.imshow(img_display(image))  # add image
                ax.set(title=f"{classes[label.item()]}")

    def vis3():
        random_idx = np.random.randint(1, 250, size=10)
        fig = plt.figure()
        i = 1
        for idx in random_idx:
            ax = fig.add_subplot(2, 5, i)
            img = Image.open(train_list[idx])
            plt.imshow(img)
            i += 1

        plt.axis('off')
        plt.show()

    vis1()
    vis2()


def train_notused(model, train_loader, val_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        epoch_loss = 0
        epoch_accuracy = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels, _ = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = ((outputs.argmax(dim=1) == labels).float().mean())
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

            with torch.no_grad():
                running_loss += loss.item()
                if i % 200 == 199:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch + 1, epoch_accuracy, epoch_loss))

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label, _ in val_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = ((val_output.argmax(dim=1) == label).float().mean())
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)

            print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch + 1, epoch_val_accuracy, epoch_val_loss))


def after_epoch(epoch, path, train_loss, train_acc, val_loss, val_acc, model, val_loader, classes, batch_size, device):
    def plot_loss(epoch, path='logs'):
        fig = plt.figure(figsize=(16, 9))
        plt.title("Train - Validation Loss")
        plt.plot(train_loss, label='train')
        plt.plot(val_loss, label='validation')
        plt.xlabel('num_epochs', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.legend(loc='best')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f'epoch_{epoch}_loss.png'), bbox_inches='tight')

    def plot_accuracy(epoch, path='logs'):
        fig = plt.figure(figsize=(16, 9))
        plt.title("Train - Validation Accuracy")
        plt.plot(train_acc, label='train')
        plt.plot(val_acc, label='validation')
        plt.xlabel('num_epochs', fontsize=12)
        plt.ylabel('accuracy', fontsize=12)
        plt.legend(loc='best')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f'epoch_{epoch}_accuracy.png'), bbox_inches='tight')

    def img_display(img):
        # img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        return npimg

    def plot_sample(epoch, path='logs'):
        dataiter = iter(val_loader)
        images, labels, _ = dataiter.next()

        import math
        ncols = math.ceil(batch_size / 4)
        nrows = math.ceil(batch_size / ncols)
        fig, axis = plt.subplots(nrows, ncols, figsize=(12, 14))
        with torch.no_grad():
            model.eval()
            for ax, image, label in zip(axis.flat, images, labels):
                ax.imshow(img_display(image))  # add image
                image_tensor = image.unsqueeze_(0)
                image_tensor = image_tensor.to(device)  # on GPU
                output_ = model(image_tensor)
                output_ = output_.argmax()
                k = output_.item() == label.item()
                ax.set_title(str(classes[label.item()]) + ": " + str(k))

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, f'epoch_{epoch}_val_sample.png'), bbox_inches='tight')

    plot_loss(epoch)
    plot_accuracy(epoch)
    plot_sample(epoch)


def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()


def train(model, device, epochs, train_loader, val_loader, classes, batch_size, log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    n_epochs = epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print_every = 100
    valid_loss_min = np.Inf
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    total_step = len(train_loader)
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        # scheduler.step(epoch)
        correct = 0
        total = 0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_, _) in enumerate(train_loader):
            data_, target_ = data_.to(device), target_.to(device)  # on GPU
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target_).item()
            total += target_.size(0)
            if (batch_idx) % print_every == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / total_step)
        print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')

        batch_loss = 0
        total_t = 0
        correct_t = 0
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        with torch.no_grad():
            model.eval()
            for data_t, target_t, _ in (val_loader):
                data_t, target_t = data_t.to(device), target_t.to(device)  # on GPU
                outputs_t = model(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)

                for label, prediction in zip(target_t, pred_t):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

            val_acc.append(100 * correct_t / total_t)
            val_loss.append(batch_loss / len(val_loader))
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

            # Saving the best weight
            if network_learned:
                valid_loss_min = batch_loss
                model_path = os.path.join(log_dir, f'epoch_{epoch}_checkpoint.pt')
                torch.save(model.state_dict(), model_path)
                print(f'Detected network improvement, saving current model: {model_path}')

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Val Accuracy for class: {classname:5s} is {accuracy:.1f} %')

        after_epoch(epoch, log_dir, train_loss, train_acc, val_loss, val_acc, model, val_loader, classes, batch_size, device)
        model.train()


# ## 4. Test Model
def save_report(model, device, test_loader):
    cat_probs = []
    model.eval()
    with torch.no_grad():
        for data, fileid, filepath in test_loader:
            data = data.to(device)
            preds = model(data)
            preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
            cat_probs += list(zip(list(fileid.tolist()), preds_list, filepath))

    cat_probs.sort(key=lambda x: x[2])

    idx = list(map(lambda x: x[0], cat_probs))
    prob = list(map(lambda x: x[1], cat_probs))
    filepath = list(map(lambda x: x[2], cat_probs))

    submission = pd.DataFrame({'id': idx, 'label': prob, 'file': filepath})
    submission.to_csv('result.csv', index=False)
    return submission


def vis_report_sample(submission, test_dir, classes):
    import random
    id_list = []
    # classes = {0: 'dog', 1: 'cat'}
    fig, axes = plt.subplots(2, 5, figsize=(20, 12), facecolor='w')
    for ax in axes.ravel():
        i = random.choice(submission['file'].values)
        label = submission.loc[submission['file'] == i, 'label'].values[0]
        if label > 0.5:
            label = 1
        else:
            label = 0

        img_path = os.path.join(test_dir, i)
        img = Image.open(img_path)
        ax.set_title(classes[label])
        ax.imshow(img)
    fig.savefig('test_sample.png', bbox_inches='tight')


def save_and_vis(model, test_dir, classes):
    submission = save_report(model)
    vis_report_sample(submission, test_dir, classes)


# ## 3. Train Model
def main():
    # base_dir = '/data/scratch/projects/punim1358/HZ_GANs/imaginaire/Experiments_PAPER/base10_zoom20/AlcDrink/'
    # train_dir = '/data/scratch/projects/punim1358/Datasets/NSW_SA2/AlcoholDrinksPerWeek/Combined_k2_s1_p256/train'
    # val_dir = '/data/scratch/projects/punim1358/Datasets/NSW_SA2/AlcoholDrinksPerWeek/Combined_k2_s1_p256/val'
    # test_dir = '/data/scratch/projects/punim1358/Datasets/NSW_SA2/AlcoholDrinksPerWeek/Combined_k2_s1_p256/val'
    train_dir = '/data/scratch/projects/punim1358/HZ_GANs/imaginaire_nonscratch/dataset/afhq_dog2cat_raw/train_bak/'
    val_dir = '/data/scratch/projects/punim1358/HZ_GANs/imaginaire_nonscratch/dataset/afhq_dog2cat_raw/test_bak/'
    test_dir = '/data/scratch/projects/punim1358/HZ_GANs/imaginaire_nonscratch/dataset/afhq_dog2cat_raw/test_bak/'
    batch_size = 16
    epochs = 2
    number_of_labels = 2
    classes = ('Dog', 'Cat')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(256),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    images_a = 'images_a'
    images_b = 'images_b'
    train_images_a = glob.glob(os.path.join(train_dir, images_a, '*.jpg'))  # [:2000]
    train_images_b = glob.glob(os.path.join(train_dir, images_b, '*.jpg'))  # [:2000]
    train_list = train_images_a + train_images_b
    print(f'len(train_list): {len(train_list)}')
    val_images_a = glob.glob(os.path.join(val_dir, images_a, '*.jpg'))  # [:200]
    val_images_b = glob.glob(os.path.join(val_dir, images_b, '*.jpg'))  # [:200]
    val_list = val_images_a + val_images_b
    print(f'len(val_list): {len(val_list)}')
    test_images_a = glob.glob(os.path.join(test_dir, images_a, '*.jpg'))  # [200:500]
    test_images_b = glob.glob(os.path.join(test_dir, images_b, '*.jpg'))  # [200:500]
    test_list = test_images_a + test_images_b
    print(f'len(test_list): {len(test_list)}')

    train_data = dataset(train_list, transform=transform)
    test_data = dataset(test_list, transform=transform)
    val_data = dataset(val_list, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    print(f'len(train_data): {len(train_data)} \t len(train_loader): {len(train_loader)}')
    print(f'len(val_data): {len(val_data)} \t len(val_loader): {len(val_loader)}')
    print(f'image size: {train_data[0][0].shape}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    torch.manual_seed(2)
    if device == 'cuda:0':
        torch.cuda.manual_seed_all(2)
    model = Net().to(device)
    model.train()

    train(model, device, epochs, train_loader, val_loader, classes, batch_size, log_dir='logs')


if __name__ == "__main__":
    main()
