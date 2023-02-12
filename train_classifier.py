import os
import glob
import numpy as np
# import pandas as pd
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import time
import datetime
import random
import collections

from imaginaire.config import Config
# import matplotlib.pyplot as plt
import sys

'''
reference:
- https://www.kaggle.com/code/reukki/pytorch-cnn-tutorial-with-cats-and-dogs
- https://wtfleming.github.io/blog/pytorch-cats-vs-dogs-part-3/
- https://www.kaggle.com/code/basu369victor/pytorch-tutorial-the-classification/notebook
1. dataset
2. model
3. trainer
4. inference
5. visualisation
sinteractive -p shortgpgpu --time=1:00:00 --qos=gpgpuresplat
# module load gcccore/10.2.0
# module load python/3.8.6
# module load cudnn/8.0.4.30-cuda-11.1.1
module load fosscuda/2020b
module load jupyter/1.0.0-python-3.8.6
jupyter notebook
'''


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

class style_dict_dataset(Dataset):
    def __init__(self, style_dict):
        self.style_list = list(style_dict.items())

    def __getitem__(self, index):
        fn, style = self.style_list[index]
        return fn, style

    def __len__(self):
        return len(self.style_list)

class style_list_dataset(Dataset):
    def __init__(self, style_list, style_fname_list):
        if len(style_list) != len(style_fname_list):
            raise (RuntimeError(f'len(style_list) != len(style_fname_list): {len(style_list)} != {len(style_fname_list)}'))
        self.style_list = style_list
        self.style_fname_list = style_fname_list

    def __getitem__(self, index):
        fn, style = self.style_fname_list[index], self.style_list[index]
        return fn, style

    def __len__(self):
        return len(self.style_list)

def get_style_dict_loader(style_dict, batch_size):
    style_dict_data = style_dict_dataset(style_dict)
    style_dict_loader = torch.utils.data.DataLoader(dataset=style_dict_data, batch_size=batch_size, shuffle=False)
    return style_dict_loader

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


class ClassifierTrainer():
    def __init__(self, cfg, model, optimizer=None, scheduler=None, train_loader=None, val_loader=None, n_epochs=None):
        print('Setup trainer.')

        # Initialize models and data loaders.
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.is_inference = train_loader is None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device: {self.device}')
        self.model = self.model.to(self.device)

        self.classes = ('Dog', 'Cat')
        if self.is_inference:
            # The initialization steps below can be skipped during inference.
            return

        self.criterion = nn.CrossEntropyLoss()
        self.print_every = 100
        self.valid_loss_min = np.Inf
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

        self.total_step = len(self.train_loader)
        self.n_epochs = n_epochs
        # self.classes = {0: 'Dog', 1: 'Cat'}
        # self.classes = ('Dog', 'Cat')

        # Initialize logging attributes.
        self.current_iteration = 0
        self.current_epoch = 0
        self.start_epoch_time = None
        self.time_epoch = None

    def load_checkpoint(self, cfg=None, checkpoint_path=None, resume=None, load_sch=True):
        if os.path.exists(checkpoint_path):
            if resume is None:
                resume = False
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            current_epoch = 0
            current_iteration = 0

            if resume:
                self.model.load_state_dict(checkpoint['model'], strict=False)
                if not self.is_inference:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                    self.current_epoch = checkpoint['current_epoch']
                    self.train_loss = checkpoint['train_loss']
                    self.train_acc = checkpoint['train_acc']
                    self.val_loss = checkpoint['val_loss']
                    self.val_acc = checkpoint['val_acc']
                    current_epoch = self.current_epoch
            else:
                try:
                    self.model.load_state_dict(checkpoint['model'], strict=False)
                except Exception:
                    raise ValueError('Checkpoint cannot be loaded.')

            print('Done with loading the checkpoint.')
            return resume, current_epoch, current_iteration

        else:
            print('No checkpoint found.')
            current_epoch = 0
            current_iteration = 0
            resume = False
            return resume, current_epoch, current_iteration

    def save_checkpoint(self):
        latest_checkpoint_path = 'epoch_{:05}_checkpoint.pt'.format(self.current_epoch)
        save_path = os.path.join(self.cfg.logdir, latest_checkpoint_path)
        torch.save(
            {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'current_epoch': self.current_epoch,
                'train_loss': self.train_loss,
                'train_acc': self.train_acc,
                'val_loss': self.val_loss,
                'val_acc': self.val_acc
            },
            save_path,
        )
        fn = os.path.join(self.cfg.logdir, 'latest_checkpoint.txt')
        with open(fn, 'wt') as f:
            f.write('latest_checkpoint: %s' % latest_checkpoint_path)
        print('Save checkpoint to {}'.format(save_path))
        return save_path

    def net_update(self):
        # For Train
        running_loss = 0.0
        correct = 0
        total = 0
        correct_prediction = {classname: 0 for classname in self.classes}
        total_prediction = {classname: 0 for classname in self.classes}
        self.model.train()
        for i, data in enumerate(self.train_loader):
            self.current_iteration = i
            inputs, labels, _ = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == labels).item()
            total += labels.size(0)
            for label, prediction in zip(labels, pred):
                if label == prediction:
                    correct_prediction[self.classes[label.item()]] += 1
                total_prediction[self.classes[label.item()]] += 1
            if (self.current_iteration) % self.print_every == 0 or self.current_iteration == self.total_step - 1:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(self.current_epoch, self.n_epochs,
                                                                         self.current_iteration, self.total_step,
                                                                         loss.item()))

        self.train_acc.append(100 * correct / total)
        self.train_loss.append(running_loss / self.total_step)
        print(f'\ntrain loss: {np.mean(self.train_loss):.4f}, train acc: {(100 * correct / total):.4f}')
        # print accuracy for each class
        for classname, correct_count in correct_prediction.items():
            accuracy = 100 * float(correct_count) / total_prediction[classname]
            print(f'Train Accuracy for class: {classname:5s} is {accuracy:.1f} %')

        # For Val
        batch_loss = 0
        total_t = 0
        correct_t = 0
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}
        # print(f'correct_pred: {correct_pred}\n total_pred: {total_pred}')
        with torch.no_grad():
            self.model.eval()
            for data_t, target_t, _ in self.val_loader:
                data_t, target_t = data_t.to(self.device), target_t.to(self.device)
                outputs_t = self.model(data_t)
                loss_t = self.criterion(outputs_t, target_t)

                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)
                for label, prediction in zip(target_t, pred_t):
                    if label == prediction:
                        correct_pred[self.classes[label.item()]] += 1
                    total_pred[self.classes[label.item()]] += 1

            self.val_acc.append(100 * correct_t / total_t)
            self.val_loss.append(batch_loss / len(self.val_loader))
            network_learned = batch_loss < self.valid_loss_min
            print(f'validation loss: {np.mean(self.val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
            # Saving the best weight
            if network_learned:
                self.valid_loss_min = batch_loss
                # model_path = os.path.join(self.cfg.logdir, f'epoch_{self.current_epoch}_checkpoint.pt')
                # torch.save(self.model.state_dict(), model_path)
                self.save_checkpoint()
                print(f'Detected network improvement, saving current model')
        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Val Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    def start_of_epoch(self, current_epoch):
        self.current_epoch = current_epoch
        self.start_epoch_time = time.time()
        print(f'Epoch: {self.current_epoch}')

    def end_of_epoch(self):
        self.scheduler.step()
        elapsed_epoch_time = time.time() - self.start_epoch_time
        print('Epoch: {}, total time: {:6f}.'.format(self.current_epoch, elapsed_epoch_time))
        self.time_epoch = elapsed_epoch_time
        self.save_plot_and_sample_images()
        self.save_report_and_sample_vis()

    def vis_train_dataset(self):
        def img_display(img):
            img = img / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            npimg = np.transpose(npimg, (1, 2, 0))
            return npimg

        def vis1():
            import matplotlib.pyplot as plt
            # get some random training images
            dataiter = iter(self.train_loader)
            images, labels, _ = dataiter.next()
            plt.imshow(img_display(torchvision.utils.make_grid(images)))
            # plt.show()
            # print labels
            print(' '.join(f'{self.classes[labels[j]]:5s}' for j in range(self.cfg.data.train.batch_size)))

        def vis2():
            # get some random training images
            dataiter = iter(self.train_loader)
            images, labels, _ = dataiter.next()
            # Viewing data examples used for training
            fig, axis = plt.subplots(4, 4, figsize=(8, 10))
            for i, ax in enumerate(axis.flat):
                with torch.no_grad():
                    image, label = images[i], labels[i]
                    ax.imshow(img_display(image))  # add image
                    ax.set(title=f"{self.classes[label.item()]}")

        vis1()
        vis2()

    def save_plot_and_sample_images(self):
        import matplotlib.pyplot as plt
        def plot_loss():
            fig = plt.figure(figsize=(16, 9))
            plt.title("Train - Validation Loss")
            plt.plot(self.train_loss, label='train')
            plt.plot(self.val_loss, label='validation')
            plt.xlabel('num_epochs', fontsize=12)
            plt.ylabel('loss', fontsize=12)
            plt.legend(loc='best')
            print('saving {}'.format(os.path.join(self.cfg.logdir, f'epoch_{self.current_epoch}_loss.png')))
            fig.savefig(os.path.join(self.cfg.logdir, f'epoch_{self.current_epoch}_loss.png'), bbox_inches='tight')

        def plot_accuracy():
            fig = plt.figure(figsize=(16, 9))
            plt.title("Train - Validation Accuracy")
            plt.plot(self.train_acc, label='train')
            plt.plot(self.val_acc, label='validation')
            plt.xlabel('num_epochs', fontsize=12)
            plt.ylabel('accuracy', fontsize=12)
            plt.legend(loc='best')
            print('saving {}'.format(os.path.join(self.cfg.logdir, f'epoch_{self.current_epoch}_accuracy.png')))
            fig.savefig(os.path.join(self.cfg.logdir, f'epoch_{self.current_epoch}_accuracy.png'), bbox_inches='tight')

        def plot_sample():
            def img_display(img):
                img = img / 2 + 0.5  # unnormalize
                npimg = img.numpy()
                npimg = np.transpose(npimg, (1, 2, 0))
                return npimg

            dataiter = iter(self.val_loader)
            images, labels, _ = dataiter.next()

            import math
            ncols = 8  # math.ceil(batch_size / 4)
            nrows = math.ceil(self.cfg.data.train.batch_size / ncols)
            fig, axis = plt.subplots(nrows, ncols, figsize=(ncols * 1.1, nrows * 1.15))
            with torch.no_grad():
                self.model.eval()
                for ax, image, label in zip(axis.flat, images, labels):
                    ax.imshow(img_display(image))  # add image
                    image_tensor = image.unsqueeze_(0)
                    image_tensor = image_tensor.to(self.device)  # on GPU
                    output_ = self.model(image_tensor)
                    output_ = output_.argmax()
                    k = output_.item() == label.item()
                    ax.axis('off')
                    colour = {True: 'tab:blue', False: 'tab:orange'}
                    # 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
                    ax.set_title(str(self.classes[label.item()]) + ": " + str(k), fontsize=9, color=colour[k])

            fig.suptitle(f'Sample predictions accuracy for validation dataset (True for Correct)', fontsize=12)
            print('saving {}'.format(os.path.join(self.cfg.logdir, f'epoch_{self.current_epoch}_val_sample8.png')))
            fig.savefig(os.path.join(self.cfg.logdir, f'epoch_{self.current_epoch}_val_sample8.png'),
                        bbox_inches='tight')

        plot_loss()
        plot_accuracy()
        plot_sample()
        # print('Save output graphs and sample images to {}'.format(self.cfg.logdir))

    # ## 4. Test Model
    def save_report_and_sample_vis(self):
        def save_report(test_loader=self.val_loader):
            import pandas as pd
            cat_probs = []
            self.model.eval()
            with torch.no_grad():
                for images, labels, filepaths in test_loader:
                    images = images.to(self.device)
                    preds = self.model(images)
                    # {0: 'Dog', 1: 'Cat'}
                    preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
                    cat_probs += list(zip(list(labels.tolist()), preds_list, filepaths))
            cat_probs.sort(key=lambda x: x[1])  # sort by cat confidence
            idx = list(map(lambda x: x[0], cat_probs))
            prob = list(map(lambda x: x[1], cat_probs))
            filepath = list(map(lambda x: x[2], cat_probs))
            submission = pd.DataFrame({'label': idx, 'probability': prob, 'file': filepath})
            print('saving {}'.format(os.path.join(self.cfg.logdir, f'epoch_{self.current_epoch}_val_results.csv')))
            submission.to_csv(os.path.join(self.cfg.logdir, f'epoch_{self.current_epoch}_val_results.csv'), index=False)
            return submission

        def vis_report_sample(submission):
            import random
            import matplotlib.pyplot as plt
            id_list = []
            # classes = {0: 'dog', 1: 'cat'}
            fig, axes = plt.subplots(3, 5, figsize=(10, 6), facecolor='w')
            for ax in axes.ravel():
                i = random.choice(submission['file'].values)
                label = submission.loc[submission['file'] == i, 'probability'].values[0]
                if label > 0.5:
                    label = 1
                else:
                    label = 0
                img_path = i
                img = Image.open(img_path)
                ax.set_title(self.classes[label])
                ax.axis('off')
                ax.imshow(img)
            fig.suptitle(f'Sample predictions for validation dataset', fontsize=16)
            print('saving {}'.format(os.path.join(self.cfg.logdir, f'epoch_{self.current_epoch}_val_sample5.png')))
            fig.savefig(os.path.join(self.cfg.logdir, f'epoch_{self.current_epoch}_val_sample5.png'),
                        bbox_inches='tight')

        def vis_head_mid_tail(submission, ncols=10, nrows=10, width=15, heigt=15.8):
            n_imgs = nrows * ncols
            df_sort = submission.sort_values(by=['probability'], inplace=False, ascending=False)
            heads = df_sort.head(n_imgs)
            df_sort = submission.sort_values(by=['probability'], inplace=False, ascending=True)
            tails = df_sort.head(n_imgs)
            # mids = df_sort.loc[(df_sort.probability - 0.5).abs().argsort()].head(n_imgs)
            df_sort['close_to_mid'] = (df_sort.probability - 0.5).abs()
            mids = df_sort.sort_values(by=['close_to_mid'], inplace=False, ascending=True).head(n_imgs)
            # print('2', mids.head(10))

            for df, pos in zip([heads, tails, mids], ['Head', 'Tail', 'Middle']):
                fig, axis = plt.subplots(nrows, ncols, figsize=(width, heigt))
                for ax, img_path, probability, label in zip(axis.flat, df['file'].to_list(),
                                                            df['probability'].to_list(), df['label'].to_list()):
                    img = Image.open(img_path)
                    ax.imshow(img)
                    title = f'{probability:.2f} | {self.classes[label]}'
                    ax.axis('off')
                    ax.set_title(title)
                fig.suptitle(f'{pos} images for probability of {self.classes[1]} (Probability | Truth)', fontsize=16)
                print('saving {}'.format(os.path.join(self.cfg.logdir, f'epoch_{self.current_epoch}_{pos}_images.png')))
                fig.savefig(os.path.join(self.cfg.logdir, f'epoch_{self.current_epoch}_{pos}_images.png'),
                            bbox_inches='tight')

        submission = save_report(self.val_loader)
        vis_report_sample(submission)
        vis_head_mid_tail(submission)

    def inference(self, images):
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            preds = self.model(images)
            # {0: 'Dog', 1: 'Cat'}
            preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
        return preds_list

    def inference_one_image(self, image):
        self.model.eval()
        with torch.no_grad():
            image_tensor = image.unsqueeze_(0)
            image_tensor = image_tensor.to(self.device)
            preds = self.model(image_tensor)
            # {0: 'Dog', 1: 'Cat'}
            preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
        assert len(preds_list) == 1, 'Error: found not len(preds_list) == 1!'
        return preds_list[0]

    def get_transform(self, normalize=True, num_channels=3):
        r"""Convert numpy to torch tensor.

        Args:
            normalize (bool): Normalize image i.e. (x - 0.5) * 2.
                Goes from [0, 1] -> [-1, 1].
        Returns:
            Composed list of torch transforms.
        """
        transform_list = [transforms.ToTensor()]
        if normalize:
            transform_list.append(
                transforms.Normalize((0.5,) * num_channels,
                                     (0.5,) * num_channels, inplace=True))
        return transforms.Compose(transform_list)

    def tranform_image(self, image):
        transform_norm = self.get_transform(normalize=True, num_channels=3)
        img_normalized = transform_norm(image)
        return img_normalized


# Some utility functions
def set_random_seed(seed):
    print(f"Using random seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_logging(config_path, logdir):
    config_file = os.path.basename(config_path)
    root_dir = 'logs_classifier'
    date_uid = str(datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S"))
    # example: logs_classifier/2019_0125_1047_58_spade_cocostuff
    log_file = '_'.join([date_uid, os.path.splitext(config_file)[0]])
    if logdir is None:
        logdir = os.path.join(root_dir, log_file)
    return date_uid, logdir


def make_logging_dir(logdir):
    print('Make folder {}'.format(logdir))
    os.makedirs(logdir, exist_ok=True)
    # tensorboard_dir = os.path.join(logdir, 'tensorboard')
    # os.makedirs(tensorboard_dir, exist_ok=True)
    # set_summary_writer(tensorboard_dir)


def get_train_and_val_dataloader(cfg):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(256),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dir = cfg.data.train.roots[0]
    val_dir = cfg.data.val.roots[0]
    # test_dir = cfg.test_data.test.roots[0]
    batch_size = cfg.data.train.batch_size
    print(f'train_dir: {train_dir}\n val_dir: {val_dir}\n batch_size: {batch_size}')

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

    # test_images_a = glob.glob(os.path.join(test_dir, images_a, '*.jpg'))  # [200:500]
    # test_images_b = glob.glob(os.path.join(test_dir, images_b, '*.jpg'))  # [200:500]
    # test_list = test_images_a + test_images_b
    # print(f'len(test_list): {len(test_list)}')

    train_data = dataset(train_list, transform=transform)
    val_data = dataset(val_list, transform=transform)
    # test_data = dataset(test_list, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    print(f'len(train_data): {len(train_data)} \t len(train_loader): {len(train_loader)}')
    print(f'len(val_data): {len(val_data)} \t len(val_loader): {len(val_loader)}')
    print(f'image size: {train_data[0][0].shape}')

    return train_loader, val_loader


def to_device(data, device):
    assert device in ['cpu', 'cuda']
    if isinstance(data, torch.Tensor):
        data = data.to(torch.device(device))
        return data
    elif isinstance(data, collections.abc.Mapping):
        return {key: to_device(data[key], device) for key in data}
    elif isinstance(data, collections.abc.Sequence) and \
            not isinstance(data, (str, bytes)):
        return [to_device(d, device) for d in data]
    else:
        return data


def to_cuda(data):
    return to_device(data, 'cuda')


def to_cpu(data):
    return to_device(data, 'cpu')


# ## 3. Train Model
def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', help='Path to the training config file.')#, required=True)
    parser.add_argument('--logdir', help='Dir for saving logs and models.')
    parser.add_argument('--checkpoint', default='', help='Checkpoint path.')
    parser.add_argument('--seed', type=int, default=2, help='Random seed.')
    parser.add_argument('--batch_size_multiplier', type=int, default=1, help='batch_size multiplier.')
    parser.add_argument('--n_epochs', type=int, default=100, help='Max num of epochs.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', type=int)

    ##--The following 4 hp are only used for multi mode inference--##
    parser.add_argument('--multi_model_inference', action='store_true')
    parser.add_argument('--output_dir_inference', help='Dir for saving inference results.')
    parser.add_argument('--config_inference', help='Path to the inference config file.')
    parser.add_argument('--redirect_stdout', action='store_true')
    ##-------------------------------------------------------------##

    args = parser.parse_args()
    return args


def main(redirect_stdout=False):
    args = parse_args()
    if args.multi_model_inference:
        main_inference(args, redirect_stdout=args.redirect_stdout)
        return

    set_random_seed(args.seed)
    cfg = Config(args.config)
    cfg.date_uid, cfg.logdir = init_logging(args.config, args.logdir)
    make_logging_dir(cfg.logdir)
    if redirect_stdout:
        log_file = os.path.join(cfg.logdir, 'logging.txt')
        print(f'main: redirecting sys.stdout to file {log_file}')
        Origin_Stdout = sys.stdout
        sys.stdout = open(log_file, "a")

    # Initialize data loaders and models.
    cfg.data.train.batch_size = cfg.data.train.batch_size * args.batch_size_multiplier
    train_loader, val_loader = get_train_and_val_dataloader(cfg)

    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1)

    trainer = ClassifierTrainer(cfg, model, optimizer, scheduler, train_loader, val_loader, args.n_epochs)
    resumed, current_epoch, current_iteration = trainer.load_checkpoint(cfg, args.checkpoint, args.resume)

    # Start training.
    for epoch in range(current_epoch, args.n_epochs):
        print('Epoch {} ...'.format(epoch))
        trainer.start_of_epoch(epoch)
        trainer.net_update()
        trainer.end_of_epoch()
        if redirect_stdout:
            sys.stdout.close()
            sys.stdout = open(log_file, "a")
    print('Done with training!!!')

    if redirect_stdout:
        sys.stdout.close()
        sys.stdout = Origin_Stdout
        print("sys.stdout recovered.")

    return


# if __name__ == "__main__":
#     main(redirect_stdout=True)


def main_inference(args, redirect_stdout=False):
    # args = parse_args()
    # set_random_seed(args.seed)
    cfg = get_config(args.config_inference)
    # print(f'cfg: {cfg}')
    output_dir = args.output_dir_inference
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f'created output dir {output_dir}')

    if redirect_stdout:
        log_file = os.path.join(output_dir, 'logging.txt')
        print(f'main: redirecting sys.stdout to file {log_file}')
        Origin_Stdout = sys.stdout
        sys.stdout = open(log_file, "a")
        print(f'\n\n{datetime.datetime.now().strftime("%d-%b-%Y, %H:%M:%S.%f")}')

    test_loader = get_test_dataloader(cfg)
    tester = MultiModelTester(cfg, test_loader)
    tester.load_checkpoint()
    tester.multi_model_inference(output_dir)
    print('Done with multi_model_inference!!!')

    if redirect_stdout:
        sys.stdout.close()
        sys.stdout = Origin_Stdout
        print("sys.stdout recovered.")

    return

def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def get_test_dataloader(cfg):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.RandomResizedCrop(256),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # test_dir = cfg['test_data']['roots'][0]
    batch_size = cfg['batch_size']
    # print(f'test_dir: {test_dir}')
    # test_list = glob.glob(os.path.join(test_dir, '*.jpg')) #f'*.{cfg.ext}'

    test_list = []
    for test_dir in cfg['test_data']['roots']:
        print(f'test_dir: {test_dir}')
        test_list += glob.glob(os.path.join(test_dir, '*.jpg'))  # f'*.{cfg.ext}'

    print(f'len(test_list): {len(test_list)}')
    test_data = test_dataset(test_list, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    print(f'len(test_data): {len(test_data)} \t len(test_loader): {len(test_loader)}')
    print(f'image size: {test_data[0][0].shape}')
    return test_loader

class test_dataset(Dataset):
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
        return image, img_path

class MultiModelTester():
    def __init__(self, cfg, test_loader):
        print('Setup MultiModelTester...')
        self.cfg = cfg
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device: {self.device}')

        self.N_models = len(self.cfg['check_points'])
        print(f'number of classifiers: {self.N_models}\n They are {self.cfg["check_points"]}')
        self.models = [Net().to(self.device) for i in range(self.N_models)]
        self.model_names = ['' for i in range(self.N_models)]
        # self.classes = ('0', '1')

    def load_checkpoint(self):
        for i, (k, v) in enumerate(self.cfg['check_points'].items()):
            if os.path.exists(v):
                try:
                    self.model_names[i] = k
                    checkpoint = torch.load(v)
                    self.models[i].load_state_dict(checkpoint['model'], strict=False)
                except Exception:
                    raise ValueError(f'Checkpoint {k}: {v} cannot be loaded.')
                print(f'checkpoint loaded. {k}: {v}')
            else:
                print(f'Nooooooooo! No checkpoint found for {k}: {v}')

    def multi_model_inference(self, output_dir, output_fn='classifier_scores.pkl'):
        score_dict = {}
        paths_list = []
        scores_list =[]
        for model in self.models:
            model.eval()

        with torch.no_grad():
            for images, paths in self.test_loader:
                # print(f'paths: {paths}') # paths is a tuple of path
                images = images.to(self.device)

                scores_modeli = []
                for model in self.models:
                    preds = model(images)
                    preds_for_1 = F.softmax(preds, dim=1)[:, 1]#.tolist()
                    scores_modeli.append(preds_for_1)

                scores = torch.cat([x.unsqueeze(-1) for x in scores_modeli], -1)
                # scores = scores.detach().cpu().squeeze().numpy() #don't squeeze
                scores = scores.detach().cpu().numpy()

                for i, path in enumerate(paths):
                    score_dict[path] = scores[i, :]
                paths_list.append(paths)
                scores_list.append(scores)

        data_paths = np.concatenate(tuple(np.asarray(x) for x in paths_list), axis=0)
        data_scores = np.concatenate(scores_list, axis=0)
        data_save = {'paths': data_paths, 'scores': data_scores, 'models': self.model_names}

        scores_pkl = os.path.join(output_dir, output_fn)
        print('Saving multi model classifier scores to {}'.format(scores_pkl))
        import pickle
        with open(scores_pkl, 'wb') as f:
            # pickle.dump(score_dict, f)
            pickle.dump(data_save, f)


if __name__ == "__main__":
    main(redirect_stdout=True)