# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import json
import os
import time

import torch
import torchvision
import wandb
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from imaginaire.utils.distributed import is_master, master_only
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.io import save_pilimage_in_jpeg
from imaginaire.utils.meters import Meter
from imaginaire.utils.misc import to_cuda, to_device, requires_grad, to_channels_last
from imaginaire.utils.model_average import (calibrate_batch_norm_momentum,
                                            reset_batch_norm)
from imaginaire.utils.visualization import tensor2pilimage

import numpy as np
import shutil
import math


class BaseTrainer(object):
    r"""Base trainer. We expect that all trainers inherit this class.

    Args:
        cfg (obj): Global configuration.
        net_G (obj): Generator network.
        net_D (obj): Discriminator network.
        opt_G (obj): Optimizer for the generator network.
        opt_D (obj): Optimizer for the discriminator network.
        sch_G (obj): Scheduler for the generator optimizer.
        sch_D (obj): Scheduler for the discriminator optimizer.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.
    """

    def __init__(self,
                 cfg,
                 net_G,
                 net_D,
                 opt_G,
                 opt_D,
                 sch_G,
                 sch_D,
                 train_data_loader,
                 val_data_loader):
        super(BaseTrainer, self).__init__()
        print('Setup trainer.')

        # Initialize models and data loaders.
        self.cfg = cfg
        self.net_G = net_G
        if cfg.trainer.model_average_config.enabled:
            # Two wrappers (DDP + model average).
            self.net_G_module = self.net_G.module.module
        else:
            # One wrapper (DDP)
            self.net_G_module = self.net_G.module
        self.val_data_loader = val_data_loader
        self.is_inference = train_data_loader is None
        self.net_D = net_D
        self.opt_G = opt_G
        self.opt_D = opt_D
        self.sch_G = sch_G
        self.sch_D = sch_D
        self.train_data_loader = train_data_loader
        if self.cfg.trainer.channels_last:
            self.net_G = self.net_G.to(memory_format=torch.channels_last)
            self.net_D = self.net_D.to(memory_format=torch.channels_last)

        # Initialize amp.
        if self.cfg.trainer.amp_config.enabled:
            print("Using automatic mixed precision training.")
        self.scaler_G = GradScaler(**vars(self.cfg.trainer.amp_config))
        self.scaler_D = GradScaler(**vars(self.cfg.trainer.amp_config))
        # In order to check whether the discriminator/generator has
        # skipped the last parameter update due to gradient overflow.
        self.last_step_count_G = 0
        self.last_step_count_D = 0
        self.skipped_G = False
        self.skipped_D = False

        # Initialize data augmentation policy.
        self.aug_policy = cfg.trainer.aug_policy
        print("Augmentation policy: {}".format(self.aug_policy))

        # Initialize loss functions.
        # All loss names have weights. Some have criterion modules.
        # Mapping from loss names to criterion modules.
        self.criteria = torch.nn.ModuleDict()
        # Mapping from loss names to loss weights.
        self.weights = dict()
        self.losses = dict(gen_update=dict(), dis_update=dict())
        self.gen_losses = self.losses['gen_update']
        self.dis_losses = self.losses['dis_update']
        self._init_loss(cfg)
        for loss_name, loss_weight in self.weights.items():
            print("Loss {:<20} Weight {}".format(loss_name, loss_weight))
            if loss_name in self.criteria.keys() and \
                    self.criteria[loss_name] is not None:
                self.criteria[loss_name].to('cuda')

        if self.is_inference:
            # The initialization steps below can be skipped during inference.
            return

        # Initialize logging attributes.
        self.current_iteration = 0
        self.current_epoch = 0
        self.start_iteration_time = None
        self.start_epoch_time = None
        self.elapsed_iteration_time = 0
        self.time_iteration = None
        self.time_epoch = None
        self.best_fid = None
        if self.cfg.speed_benchmark:
            self.accu_gen_forw_iter_time = 0
            self.accu_gen_loss_iter_time = 0
            self.accu_gen_back_iter_time = 0
            self.accu_gen_step_iter_time = 0
            self.accu_gen_avg_iter_time = 0
            self.accu_dis_forw_iter_time = 0
            self.accu_dis_loss_iter_time = 0
            self.accu_dis_back_iter_time = 0
            self.accu_dis_step_iter_time = 0

        # Initialize tensorboard and hparams.
        self._init_tensorboard()
        self._init_hparams()

        # Initialize validation parameters.
        self.val_sample_size = getattr(cfg.trainer, 'val_sample_size', 50000)
        self.kid_num_subsets = getattr(cfg.trainer, 'kid_num_subsets', 10)
        self.kid_subset_size = self.val_sample_size // self.kid_num_subsets
        self.metrics_path = os.path.join(torch.hub.get_dir(), 'metrics')
        self.best_metrics = {}
        self.eval_networks = getattr(cfg.trainer, 'eval_network', ['clean_inception'])
        if self.cfg.metrics_iter is None:
            self.cfg.metrics_iter = self.cfg.snapshot_save_iter
        if self.cfg.metrics_epoch is None:
            self.cfg.metrics_epoch = self.cfg.snapshot_save_epoch

        # AWS credentials.
        if hasattr(cfg, 'aws_credentials_file'):
            with open(cfg.aws_credentials_file) as fin:
                self.credentials = json.load(fin)
        else:
            self.credentials = None

        if 'TORCH_HOME' not in os.environ:
            os.environ['TORCH_HOME'] = os.path.join(
                os.environ['HOME'], ".cache")

    def _init_tensorboard(self):
        r"""Initialize the tensorboard. Different algorithms might require
        different performance metrics. Hence, custom tensorboard
        initialization might be necessary.
        """
        # Logging frequency: self.cfg.logging_iter
        self.meters = {}

        # Logging frequency: self.cfg.snapshot_save_iter
        self.metric_meters = {}

        # Logging frequency: self.cfg.image_display_iter
        self.image_meter = Meter('images', reduce=False)

    def _init_hparams(self):
        r"""Initialize a dictionary of hyperparameters that we want to monitor
        in the HParams dashboard in tensorBoard.
        """
        self.hparam_dict = {}

    def _write_tensorboard(self):
        r"""Write values to tensorboard. By default, we will log the time used
        per iteration, time used per epoch, generator learning rate, and
        discriminator learning rate. We will log all the losses as well as
        custom meters.
        """
        # Logs that are shared by all models.
        self._write_to_meters({'time/iteration': self.time_iteration,
                               'time/epoch': self.time_epoch,
                               'optim/gen_lr': self.sch_G.get_last_lr()[0],
                               'optim/dis_lr': self.sch_D.get_last_lr()[0]},
                              self.meters,
                              reduce=False)
        # Logs for loss values. Different models have different losses.
        self._write_loss_meters()
        # Other custom logs.
        self._write_custom_meters()

    def _write_loss_meters(self):
        r"""Write all loss values to tensorboard."""
        for update, losses in self.losses.items():
            # update is 'gen_update' or 'dis_update'.
            assert update == 'gen_update' or update == 'dis_update'
            for loss_name, loss in losses.items():
                if loss is not None:
                    full_loss_name = update + '/' + loss_name
                    if full_loss_name not in self.meters.keys():
                        # Create a new meter if it doesn't exist.
                        self.meters[full_loss_name] = Meter(
                            full_loss_name, reduce=True)
                    self.meters[full_loss_name].write(loss.item())

    def _write_custom_meters(self):
        r"""Dummy member function to be overloaded by the child class.
        In the child class, you can write down whatever you want to track.
        """
        pass

    @staticmethod
    def _write_to_meters(data, meters, reduce=True):
        r"""Write values to meters."""
        if reduce or is_master():
            for key, value in data.items():
                if key not in meters:
                    meters[key] = Meter(key, reduce=reduce)
                meters[key].write(value)

    def _flush_meters(self, meters):
        r"""Flush all meters using the current iteration."""
        for meter in meters.values():
            meter.flush(self.current_iteration)

    def _pre_save_checkpoint(self):
        r"""Implement the things you want to do before saving a checkpoint.
        For example, you can compute the K-mean features (pix2pixHD) before
        saving the model weights to a checkpoint.
        """
        pass

    def save_checkpoint(self, current_epoch, current_iteration):
        r"""Save network weights, optimizer parameters, scheduler parameters
        to a checkpoint.
        """
        self._pre_save_checkpoint()
        _save_checkpoint(self.cfg,
                         self.net_G, self.net_D,
                         self.opt_G, self.opt_D,
                         self.sch_G, self.sch_D,
                         current_epoch, current_iteration)

    def load_checkpoint(self, cfg, checkpoint_path, resume=None, load_sch=True):
        r"""Load network weights, optimizer parameters, scheduler parameters
        from a checkpoint.

        Args:
            cfg (obj): Global configuration.
            checkpoint_path (str): Path to the checkpoint.
            resume (bool or None): If not ``None``, will determine whether or
                not to load optimizers in addition to network weights.
        """
        if os.path.exists(checkpoint_path):
            # If checkpoint_path exists, we will load its weights to
            # initialize our network.
            if resume is None:
                resume = False
        elif os.path.exists(os.path.join(cfg.logdir, 'latest_checkpoint.txt')):
            # This is for resuming the training from the previously saved
            # checkpoint.
            fn = os.path.join(cfg.logdir, 'latest_checkpoint.txt')
            with open(fn, 'r') as f:
                line = f.read().splitlines()
            checkpoint_path = os.path.join(cfg.logdir, line[0].split(' ')[-1])
            if resume is None:
                resume = True
        else:
            # checkpoint not found and not specified. We will train
            # everything from scratch.
            current_epoch = 0
            current_iteration = 0
            print('No checkpoint found.')
            resume = False
            return resume, current_epoch, current_iteration
        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
        current_epoch = 0
        current_iteration = 0
        if resume:
            self.net_G.load_state_dict(checkpoint['net_G'], strict=self.cfg.trainer.strict_resume)
            if not self.is_inference:
                self.net_D.load_state_dict(checkpoint['net_D'], strict=self.cfg.trainer.strict_resume)
                if 'opt_G' in checkpoint:
                    current_epoch = checkpoint['current_epoch']
                    current_iteration = checkpoint['current_iteration']
                    self.opt_G.load_state_dict(checkpoint['opt_G'])
                    self.opt_D.load_state_dict(checkpoint['opt_D'])
                    if load_sch:
                        self.sch_G.load_state_dict(checkpoint['sch_G'])
                        self.sch_D.load_state_dict(checkpoint['sch_D'])
                    else:
                        if self.cfg.gen_opt.lr_policy.iteration_mode:
                            self.sch_G.last_epoch = current_iteration
                        else:
                            self.sch_G.last_epoch = current_epoch
                        if self.cfg.dis_opt.lr_policy.iteration_mode:
                            self.sch_D.last_epoch = current_iteration
                        else:
                            self.sch_D.last_epoch = current_epoch
                    print('Load from: {}'.format(checkpoint_path))
                else:
                    print('Load network weights only.')
        else:
            try:
                self.net_G.load_state_dict(checkpoint['net_G'], strict=self.cfg.trainer.strict_resume)
                if 'net_D' in checkpoint:
                    self.net_D.load_state_dict(checkpoint['net_D'], strict=self.cfg.trainer.strict_resume)
            except Exception:
                if self.cfg.trainer.model_average_config.enabled:
                    net_G_module = self.net_G.module.module
                else:
                    net_G_module = self.net_G.module
                if hasattr(net_G_module, 'load_pretrained_network'):
                    net_G_module.load_pretrained_network(self.net_G, checkpoint['net_G'])
                    print('Load generator weights only.')
                else:
                    raise ValueError('Checkpoint cannot be loaded.')

        print('Done with loading the checkpoint.')
        return resume, current_epoch, current_iteration

    def start_of_epoch(self, current_epoch):
        r"""Things to do before an epoch.

        Args:
            current_epoch (int): Current number of epoch.
        """
        self._start_of_epoch(current_epoch)
        self.current_epoch = current_epoch
        self.start_epoch_time = time.time()

    def start_of_iteration(self, data, current_iteration):
        r"""Things to do before an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_iteration (int): Current number of iteration.
        """
        data = self._start_of_iteration(data, current_iteration)
        data = to_cuda(data)
        if self.cfg.trainer.channels_last:
            data = to_channels_last(data)
        self.current_iteration = current_iteration
        if not self.is_inference:
            self.net_D.train()
        self.net_G.train()
        # torch.cuda.synchronize()
        self.start_iteration_time = time.time()
        return data

    def end_of_iteration(self, data, current_epoch, current_iteration):
        r"""Things to do after an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current number of iteration.
        """
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch
        # Update the learning rate policy for the generator if operating in the
        # iteration mode.
        if self.cfg.gen_opt.lr_policy.iteration_mode:
            self.sch_G.step()
        # Update the learning rate policy for the discriminator if operating in
        # the iteration mode.
        if self.cfg.dis_opt.lr_policy.iteration_mode:
            self.sch_D.step()

        # Accumulate time
        # torch.cuda.synchronize()
        self.elapsed_iteration_time += time.time() - self.start_iteration_time
        # Logging.
        if current_iteration % self.cfg.logging_iter == 0:
            ave_t = self.elapsed_iteration_time / self.cfg.logging_iter
            self.time_iteration = ave_t
            print('Iteration: {}, average iter time: '
                  '{:6f}.'.format(current_iteration, ave_t))
            self.elapsed_iteration_time = 0

            if self.cfg.speed_benchmark:
                # Below code block only needed when analyzing computation
                # bottleneck.
                print('\tGenerator FWD time {:6f}'.format(
                    self.accu_gen_forw_iter_time / self.cfg.logging_iter))
                print('\tGenerator LOS time {:6f}'.format(
                    self.accu_gen_loss_iter_time / self.cfg.logging_iter))
                print('\tGenerator BCK time {:6f}'.format(
                    self.accu_gen_back_iter_time / self.cfg.logging_iter))
                print('\tGenerator STP time {:6f}'.format(
                    self.accu_gen_step_iter_time / self.cfg.logging_iter))
                print('\tGenerator AVG time {:6f}'.format(
                    self.accu_gen_avg_iter_time / self.cfg.logging_iter))

                print('\tDiscriminator FWD time {:6f}'.format(
                    self.accu_dis_forw_iter_time / self.cfg.logging_iter))
                print('\tDiscriminator LOS time {:6f}'.format(
                    self.accu_dis_loss_iter_time / self.cfg.logging_iter))
                print('\tDiscriminator BCK time {:6f}'.format(
                    self.accu_dis_back_iter_time / self.cfg.logging_iter))
                print('\tDiscriminator STP time {:6f}'.format(
                    self.accu_dis_step_iter_time / self.cfg.logging_iter))

                print('{:6f}'.format(ave_t))

                self.accu_gen_forw_iter_time = 0
                self.accu_gen_loss_iter_time = 0
                self.accu_gen_back_iter_time = 0
                self.accu_gen_step_iter_time = 0
                self.accu_gen_avg_iter_time = 0
                self.accu_dis_forw_iter_time = 0
                self.accu_dis_loss_iter_time = 0
                self.accu_dis_back_iter_time = 0
                self.accu_dis_step_iter_time = 0

        self._end_of_iteration(data, current_epoch, current_iteration)

        # Save everything to the checkpoint.
        if current_iteration % self.cfg.snapshot_save_iter == 0:
            if current_iteration >= self.cfg.snapshot_save_start_iter:
                self.save_checkpoint(current_epoch, current_iteration)

        # Compute metrics.
        if current_iteration % self.cfg.metrics_iter == 0:
            self.save_image(self._get_save_path('images', 'jpg'), data)
            self.write_metrics()

        # Compute image to be saved.
        elif current_iteration % self.cfg.image_save_iter == 0:
            self.save_image(self._get_save_path('images', 'jpg'), data)
        elif current_iteration % self.cfg.image_display_iter == 0:
            image_path = os.path.join(self.cfg.logdir, 'images', 'current.jpg')
            self.save_image(image_path, data)

        # Logging.
        self._write_tensorboard()
        if current_iteration % self.cfg.logging_iter == 0:
            # Write all logs to tensorboard.
            self._flush_meters(self.meters)

        from torch.distributed import barrier
        import torch.distributed as dist
        if dist.is_initialized():
            barrier()

    def end_of_epoch(self, data, current_epoch, current_iteration):
        r"""Things to do after an epoch.

        Args:
            data (dict): Data used for the current iteration.

            current_epoch (int): Current number of epoch.
            current_iteration (int): Current number of iteration.
        """
        # Update the learning rate policy for the generator if operating in the
        # epoch mode.
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch
        if not self.cfg.gen_opt.lr_policy.iteration_mode:
            self.sch_G.step()
        # Update the learning rate policy for the discriminator if operating
        # in the epoch mode.
        if not self.cfg.dis_opt.lr_policy.iteration_mode:
            self.sch_D.step()
        elapsed_epoch_time = time.time() - self.start_epoch_time
        # Logging.
        print('Epoch: {}, total time: {:6f}.'.format(current_epoch,
                                                     elapsed_epoch_time))
        self.time_epoch = elapsed_epoch_time
        self._end_of_epoch(data, current_epoch, current_iteration)

        # Save everything to the checkpoint.
        if current_iteration % self.cfg.snapshot_save_iter == 0:
            if current_epoch >= self.cfg.snapshot_save_start_epoch:
                self.save_checkpoint(current_epoch, current_iteration)

        # Compute metrics.
        if current_iteration % self.cfg.metrics_iter == 0:
            self.save_image(self._get_save_path('images', 'jpg'), data)
            self.write_metrics()

    def pre_process(self, data):
        r"""Custom data pre-processing function. Utilize this function if you
        need to preprocess your data before sending it to the generator and
        discriminator.

        Args:
            data (dict): Data used for the current iteration.
        """

    def recalculate_batch_norm_statistics(self, data_loader, averaged=True):
        r"""Update the statistics in the moving average model.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for
                estimating the statistics.
            averaged (Boolean): True/False, we recalculate batch norm statistics for EMA/regular
        """
        if not self.cfg.trainer.model_average_config.enabled:
            return
        if averaged:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G_module
        model_average_iteration = \
            self.cfg.trainer.model_average_config.num_batch_norm_estimation_iterations
        if model_average_iteration == 0:
            return
        with torch.no_grad():
            # Accumulate bn stats..
            net_G.train()
            # Reset running stats.
            net_G.apply(reset_batch_norm)
            for cal_it, cal_data in enumerate(data_loader):
                if cal_it >= model_average_iteration:
                    print('Done with {} iterations of updating batch norm '
                          'statistics'.format(model_average_iteration))
                    break
                cal_data = to_device(cal_data, 'cuda')
                cal_data = self.pre_process(cal_data)
                # Averaging over all batches
                net_G.apply(calibrate_batch_norm_momentum)
                net_G(cal_data)

    def save_image(self, path, data):
        r"""Compute visualization images and save them to the disk.

        Args:
            path (str): Location of the file.
            data (dict): Data used for the current iteration.
        """
        self.net_G.eval()
        vis_images = self._get_visualizations(data)
        if is_master() and vis_images is not None:
            vis_images = torch.cat(
                [img for img in vis_images if img is not None], dim=3).float()
            vis_images = (vis_images + 1) / 2
            print('Save output images to {}'.format(path))
            vis_images.clamp_(0, 1)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image_grid = torchvision.utils.make_grid(
                vis_images, nrow=1, padding=0, normalize=False)
            if self.cfg.trainer.image_to_tensorboard:
                self.image_meter.write_image(image_grid, self.current_iteration)
            torchvision.utils.save_image(image_grid, path, nrow=1)
            wandb.log({os.path.splitext(os.path.basename(path))[0]: [wandb.Image(path)]})

    def write_metrics(self):
        r"""Write metrics to the tensorboard."""
        cur_fid = self._compute_fid()
        if cur_fid is not None:
            if self.best_fid is not None:
                self.best_fid = min(self.best_fid, cur_fid)
            else:
                self.best_fid = cur_fid
            metric_dict = {'FID': cur_fid, 'best_FID': self.best_fid}
            self._write_to_meters(metric_dict, self.metric_meters, reduce=False)
            self._flush_meters(self.metric_meters)

    def _get_save_path(self, subdir, ext):
        r"""Get the image save path.

        Args:
            subdir (str): Sub-directory under the main directory for saving
                the outputs.
            ext (str): Filename extension for the image (e.g., jpg, png, ...).
        Return:
            (str): image filename to be used to save the visualization results.
        """
        subdir_path = os.path.join(self.cfg.logdir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)
        return os.path.join(
            subdir_path, 'epoch_{:05}_iteration_{:09}.{}'.format(
                self.current_epoch, self.current_iteration, ext))

    def _get_outputs(self, net_D_output, real=True):
        r"""Return output values. Note that when the gan mode is relativistic.
        It will do the difference before returning.

        Args:
           net_D_output (dict):
               real_outputs (tensor): Real output values.
               fake_outputs (tensor): Fake output values.
           real (bool): Return real or fake.
        """

        def _get_difference(a, b):
            r"""Get difference between two lists of tensors or two tensors.

            Args:
                a: list of tensors or tensor
                b: list of tensors or tensor
            """
            out = list()
            for x, y in zip(a, b):
                if isinstance(x, list):
                    res = _get_difference(x, y)
                else:
                    res = x - y
                out.append(res)
            return out

        if real:
            if self.cfg.trainer.gan_relativistic:
                return _get_difference(net_D_output['real_outputs'], net_D_output['fake_outputs'])
            else:
                return net_D_output['real_outputs']
        else:
            if self.cfg.trainer.gan_relativistic:
                return _get_difference(net_D_output['fake_outputs'], net_D_output['real_outputs'])
            else:
                return net_D_output['fake_outputs']

    def _start_of_epoch(self, current_epoch):
        r"""Operations to do before starting an epoch.

        Args:
            current_epoch (int): Current number of epoch.
        """
        pass

    def _start_of_iteration(self, data, current_iteration):
        r"""Operations to do before starting an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_iteration (int): Current epoch number.
        Returns:
            (dict): Data used for the current iteration. They might be
                processed by the custom _start_of_iteration function.
        """
        return data

    def _end_of_iteration(self, data, current_epoch, current_iteration):
        r"""Operations to do after an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current epoch number.
        """
        pass

    def _end_of_epoch(self, data, current_epoch, current_iteration):
        r"""Operations to do after an epoch.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current epoch number.
        """
        pass

    def _get_visualizations(self, data):
        r"""Compute visualization outputs.

        Args:
            data (dict): Data used for the current iteration.
        """
        return None

    def _compute_fid(self):
        r"""FID computation function to be overloaded."""
        return None

    def _init_loss(self, cfg):
        r"""Every trainer should implement its own init loss function."""
        raise NotImplementedError

    def gen_update(self, data):
        r"""Update the generator.

        Args:
            data (dict): Data used for the current iteration.
        """
        update_finished = False
        while not update_finished:
            # Set requires_grad flags.
            requires_grad(self.net_G_module, True)
            requires_grad(self.net_D, False)

            # Compute the loss.
            self._time_before_forward()
            with autocast(enabled=self.cfg.trainer.amp_config.enabled):
                total_loss = self.gen_forward(data)
            if total_loss is None:
                return

            # Zero-grad and backpropagate the loss.
            self.opt_G.zero_grad(set_to_none=True)
            self._time_before_backward()
            self.scaler_G.scale(total_loss).backward()

            # Optionally clip gradient norm.
            if hasattr(self.cfg.gen_opt, 'clip_grad_norm'):
                self.scaler_G.unscale_(self.opt_G)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.net_G_module.parameters(),
                    self.cfg.gen_opt.clip_grad_norm
                )
                self.gen_grad_norm = total_norm
                if torch.isfinite(total_norm) and \
                        total_norm > self.cfg.gen_opt.clip_grad_norm:
                    # print(f"Gradient norm of the generator ({total_norm}) "
                    #       f"too large.")
                    if getattr(self.cfg.gen_opt, 'skip_grad', False):
                        print(f"Skip gradient update.")
                        self.opt_G.zero_grad(set_to_none=True)
                        self.scaler_G.step(self.opt_G)
                        self.scaler_G.update()
                        break
                    # else:
                    #     print(f"Clip gradient norm to "
                    #           f"{self.cfg.gen_opt.clip_grad_norm}.")

            # Perform an optimizer step.
            self._time_before_step()
            self.scaler_G.step(self.opt_G)
            self.scaler_G.update()
            # Whether the step above was skipped.
            if self.last_step_count_G == self.opt_G._step_count:
                print("Generator overflowed!")
                if not torch.isfinite(total_loss):
                    print("Generator loss is not finite. Skip this iteration!")
                    update_finished = True
            else:
                self.last_step_count_G = self.opt_G._step_count
                update_finished = True

        self._extra_gen_step(data)

        # Update model average.
        self._time_before_model_avg()
        if self.cfg.trainer.model_average_config.enabled:
            self.net_G.module.update_average()

        self._detach_losses()
        self._time_before_leave_gen()

    def gen_forward(self, data):
        r"""Every trainer should implement its own generator forward."""
        raise NotImplementedError

    def _extra_gen_step(self, data):
        pass

    def dis_update(self, data):
        r"""Update the discriminator.

        Args:
            data (dict): Data used for the current iteration.
        """
        update_finished = False
        while not update_finished:
            # Set requires_grad flags.
            requires_grad(self.net_G_module, False)
            requires_grad(self.net_D, True)

            # Compute the loss.
            self._time_before_forward()
            with autocast(enabled=self.cfg.trainer.amp_config.enabled):
                total_loss = self.dis_forward(data)
            if total_loss is None:
                return

            # Zero-grad and backpropagate the loss.
            self.opt_D.zero_grad(set_to_none=True)
            self._time_before_backward()
            self.scaler_D.scale(total_loss).backward()

            # Optionally clip gradient norm.
            if hasattr(self.cfg.dis_opt, 'clip_grad_norm'):
                self.scaler_D.unscale_(self.opt_D)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.net_D.parameters(), self.cfg.dis_opt.clip_grad_norm
                )
                self.dis_grad_norm = total_norm
                if torch.isfinite(total_norm) and \
                        total_norm > self.cfg.dis_opt.clip_grad_norm:
                    print(f"Gradient norm of the discriminator ({total_norm}) "
                          f"too large.")
                    if getattr(self.cfg.dis_opt, 'skip_grad', False):
                        print(f"Skip gradient update.")
                        self.opt_D.zero_grad(set_to_none=True)
                        self.scaler_D.step(self.opt_D)
                        self.scaler_D.update()
                        continue
                    else:
                        print(f"Clip gradient norm to "
                              f"{self.cfg.dis_opt.clip_grad_norm}.")

            # Perform an optimizer step.
            self._time_before_step()
            self.scaler_D.step(self.opt_D)
            self.scaler_D.update()
            # Whether the step above was skipped.
            if self.last_step_count_D == self.opt_D._step_count:
                print("Discriminator overflowed!")
                if not torch.isfinite(total_loss):
                    print("Discriminator loss is not finite. "
                          "Skip this iteration!")
                    update_finished = True
            else:
                self.last_step_count_D = self.opt_D._step_count
                update_finished = True

        self._extra_dis_step(data)

        self._detach_losses()
        self._time_before_leave_dis()

    def dis_forward(self, data):
        r"""Every trainer should implement its own discriminator forward."""
        raise NotImplementedError

    def _extra_dis_step(self, data):
        pass

    def test(self, data_loader, output_dir, inference_args):
        r"""Compute results images for a batch of input data and save the
        results in the specified folder.

        Args:
            data_loader (torch.utils.data.DataLoader): PyTorch dataloader.
            output_dir (str): Target location for saving the output image.
        """
        if self.cfg.trainer.model_average_config.enabled:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()

        print('# of samples %d' % len(data_loader))
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration=-1)
            with torch.no_grad():
                output_images, file_names = \
                    net_G.inference(data, **vars(inference_args))
            for output_image, file_name in zip(output_images, file_names):
                fullname = os.path.join(output_dir, file_name + '.jpg')
                output_image = tensor2pilimage(output_image.clamp_(-1, 1),
                                               minus1to1_normalized=True)
                save_pilimage_in_jpeg(fullname, output_image)


    def test_cocofunit(self, data_loader, output_dir, cocofunit_option, tsne_one_image_id, inference_root, inference_args):
        if self.cfg.trainer.model_average_config.enabled:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()

        if cocofunit_option == 0:
            print('# of samples %d' % len(data_loader))
            for it, data in enumerate(tqdm(data_loader)):
                data = self.start_of_iteration(data, current_iteration=-1)
                with torch.no_grad():
                    output_images, file_names = \
                        net_G.inference_test(data, **vars(inference_args))
                for output_image, file_name in zip(output_images, file_names):
                    fullname = os.path.join(output_dir, file_name + '.jpg')
                    output_image = tensor2pilimage(output_image.clamp_(-1, 1),
                                                   minus1to1_normalized=True)
                    save_pilimage_in_jpeg(fullname, output_image)
        elif cocofunit_option == 1:
            self.test_tsne_cocofunit(data_loader, output_dir, inference_args)
        elif cocofunit_option == 2:
            self.test_tsne_cocofunit_squeezed(data_loader, output_dir, inference_args)
        elif cocofunit_option == 3:
            self.test_tsne_one_image_cocofunit(data_loader, output_dir, tsne_one_image_id, inference_root, inference_args)

    def test_tsne_cocofunit(self, data_loader, output_dir, inference_args):
        if self.cfg.trainer.model_average_config.enabled:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()

        debugging = True
        print('# of samples for getting content and style code: %d' % len(data_loader))

        with torch.no_grad():
            content_dict = {}
            content_list = []
            content_fname_list = []
            style_dict = {}
            style_list = []
            style_fname_list = []
            class_list = []
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration=-1)
            with torch.no_grad():
                # style_tensors, style_filenames, style_dirname = net_G.get_style_code(data, **vars(inference_args))
                content, content_filenames, content_dirname, style, style_filenames, style_dirname = net_G.get_content_and_style_code(
                    data, **vars(inference_args))

                for cont, fn in zip(content, content_filenames):
                    if fn not in content_dict:
                        content_dict[fn] = cont
                        content_list.append(cont)
                        content_fname_list.append(fn)

                for st, fn in zip(style, style_filenames):
                    if fn not in style_dict:
                        style_dict[fn] = st
                        style_list.append(st)
                        style_fname_list.append(fn)
                        class_list.append(fn.split('/')[0])

        contents = torch.cat([x.unsqueeze(0) for x in content_list], 0)
        styles = torch.cat([x.unsqueeze(0) for x in style_list], 0)

        if debugging:
            print(f'contents.size(): {contents.size()}')
            print(f'len(content_list): {len(content_list)}')
            print(f'len(content_dict): {len(content_dict)}')
            print(f'styles.size(): {styles.size()}')
            print(f'len(style_list): {len(style_list)}')
            print(f'len(style_dict): {len(style_dict)}')

            print(f'last content.size(): {content.size()}')
            print(f'last style.size(): {style.size()}')

        import numpy as np
        # content_data = {}
        style_data = {}

        # content_data['data'] = contents.detach().cpu().squeeze().numpy()
        style_data['data'] = styles.detach().cpu().squeeze().numpy()

        # content_data['filename'] = np.asarray(content_fname_list)
        style_data['filename'] = np.asarray(style_fname_list)

        style_data['class'] = np.asarray(class_list)

        # content_data['dirname'] = content_dirname
        style_data['dirname'] = style_dirname

        # content_data['a2b'] = dict_inference_args["a2b"]
        # style_data['a2b'] = dict_inference_args["a2b"]

        # contents_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_contents.pkl')
        styles_pkl = os.path.join(output_dir, f'styles.pkl')

        if debugging:
            # print(f'content_data["data"].shape: {content_data["data"].shape}')
            # print(f'content_data["dirname"]: {content_data["dirname"]}')
            # print(f'content_data["filename"].shape: {content_data["filename"].shape}')
            print(f'style_data: {style_data}')
            print(f'style_data["data"].shape: {style_data["data"].shape}')
            print(f'style_data["dirname"]: {style_data["dirname"]}')
            print(f'style_data["filename"].shape: {style_data["filename"].shape}')
            print(f'content_data["class"]: {style_data["class"]}')

        print('Saving style codes to {}'.format(styles_pkl))

        import pickle
        '''
        with open(contents_pkl, 'wb') as f:
            pickle.dump(content_data, f)
        '''
        with open(styles_pkl, 'wb') as f:
            pickle.dump(style_data, f)
        '''
        with open(contents_pkl, 'rb') as f:
            content_data = pickle.load(f)
        with open(styles_pkl, 'rb') as f:
            style_data = pickle.load(f)

        path = [os.path.join(content_data['dirname'], fn + '.jpg') for fn in content_data['filename']]
        '''

    def test_tsne_cocofunit_squeezed(self, data_loader, output_dir, inference_args):
        if self.cfg.trainer.model_average_config.enabled:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()

        debugging = True
        print('# of samples for getting content and style code: %d' % len(data_loader))

        with torch.no_grad():
            content_dict = {}
            content_list = []
            content_fname_list = []
            style_dict = {}
            style_list = []
            style_fname_list = []
            class_list = []
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration=-1)
            with torch.no_grad():
                # style_tensors, style_filenames, style_dirname = net_G.get_style_code(data, **vars(inference_args))
                content, content_filenames, content_dirname, style, style_filenames, style_dirname = net_G.get_content_and_style_code_squeezed(
                    data, **vars(inference_args))

                for cont, fn in zip(content, content_filenames):
                    if fn not in content_dict:
                        content_dict[fn] = cont
                        content_list.append(cont)
                        content_fname_list.append(fn)

                for st, fn in zip(style, style_filenames):
                    if fn not in style_dict:
                        style_dict[fn] = st
                        style_list.append(st)
                        style_fname_list.append(fn)
                        class_list.append(fn.split('/')[0])

        contents = torch.cat([x.unsqueeze(0) for x in content_list], 0)
        styles = torch.cat([x.unsqueeze(0) for x in style_list], 0)

        if debugging:
            print(f'contents.size(): {contents.size()}')
            print(f'len(content_list): {len(content_list)}')
            print(f'len(content_dict): {len(content_dict)}')
            print(f'styles.size(): {styles.size()}')
            print(f'len(style_list): {len(style_list)}')
            print(f'len(style_dict): {len(style_dict)}')

            print(f'last content.size(): {content.size()}')
            print(f'last style.size(): {style.size()}')

        import numpy as np
        # content_data = {}
        style_data = {}

        # content_data['data'] = contents.detach().cpu().squeeze().numpy()
        style_data['data'] = styles.detach().cpu().squeeze().numpy()

        # content_data['filename'] = np.asarray(content_fname_list)
        style_data['filename'] = np.asarray(style_fname_list)

        style_data['class'] = np.asarray(class_list)

        # content_data['dirname'] = content_dirname
        style_data['dirname'] = style_dirname

        # content_data['a2b'] = dict_inference_args["a2b"]
        # style_data['a2b'] = dict_inference_args["a2b"]

        # contents_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_contents.pkl')
        styles_pkl = os.path.join(output_dir, f'styles.pkl')

        if debugging:
            # print(f'content_data["data"].shape: {content_data["data"].shape}')
            # print(f'content_data["dirname"]: {content_data["dirname"]}')
            # print(f'content_data["filename"].shape: {content_data["filename"].shape}')
            print(f'style_data: {style_data}')
            print(f'style_data["data"].shape: {style_data["data"].shape}')
            print(f'style_data["dirname"]: {style_data["dirname"]}')
            print(f'style_data["filename"].shape: {style_data["filename"].shape}')
            print(f'content_data["class"]: {style_data["class"]}')

        print('Saving style codes to {}'.format(styles_pkl))

        import pickle
        '''
        with open(contents_pkl, 'wb') as f:
            pickle.dump(content_data, f)
        '''
        with open(styles_pkl, 'wb') as f:
            pickle.dump(style_data, f)
        '''
        with open(contents_pkl, 'rb') as f:
            content_data = pickle.load(f)
        with open(styles_pkl, 'rb') as f:
            style_data = pickle.load(f)

        path = [os.path.join(content_data['dirname'], fn + '.jpg') for fn in content_data['filename']]
        '''

    def test_tsne_one_image_cocofunit(self, data_loader, output_dir, tsne_one_image_id, inference_root, inference_args):
        if self.cfg.trainer.model_average_config.enabled:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()

        debugging = False  # True
        print('# of samples for getting content and style code: %d' % len(data_loader))

        with torch.no_grad():
            content_dict = {}
            content_list = []
            content_fname_list = []
            style_dict = {}
            style_list = []
            style_fname_list = []
            class_list = []
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration=-1)
            with torch.no_grad():
                # style_tensors, style_filenames, style_dirname = net_G.get_style_code(data, **vars(inference_args))
                content, content_filenames, content_dirname, style, style_filenames, style_dirname = net_G.get_content_and_style_code(
                    data, **vars(inference_args))

                for cont, fn in zip(content, content_filenames):
                    if fn not in content_dict:
                        content_dict[fn] = cont
                        content_list.append(cont)
                        content_fname_list.append(fn)

                for st, fn in zip(style, style_filenames):
                    if fn not in style_dict:
                        style_dict[fn] = st
                        style_list.append(st)
                        style_fname_list.append(fn)
                        class_list.append(fn.split('/')[0])

        # contents = torch.cat([x.unsqueeze(0) for x in content_list], 0)
        styles = torch.cat([x.unsqueeze(0) for x in style_list], 0)

        for cls_dir in set(class_list):
            newpath = os.path.join(output_dir, cls_dir)
            if not os.path.exists(newpath):
                print(f'making {newpath}')
                os.makedirs(newpath, exist_ok=True)

        import numpy as np
        import shutil
        # content = contents[tsne_one_image_id].unsqueeze(0)
        content = content_list[tsne_one_image_id].unsqueeze(0)
        print(f'The one image to translate is {content_fname_list[tsne_one_image_id]}')
        # inference_root = '/data/scratch/projects/punim1358/HZ_GANs/imaginaire/Experiments/coco_animal/inference/'
        if inference_root == '':
            inference_root = os.path.dirname(output_dir)
        content_image_src = os.path.join(inference_root, content_dirname, f'{content_fname_list[tsne_one_image_id]}.jpg')
        content_image_copy = os.path.join(output_dir,
                                          f'content_image_{tsne_one_image_id}.jpg')
        print(f'Make a copy of content image from {content_image_src} to \n {content_image_copy}')
        shutil.copyfile(content_image_src, content_image_copy)

        print(f'Translating one image id {tsne_one_image_id} for all styles...')
        for file_names, style in tqdm(style_dict.items()):  # zip(styles, style_fname_list):
            style = style.unsqueeze(0)
            # for style, file_names in zip(styles, style_fname_list):
            with torch.no_grad():
                output_images = net_G.inference_tensor(content, style)
                file_names = np.atleast_1d(file_names)
            for output_image, file_name in zip(output_images, file_names):
                fullname = os.path.join(output_dir, file_name + '.jpg')
                if debugging:
                    print(fullname)
                output_image = tensor2pilimage(output_image.clamp_(-1, 1),
                                               minus1to1_normalized=True)
                save_pilimage_in_jpeg(fullname, output_image)

        if debugging:
            # print(f'contents.size(): {contents.size()}')
            print(f'len(content_list): {len(content_list)}')
            print(f'len(content_dict): {len(content_dict)}')
            print(f'styles.size(): {styles.size()}')
            print(f'len(style_list): {len(style_list)}')
            print(f'len(style_dict): {len(style_dict)}')

            print(f'last content.size(): {content.size()}')
            print(f'last style.size(): {style.size()}')

            # import numpy as np
        content_data = {}
        style_data = {}

        # content_data['data'] = contents.detach().cpu().squeeze().numpy()
        style_data['data'] = styles.detach().cpu().squeeze().numpy()

        # content_data['filename'] = np.asarray(content_fname_list)
        style_data['filename'] = np.asarray(style_fname_list)

        style_data['class'] = np.asarray(class_list)

        # content_data['dirname'] = content_dirname
        style_data['dirname'] = style_dirname

        # content_data['a2b'] = dict_inference_args["a2b"]
        # style_data['a2b'] = dict_inference_args["a2b"]

        # contents_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_contents.pkl')
        styles_pkl = os.path.join(output_dir, f'styles.pkl')

        if debugging:
            print(f'content_data["data"].shape: {content_data["data"].shape}')
            print(f'content_data["dirname"]: {content_data["dirname"]}')
            print(f'content_data["filename"].shape: {content_data["filename"].shape}')
            print(f'style_data["data"].shape: {style_data["data"].shape}')
            print(f'style_data["dirname"]: {style_data["dirname"]}')
            print(f'style_data["filename"].shape: {style_data["filename"].shape}')

        # print('Saving content and style codes to {} and\n {}'.format(contents_pkl, styles_pkl))
        print(f'Saving style codes to {styles_pkl}')

        import pickle
        '''
        with open(contents_pkl, 'wb') as f:
            pickle.dump(content_data, f)
        '''
        with open(styles_pkl, 'wb') as f:
            pickle.dump(style_data, f)
        '''
        with open(contents_pkl, 'rb') as f:
            content_data = pickle.load(f)
        with open(styles_pkl, 'rb') as f:
            style_data = pickle.load(f)

        path = [os.path.join(content_data['dirname'], fn + '.jpg') for fn in content_data['filename']]
        '''


    def test_style(self, data_loader, output_dir, munit_style, save_style_codes_only, all_styles, simple_grid, grid_styles, inference_args):
        r"""Compute results images for a batch of input data and save the
        results in the specified folder.

        Args:
            data_loader (torch.utils.data.DataLoader): PyTorch dataloader.
            output_dir (str): Target location for saving the output image.
        """
        if self.cfg.trainer.model_average_config.enabled:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()
        
        
        #print(f"inference_args: {inference_args}\n type(inference_args): {type(inference_args)}\n dict(inference_args): {dict(inference_args)}")
        dict_inference_args = dict(inference_args)
        print(f"dict_inference_args: {dict_inference_args}") #if dict_inference_args['a2b']:

        debugging = False #True
        print('# of samples for getting style code %d' % len(data_loader))
        style_dict = {}
        style_list = []
        name_list = []
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration=-1)
            with torch.no_grad():
                style_tensors, style_filenames, style_dirname = net_G.get_style_code(data, **vars(inference_args))                
                for st, fn in zip(style_tensors, style_filenames):
                    if fn not in style_dict:
                        style_dict[fn] = st                        
                        style_list.append(st)
                        name_list.append(fn)                        
                        if debugging:
                            print(f'st: {st}')
                            print(f'fn: {fn}')
                            #print(f"data: {data}")
                    
        styles = torch.cat([x.unsqueeze(0) for x in style_list], 0)
        style_mean = styles.mean(dim=0, keepdim=True) #or std_mean
        if debugging:
            print(f'styles: {styles}')
            print(f'styles.size(): {styles.size()}')
            print(f'style_mean: {style_mean}')
            print(f'style_mean.size(): {style_mean.size()}')
            print(f'len(style_list): {len(style_list)}')
            print(f'len(name_list):  {len(name_list)}')
            print(f'len(style_dict): {len(style_dict)}')
        
        max_dist = -1000000
        min_dist = 1000000      
        for file_name, style in style_dict.items(): #zip(style_list, name_list):
            style = style.unsqueeze(0)
            dist = (style - style_mean).pow(2).sum(1).sqrt()
            if debugging:
                print(f'file_name: {file_name}\n style.size(): {style.size()}')
            #euclidena_dist = sum(((a - b)**2).reshape(8))
            if min_dist > dist:
                min_dist = dist
                min_style = style
                min_filename = file_name           
            
            if max_dist < dist:
                max_dist = dist
                max_style = style
                max_filename = file_name
                
        min_filename = style_dirname + '/' + min_filename + '.jpg'
        max_filename = style_dirname + '/' + max_filename + '.jpg'
        if debugging:
            print(f'style_dirname: {style_dirname}')
            print(f'min_dist: {min_dist}')
            print(f'min_filename: {min_filename}')
            print(f'min_style: {min_style}')
            print(f'max_dist: {max_dist}')
            print(f'max_filename: {max_filename}')
            print(f'max_style: {max_style}')
        
        style_tensor = style_mean
        if munit_style == 'max':
            style_tensor = max_style        
        elif munit_style == 'min':
            style_tensor = min_style
        elif munit_style == 'mean':
            style_tensor = style_mean
        elif munit_style == 'random':
            style_tensor = 'random'
        else:
            print("Wrong choice!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!, please check your option for munit style!!")            
        
        from datetime import datetime
        import shutil
        if all_styles or save_style_codes_only:
            txt_log = os.path.join(output_dir, 'min_max_dist_images.txt')
            min_copy = os.path.join(output_dir, f'min_style_a2b_{dict_inference_args["a2b"]}.jpg')
            max_copy = os.path.join(output_dir, f'max_style_a2b_{dict_inference_args["a2b"]}.jpg')
            styles_log = os.path.join(output_dir, f'styles_a2b_{dict_inference_args["a2b"]}')
        else:
            txt_log = os.path.join(output_dir, '../min_max_dist_images.txt')
            min_copy = os.path.join(output_dir, f'../min_style_a2b_{dict_inference_args["a2b"]}.jpg')
            max_copy = os.path.join(output_dir, f'../max_style_a2b_{dict_inference_args["a2b"]}.jpg')
            styles_log = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}')
        with open(txt_log,"a") as f:
            f.write(f'{datetime.now().strftime("%d-%b-%Y, %H:%M:%S.%f")}\n')
            f.write(f'inference_args: {inference_args}\n\n')
            
            f.write(f'len(style_dict): {len(style_dict)}\n')
            f.write(f'len(style_list): {len(style_list)}\n')
            f.write(f'len(name_list):  {len(name_list)}\n\n') 
            
            f.write(f'style_mean: {style_mean}\n')
            f.write(f'style_mean.size(): {style_mean.size()}\n\n')

            f.write(f'min_dist: {min_dist}\n')
            f.write(f'min_filename: {min_filename}\n')
            f.write(f'min_style: {min_style}\n\n')
            
            f.write(f'max_dist: {max_dist}\n')
            f.write(f'max_filename: {max_filename}\n')
            f.write(f'max_style: {max_style}\n\n')

            shutil.copyfile(min_filename, min_copy)
            shutil.copyfile(max_filename, max_copy)
            f.write(f'save a copy of min style image from {min_filename} to {min_copy}\n')
            f.write(f'save a copy of max style image from {max_filename} to {max_copy}\n\n')
            print(f'save a copy of min style image from {min_filename} to {min_copy}\n')
            print(f'save a copy of max style image from {max_filename} to {max_copy}\n\n')
            
            f.write(f'munit_style: {munit_style}\n')
            f.write(f'style_tensor: {style_tensor}\n\n\n')            
        
        
        if simple_grid:
            lft, _ = styles.min(dim=0, keepdim=True)
            rght, _ = styles.max(dim=0, keepdim=True)
            if debugging:
                print(f'lft: {lft}\n rght: {rght}')
                print(f'style_mean: {style_mean}')
                print(f'torch.add(style_mean, lft) : {torch.add(style_mean, lft)}')
                print(f'torch.add(style_mean, lft)/2 : {torch.add(style_mean, lft)/2}')
            half_lft = torch.add(lft, torch.sub(style_mean, lft)/2)
            half_rght = torch.sub(rght, torch.sub(rght, style_mean)/2)
            grid_style_tensors = [lft, half_lft, style_mean, half_rght, rght]
            
            print('# of samples %d' % len(data_loader))
            for it, data in enumerate(tqdm(data_loader)):
                data = self.start_of_iteration(data, current_iteration=-1)
                vis_images = []
                if dict_inference_args["a2b"]:
                    vis_images.append(data['images_a'])
                else:
                    vis_images.append(data['images_b'])                          
                
                for style_tensor in grid_style_tensors:                
                    with torch.no_grad():
                        output_images, file_names = net_G.inference_style(data, style_tensor, **vars(inference_args))
                        vis_images.append(output_images)
                        
                path = os.path.join(output_dir, file_names[0] + '.jpg')
                if vis_images is not None:
                    vis_images = torch.cat([img for img in vis_images if img is not None], dim=3).float()
                    vis_images = (vis_images + 1) / 2
                    #print('Save output images to {}'.format(path))
                    vis_images.clamp_(0, 1)
                    #os.makedirs(os.path.dirname(path), exist_ok=True)
                    image_grid = torchvision.utils.make_grid(vis_images, nrow=1, padding=2, normalize=False)        
                    torchvision.utils.save_image(image_grid, path, nrow=1)
        elif grid_styles:
            lft, _ = styles.min(dim=0, keepdim=True)
            rght, _ = styles.max(dim=0, keepdim=True)
            if debugging:
                print(f'lft: {lft}\n rght: {rght}')
                print(f'style_mean: {style_mean}')
                print(f'style_mean.size(): {style_mean.size()}')
                print(f'style_mean.size()[1]: {style_mean.size()[1]}')
            half_lft = torch.add(lft, torch.sub(style_mean, lft)/2)
            half_rght = torch.sub(rght, torch.sub(rght, style_mean)/2)
            grid_style_tensors = [lft, half_lft, style_mean, half_rght, rght]
            full_grid_tensors = []
            
            for step in grid_style_tensors:
                for dim in range(style_mean.size()[1]):
                    tmp = style_mean.detach().clone()
                    tmp[0, dim, 0, 0] = step[0, dim, 0, 0]
                    #print(f'tmp[0, dim, 0, 0]:\n {tmp[0, dim, 0, 0]}\n')
                    #print(f'tmp:\n {tmp} \n style_mean:\n {style_mean}\n')
                    full_grid_tensors.append(tmp)
            
            print('# of samples %d' % len(data_loader))
            for it, data in enumerate(tqdm(data_loader)):
                data = self.start_of_iteration(data, current_iteration=-1)
                vis_images = []
                
                for style_tensor in full_grid_tensors:                
                    with torch.no_grad():
                        output_images, file_names = net_G.inference_style(data, style_tensor, **vars(inference_args))
                        vis_images.append(output_images)
                        #print(f'output_images.size(): {output_images.size()}')
                        
                if dict_inference_args["a2b"]:
                    vis_images.append(data['images_a'])
                else:
                    vis_images.append(data['images_b'])
                
                if len(file_names) == 1:
                    path = os.path.join(output_dir, file_names[0] + '.jpg')
                    vis_images = torch.cat([img for img in vis_images if img is not None], dim=0).float()#dim=3).float()
                    vis_images = (vis_images + 1) / 2
                    vis_images.clamp_(0, 1)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    #print(f'vis_images.size(): {vis_images.size()}')
                    image_grid = torchvision.utils.make_grid(vis_images, nrow=8, padding=2, normalize=False)        
                    #torchvision.utils.save_image(image_grid, path, nrow=8)
                    torchvision.transforms.ToPILImage()(image_grid).save(path)
                elif len(file_names) > 1: # for batch > 1
                    for i in range(len(file_names)):
                        path = os.path.join(output_dir, file_names[i] + '.jpg')
                        #print(f'pos1: vis_images[0].size(): {vis_images[0].size()}')
                        #img = vis_images[0]
                        #print(f'img[i].size(): {img[i].size()}')
                        #print(f'img[i].unsqueeze(0).size(): {img[i].unsqueeze(0).size()}')
                        #print(f'img[i, :, :, :].size(): {img[i, :, :, :].size()}')
                        #print(f'img[i, :, :, :].unsqueeze(0).size(): {img[i, :, :, :].unsqueeze(0).size()}')
                        vis_img = torch.cat([img[i].unsqueeze(0) for img in vis_images if img is not None], dim=0).float()#dim=3).float()
                        vis_img = (vis_img + 1) / 2
                        vis_img.clamp_(0, 1)
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        #print(f'pos2: vis_img.size(): {vis_img.size()}')
                        image_grid = torchvision.utils.make_grid(vis_img, nrow=8, padding=2, normalize=False)        
                        #torchvision.utils.save_image(image_grid, path, nrow=8)
                        torchvision.transforms.ToPILImage()(image_grid).save(path)
                else:
                    print("No output images available!")
                    
        elif all_styles:
            all_style_tensors = [style_mean, min_style, max_style, 'random']
            all_subfolders = ['mean', 'min', 'max', 'random']
            for style_tensor, subfolder in zip(all_style_tensors, all_subfolders):
                print('# of samples %d' % len(data_loader))
                for it, data in enumerate(tqdm(data_loader)):
                    data = self.start_of_iteration(data, current_iteration=-1)
                    with torch.no_grad():
                        output_images, file_names = \
                            net_G.inference_style(data, style_tensor, **vars(inference_args))
                    for output_image, file_name in zip(output_images, file_names):
                        fullname = os.path.join(output_dir, subfolder, file_name + '.jpg')
                        output_image = tensor2pilimage(output_image.clamp_(-1, 1),
                                                       minus1to1_normalized=True)
                        save_pilimage_in_jpeg(fullname, output_image)
                        
        elif save_style_codes_only:
            styles_numpy = styles.cpu().detach().clone().numpy()
            print(f'styles_numpy.shape: {styles_numpy.shape}')
            import numpy as np
            np.save(styles_log, styles_numpy)
            print('Saving style codes numpy to {}'.format(styles_log))
            return
        else:            
            print('# of samples %d' % len(data_loader))
            for it, data in enumerate(tqdm(data_loader)):
                data = self.start_of_iteration(data, current_iteration=-1)
                with torch.no_grad():
                    output_images, file_names = \
                        net_G.inference_style(data, style_tensor, **vars(inference_args))
                for output_image, file_name in zip(output_images, file_names):
                    fullname = os.path.join(output_dir, file_name + '.jpg')
                    output_image = tensor2pilimage(output_image.clamp_(-1, 1),
                                                   minus1to1_normalized=True)
                    save_pilimage_in_jpeg(fullname, output_image)
                    
        styles_numpy = styles.cpu().detach().clone().numpy()
        print(f'styles_numpy.shape: {styles_numpy.shape}')
        import numpy as np
        np.save(styles_log, styles_numpy)
        print('Saving style codes numpy to {}'.format(styles_log))
        

    def test_tsne(self, data_loader, output_dir, inference_args):
        if self.cfg.trainer.model_average_config.enabled:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()
        
        dict_inference_args = dict(inference_args)
        print(f"dict_inference_args: {dict_inference_args}")
        debugging = True
        print('# of samples for getting content and style code: %d' % len(data_loader))
        
        with torch.no_grad():
            content_dict = {}
            content_list = []
            content_fname_list = []        
            style_dict = {}
            style_list = []
            style_fname_list = []
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration=-1)
            with torch.no_grad():
                #style_tensors, style_filenames, style_dirname = net_G.get_style_code(data, **vars(inference_args))
                content, content_filenames, content_dirname, style, style_filenames, style_dirname = net_G.get_content_and_style_code(data, **vars(inference_args))
                
                for cont, fn in zip(content, content_filenames):
                    if fn not in content_dict:
                        content_dict[fn] = cont                        
                        content_list.append(cont)
                        content_fname_list.append(fn)
                        
                for st, fn in zip(style, style_filenames):
                    if fn not in style_dict:
                        style_dict[fn] = st                        
                        style_list.append(st)
                        style_fname_list.append(fn)
                        
        #contents = torch.cat([x.unsqueeze(0) for x in content_list], 0)
        styles   = torch.cat([x.unsqueeze(0) for x in style_list], 0)
        
        if debugging:
            print(f'contents.size(): {contents.size()}')            
            print(f'len(content_list): {len(content_list)}')            
            print(f'len(content_dict): {len(content_dict)}')        
            print(f'styles.size(): {styles.size()}')           
            print(f'len(style_list): {len(style_list)}')            
            print(f'len(style_dict): {len(style_dict)}')
            
            print(f'last content.size(): {content.size()}')    
            print(f'last style.size(): {style.size()}')   
            
        import numpy as np
        #content_data = {}
        style_data   = {}
        
        #content_data['data'] = contents.detach().cpu().squeeze().numpy()        
        style_data['data']   = styles.detach().cpu().squeeze().numpy()
        
        #content_data['filename'] = np.asarray(content_fname_list)
        style_data['filename']   = np.asarray(style_fname_list)
        
        #content_data['dirname'] = content_dirname
        style_data['dirname']   = style_dirname
        
        #content_data['a2b'] = dict_inference_args["a2b"]
        style_data['a2b']   = dict_inference_args["a2b"]
        
        #contents_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_contents.pkl')
        styles_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_styles.pkl')
        
        if debugging:
            print(f'content_data["data"].shape: {content_data["data"].shape}')
            print(f'content_data["dirname"]: {content_data["dirname"]}')
            print(f'content_data["filename"].shape: {content_data["filename"].shape}')
            print(f'style_data["data"].shape: {style_data["data"].shape}')
            print(f'style_data["dirname"]: {style_data["dirname"]}')
            print(f'style_data["filename"].shape: {style_data["filename"].shape}')
            
        print('Saving content and style codes to {} and\n {}'.format(contents_pkl, styles_pkl))
        
        import pickle
        '''
        with open(contents_pkl, 'wb') as f:
            pickle.dump(content_data, f)
        '''
        with open(styles_pkl, 'wb') as f:
            pickle.dump(style_data, f)
        '''
        with open(contents_pkl, 'rb') as f:
            content_data = pickle.load(f)
        with open(styles_pkl, 'rb') as f:
            style_data = pickle.load(f)
            
        path = [os.path.join(content_data['dirname'], fn + '.jpg') for fn in content_data['filename']]
        '''
        
    def test_tsne_bak(self, data_loader, output_dir, inference_args):
        if self.cfg.trainer.model_average_config.enabled:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()
        
        dict_inference_args = dict(inference_args)
        print(f"dict_inference_args: {dict_inference_args}")
        debugging = True
        print('# of samples for getting content and style code: %d' % len(data_loader))
        
        content_dict = {}
        content_list = []
        content_fname_list = []        
        style_dict = {}
        style_list = []
        style_fname_list = []
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration=-1)
            with torch.no_grad():
                #style_tensors, style_filenames, style_dirname = net_G.get_style_code(data, **vars(inference_args))
                content, content_filenames, content_dirname, style, style_filenames, style_dirname = net_G.get_content_and_style_code(data, **vars(inference_args))
                
                for cont, fn in zip(content, content_filenames):
                    if fn not in content_dict:
                        content_dict[fn] = cont                        
                        content_list.append(cont)
                        content_fname_list.append(fn)
                        
                for st, fn in zip(style, style_filenames):
                    if fn not in style_dict:
                        style_dict[fn] = st                        
                        style_list.append(st)
                        style_fname_list.append(fn)
                        
        contents = torch.cat([x.unsqueeze(0) for x in content_list], 0)
        styles   = torch.cat([x.unsqueeze(0) for x in style_list], 0)
        
        if debugging:
            print(f'contents.size(): {contents.size()}')            
            print(f'len(content_list): {len(content_list)}')            
            print(f'len(content_dict): {len(content_dict)}')        
            print(f'styles.size(): {styles.size()}')           
            print(f'len(style_list): {len(style_list)}')            
            print(f'len(style_dict): {len(style_dict)}')
            
            print(f'last content.size(): {content.size()}')    
            print(f'last style.size(): {style.size()}')   
            
        import numpy as np
        content_data = {}
        style_data   = {}
        
        content_data['data'] = contents.detach().cpu().squeeze().numpy()        
        style_data['data']   = styles.detach().cpu().squeeze().numpy()
        
        content_data['filename'] = np.asarray(content_fname_list)
        style_data['filename']   = np.asarray(style_fname_list)
        
        content_data['dirname'] = content_dirname
        style_data['dirname']   = style_dirname
        
        content_data['a2b'] = dict_inference_args["a2b"]
        style_data['a2b']   = dict_inference_args["a2b"]
        
        contents_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_contents.pkl')
        styles_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_styles.pkl')
        
        if debugging:
            print(f'content_data["data"].shape: {content_data["data"].shape}')
            print(f'content_data["dirname"]: {content_data["dirname"]}')
            print(f'content_data["filename"].shape: {content_data["filename"].shape}')
            print(f'style_data["data"].shape: {style_data["data"].shape}')
            print(f'style_data["dirname"]: {style_data["dirname"]}')
            print(f'style_data["filename"].shape: {style_data["filename"].shape}')
            
        print('Saving content and style codes to {} and\n {}'.format(contents_pkl, styles_pkl))
        
        import pickle
        with open(contents_pkl, 'wb') as f:
            pickle.dump(content_data, f)
        with open(styles_pkl, 'wb') as f:
            pickle.dump(style_data, f)
        '''
        with open(contents_pkl, 'rb') as f:
            content_data = pickle.load(f)
        with open(styles_pkl, 'rb') as f:
            style_data = pickle.load(f)
            
        path = [os.path.join(content_data['dirname'], fn + '.jpg') for fn in content_data['filename']]
        '''
        
        
    def test_tsne_one_image(self, data_loader, output_dir, tsne_one_image_id, inference_args):
        if self.cfg.trainer.model_average_config.enabled:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()
        
        dict_inference_args = dict(inference_args)
        print(f"dict_inference_args: {dict_inference_args}")
        debugging = False #True
        print('# of samples for getting content and style code: %d' % len(data_loader))
        
        with torch.no_grad():
            content_dict = {}
            content_list = []
            content_fname_list = []        
            style_dict = {}
            style_list = []
            style_fname_list = []
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration=-1)
            with torch.no_grad():
                #style_tensors, style_filenames, style_dirname = net_G.get_style_code(data, **vars(inference_args))
                content, content_filenames, content_dirname, style, style_filenames, style_dirname = net_G.get_content_and_style_code(data, **vars(inference_args))
                
                for cont, fn in zip(content, content_filenames):
                    if fn not in content_dict:
                        content_dict[fn] = cont                        
                        content_list.append(cont)
                        content_fname_list.append(fn)
                        
                for st, fn in zip(style, style_filenames):
                    if fn not in style_dict:
                        style_dict[fn] = st                        
                        style_list.append(st)
                        style_fname_list.append(fn)
                        
        #contents = torch.cat([x.unsqueeze(0) for x in content_list], 0)
        styles   = torch.cat([x.unsqueeze(0) for x in style_list], 0)
        
        import numpy as np
        import shutil
        #content = contents[tsne_one_image_id].unsqueeze(0)
        content = content_list[tsne_one_image_id].unsqueeze(0)
        print(f'The one image to translate is {content_fname_list[tsne_one_image_id]}')
        content_image_src = os.path.join(content_dirname, f'{content_fname_list[tsne_one_image_id]}.jpg')
        content_image_copy = os.path.join(output_dir, f'../image_{tsne_one_image_id}_a2b_{dict_inference_args["a2b"]}.jpg')
        print(f'Make a copy of content image from {content_image_src} to \n {content_image_copy}')
        shutil.copyfile(content_image_src, content_image_copy)
        
        print(f'Translating one image id {tsne_one_image_id} for all styles...')
        for file_names, style in tqdm(style_dict.items()): #zip(styles, style_fname_list):
            style = style.unsqueeze(0)            
            #for style, file_names in zip(styles, style_fname_list):
            with torch.no_grad():
                output_images = net_G.inference_tensor(content, style, **vars(inference_args))
                file_names = np.atleast_1d(file_names)
            for output_image, file_name in zip(output_images, file_names):
                fullname = os.path.join(output_dir, file_name + '.jpg')
                if debugging:
                    print(fullname)
                output_image = tensor2pilimage(output_image.clamp_(-1, 1),
                                               minus1to1_normalized=True)
                save_pilimage_in_jpeg(fullname, output_image)
                
        if debugging:
            print(f'contents.size(): {contents.size()}')            
            print(f'len(content_list): {len(content_list)}')            
            print(f'len(content_dict): {len(content_dict)}')        
            print(f'styles.size(): {styles.size()}')           
            print(f'len(style_list): {len(style_list)}')            
            print(f'len(style_dict): {len(style_dict)}')
            
            print(f'last content.size(): {content.size()}')    
            print(f'last style.size(): {style.size()}')   
            
        #import numpy as np
        content_data = {}
        style_data   = {}
        
        #content_data['data'] = contents.detach().cpu().squeeze().numpy()        
        style_data['data']   = styles.detach().cpu().squeeze().numpy()
        
        #content_data['filename'] = np.asarray(content_fname_list)
        style_data['filename']   = np.asarray(style_fname_list)
        
        #content_data['dirname'] = content_dirname
        style_data['dirname']   = style_dirname
        
        #content_data['a2b'] = dict_inference_args["a2b"]
        style_data['a2b']   = dict_inference_args["a2b"]
        
        #contents_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_contents.pkl')
        styles_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_styles.pkl')
        
        if debugging:
            print(f'content_data["data"].shape: {content_data["data"].shape}')
            print(f'content_data["dirname"]: {content_data["dirname"]}')
            print(f'content_data["filename"].shape: {content_data["filename"].shape}')
            print(f'style_data["data"].shape: {style_data["data"].shape}')
            print(f'style_data["dirname"]: {style_data["dirname"]}')
            print(f'style_data["filename"].shape: {style_data["filename"].shape}')
            
        #print('Saving content and style codes to {} and\n {}'.format(contents_pkl, styles_pkl))
        print(f'Saving style codes to {styles_pkl}')
        
        import pickle
        '''
        with open(contents_pkl, 'wb') as f:
            pickle.dump(content_data, f)
        '''
        with open(styles_pkl, 'wb') as f:
            pickle.dump(style_data, f)
        '''
        with open(contents_pkl, 'rb') as f:
            content_data = pickle.load(f)
        with open(styles_pkl, 'rb') as f:
            style_data = pickle.load(f)
            
        path = [os.path.join(content_data['dirname'], fn + '.jpg') for fn in content_data['filename']]
        '''        
        
    def test_tsne_one_image_bak(self, data_loader, output_dir, tsne_one_image_id, inference_args):
        if self.cfg.trainer.model_average_config.enabled:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()
        
        dict_inference_args = dict(inference_args)
        print(f"dict_inference_args: {dict_inference_args}")
        debugging = False #True
        print('# of samples for getting content and style code: %d' % len(data_loader))
        
        content_dict = {}
        content_list = []
        content_fname_list = []        
        style_dict = {}
        style_list = []
        style_fname_list = []
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration=-1)
            with torch.no_grad():
                #style_tensors, style_filenames, style_dirname = net_G.get_style_code(data, **vars(inference_args))
                content, content_filenames, content_dirname, style, style_filenames, style_dirname = net_G.get_content_and_style_code(data, **vars(inference_args))
                
                for cont, fn in zip(content, content_filenames):
                    if fn not in content_dict:
                        content_dict[fn] = cont                        
                        content_list.append(cont)
                        content_fname_list.append(fn)
                        
                for st, fn in zip(style, style_filenames):
                    if fn not in style_dict:
                        style_dict[fn] = st                        
                        style_list.append(st)
                        style_fname_list.append(fn)
                        
        contents = torch.cat([x.unsqueeze(0) for x in content_list], 0)
        styles   = torch.cat([x.unsqueeze(0) for x in style_list], 0)
        
        import numpy as np
        import shutil
        content = contents[tsne_one_image_id].unsqueeze(0)
        print(f'The one image to translate is {content_fname_list[tsne_one_image_id]}')
        content_image_src = os.path.join(content_dirname, f'{content_fname_list[tsne_one_image_id]}.jpg')
        content_image_copy = os.path.join(output_dir, f'../image_{tsne_one_image_id}_a2b_{dict_inference_args["a2b"]}.jpg')
        print(f'Make a copy of content image from {content_image_src} to \n {content_image_copy}')
        shutil.copyfile(content_image_src, content_image_copy)
        
        print(f'Translating one image id {tsne_one_image_id} for all styles...')
        for file_names, style in tqdm(style_dict.items()): #zip(styles, style_fname_list):
            style = style.unsqueeze(0)            
            #for style, file_names in zip(styles, style_fname_list):
            with torch.no_grad():
                output_images = net_G.inference_tensor(content, style, **vars(inference_args))
                file_names = np.atleast_1d(file_names)
            for output_image, file_name in zip(output_images, file_names):
                fullname = os.path.join(output_dir, file_name + '.jpg')
                if debugging:
                    print(fullname)
                output_image = tensor2pilimage(output_image.clamp_(-1, 1),
                                               minus1to1_normalized=True)
                save_pilimage_in_jpeg(fullname, output_image)
                
        if debugging:
            print(f'contents.size(): {contents.size()}')            
            print(f'len(content_list): {len(content_list)}')            
            print(f'len(content_dict): {len(content_dict)}')        
            print(f'styles.size(): {styles.size()}')           
            print(f'len(style_list): {len(style_list)}')            
            print(f'len(style_dict): {len(style_dict)}')
            
            print(f'last content.size(): {content.size()}')    
            print(f'last style.size(): {style.size()}')   
            
        #import numpy as np
        content_data = {}
        style_data   = {}
        
        content_data['data'] = contents.detach().cpu().squeeze().numpy()        
        style_data['data']   = styles.detach().cpu().squeeze().numpy()
        
        content_data['filename'] = np.asarray(content_fname_list)
        style_data['filename']   = np.asarray(style_fname_list)
        
        content_data['dirname'] = content_dirname
        style_data['dirname']   = style_dirname
        
        content_data['a2b'] = dict_inference_args["a2b"]
        style_data['a2b']   = dict_inference_args["a2b"]
        
        contents_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_contents.pkl')
        styles_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_styles.pkl')
        
        if debugging:
            print(f'content_data["data"].shape: {content_data["data"].shape}')
            print(f'content_data["dirname"]: {content_data["dirname"]}')
            print(f'content_data["filename"].shape: {content_data["filename"].shape}')
            print(f'style_data["data"].shape: {style_data["data"].shape}')
            print(f'style_data["dirname"]: {style_data["dirname"]}')
            print(f'style_data["filename"].shape: {style_data["filename"].shape}')
            
        print('Saving content and style codes to {} and\n {}'.format(contents_pkl, styles_pkl))
        
        import pickle
        with open(contents_pkl, 'wb') as f:
            pickle.dump(content_data, f)
        with open(styles_pkl, 'wb') as f:
            pickle.dump(style_data, f)
        '''
        with open(contents_pkl, 'rb') as f:
            content_data = pickle.load(f)
        with open(styles_pkl, 'rb') as f:
            style_data = pickle.load(f)
            
        path = [os.path.join(content_data['dirname'], fn + '.jpg') for fn in content_data['filename']]
        '''

    def test_tsne_one_image_discriminator(self, data_loader, output_dir, tsne_one_image_id, inference_args):
        if self.cfg.trainer.model_average_config.enabled:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()

        # net_D = self.net_D.module
        # print(f'self.net_D = :\n{self.net_D} \nself.net_G = :\n{self.net_G}')
        # return

        dict_inference_args = dict(inference_args)
        print(f"dict_inference_args: {dict_inference_args}")
        debugging = False  # True
        print('# of samples for getting content and style code: %d' % len(data_loader))

        with torch.no_grad():
            content_dict = {}
            content_list = []
            content_fname_list = []
            style_dict = {}
            style_list = []
            style_fname_list = []
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration=-1)
            with torch.no_grad():
                # style_tensors, style_filenames, style_dirname = net_G.get_style_code(data, **vars(inference_args))
                content, content_filenames, content_dirname, style, style_filenames, style_dirname = net_G.get_content_and_style_code(
                    data, **vars(inference_args))

                for cont, fn in zip(content, content_filenames):
                    if fn not in content_dict:
                        content_dict[fn] = cont
                        content_list.append(cont)
                        content_fname_list.append(fn)

                for st, fn in zip(style, style_filenames):
                    if fn not in style_dict:
                        style_dict[fn] = st
                        style_list.append(st)
                        style_fname_list.append(fn)

        # contents = torch.cat([x.unsqueeze(0) for x in content_list], 0)
        styles = torch.cat([x.unsqueeze(0) for x in style_list], 0)

        import numpy as np
        import shutil

        # if not os.path.isdir(output_dir):
        #     os.makedirs(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # content = contents[tsne_one_image_id].unsqueeze(0)
        content = content_list[tsne_one_image_id].unsqueeze(0)
        print(f'The one image to translate is {content_fname_list[tsne_one_image_id]}')
        content_image_src = os.path.join(content_dirname, f'{content_fname_list[tsne_one_image_id]}.jpg')
        content_image_copy = os.path.join(output_dir,
                                          f'../image_{tsne_one_image_id}_a2b_{dict_inference_args["a2b"]}.jpg')
        print(f'Make a copy of content image from {content_image_src} to \n {content_image_copy}')
        shutil.copyfile(content_image_src, content_image_copy)

        print(f'Translating one image id {tsne_one_image_id} for all styles...')
        for file_names, style in tqdm(style_dict.items()):  # zip(styles, style_fname_list):
            style = style.unsqueeze(0)
            # for style, file_names in zip(styles, style_fname_list):
            with torch.no_grad():
                output_images = net_G.inference_tensor(content, style, **vars(inference_args))
                file_names = np.atleast_1d(file_names)
                discriminator_outputs, _ = self.net_D.module.inference(output_images, **vars(inference_args))
                # print(f'discriminator_outputs: \n{discriminator_outputs}')
                # continue

            for output_image, file_name, disc_out in zip(output_images, file_names, discriminator_outputs):
                fullname = os.path.join(output_dir, file_name + '.jpg')
                if debugging:
                    print(fullname)
                print(f'disc_out: {disc_out}')
                if disc_out.item() <= -1:
                # if disc_out.item() >= 1:
                    print(f'disc_out: {disc_out}')
                    output_image = tensor2pilimage(output_image.clamp_(-1, 1),
                                                   minus1to1_normalized=True)
                    save_pilimage_in_jpeg(fullname, output_image)
        print("Debug Done, return.")
        return

        if debugging:
            print(f'contents.size(): {contents.size()}')
            print(f'len(content_list): {len(content_list)}')
            print(f'len(content_dict): {len(content_dict)}')
            print(f'styles.size(): {styles.size()}')
            print(f'len(style_list): {len(style_list)}')
            print(f'len(style_dict): {len(style_dict)}')

            print(f'last content.size(): {content.size()}')
            print(f'last style.size(): {style.size()}')

            # import numpy as np
        content_data = {}
        style_data = {}

        # content_data['data'] = contents.detach().cpu().squeeze().numpy()
        style_data['data'] = styles.detach().cpu().squeeze().numpy()

        # content_data['filename'] = np.asarray(content_fname_list)
        style_data['filename'] = np.asarray(style_fname_list)

        # content_data['dirname'] = content_dirname
        style_data['dirname'] = style_dirname

        # content_data['a2b'] = dict_inference_args["a2b"]
        style_data['a2b'] = dict_inference_args["a2b"]

        # contents_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_contents.pkl')
        styles_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_styles.pkl')

        if debugging:
            print(f'content_data["data"].shape: {content_data["data"].shape}')
            print(f'content_data["dirname"]: {content_data["dirname"]}')
            print(f'content_data["filename"].shape: {content_data["filename"].shape}')
            print(f'style_data["data"].shape: {style_data["data"].shape}')
            print(f'style_data["dirname"]: {style_data["dirname"]}')
            print(f'style_data["filename"].shape: {style_data["filename"].shape}')

        # print('Saving content and style codes to {} and\n {}'.format(contents_pkl, styles_pkl))
        print(f'Saving style codes to {styles_pkl}')

        import pickle
        '''
        with open(contents_pkl, 'wb') as f:
            pickle.dump(content_data, f)
        '''
        with open(styles_pkl, 'wb') as f:
            pickle.dump(style_data, f)
        '''
        with open(contents_pkl, 'rb') as f:
            content_data = pickle.load(f)
        with open(styles_pkl, 'rb') as f:
            style_data = pickle.load(f)

        path = [os.path.join(content_data['dirname'], fn + '.jpg') for fn in content_data['filename']]
        '''


    def test_tsne_one_image_classifier_bak(self, data_loader, output_dir, tsne_one_image_id, classifier, inference_args):
        if self.cfg.trainer.model_average_config.enabled:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()

        # net_D = self.net_D.module
        # print(f'self.net_D = :\n{self.net_D} \nself.net_G = :\n{self.net_G}')
        # return

        dict_inference_args = dict(inference_args)
        print(f"dict_inference_args: {dict_inference_args}")
        debugging = False  # True
        print('# of samples for getting content and style code: %d' % len(data_loader))

        with torch.no_grad():
            content_dict = {}
            content_list = []
            content_fname_list = []
            style_dict = {}
            style_list = []
            style_fname_list = []
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration=-1)
            with torch.no_grad():
                # style_tensors, style_filenames, style_dirname = net_G.get_style_code(data, **vars(inference_args))
                content, content_filenames, content_dirname, style, style_filenames, style_dirname = net_G.get_content_and_style_code(
                    data, **vars(inference_args))

                for cont, fn in zip(content, content_filenames):
                    if fn not in content_dict:
                        content_dict[fn] = cont
                        content_list.append(cont)
                        content_fname_list.append(fn)

                for st, fn in zip(style, style_filenames):
                    if fn not in style_dict:
                        style_dict[fn] = st
                        style_list.append(st)
                        style_fname_list.append(fn)

        # contents = torch.cat([x.unsqueeze(0) for x in content_list], 0)
        styles = torch.cat([x.unsqueeze(0) for x in style_list], 0)

        import numpy as np
        import shutil

        # if not os.path.isdir(output_dir):
        #     os.makedirs(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # content = contents[tsne_one_image_id].unsqueeze(0)
        content = content_list[tsne_one_image_id].unsqueeze(0)
        content_fn = content_fname_list[tsne_one_image_id]
        print(f'The one image to translate is {content_fname_list[tsne_one_image_id]}')
        content_image_src = os.path.join(content_dirname, f'{content_fname_list[tsne_one_image_id]}.jpg')
        content_image_copy = os.path.join(output_dir,
                                          f'../image_{tsne_one_image_id}_a2b_{dict_inference_args["a2b"]}.jpg')
        print(f'Make a copy of content image from {content_image_src} to \n {content_image_copy}')
        shutil.copyfile(content_image_src, content_image_copy)

        print(f'Translating one image id {tsne_one_image_id} for all styles...')

        fn_lst = []
        dis_lst = []
        cls_lst = []
        img_lst = []
        clscheck_lst = []
        for file_names, style in tqdm(style_dict.items()):  # zip(styles, style_fname_list):
            style = style.unsqueeze(0)
            # for style, file_names in zip(styles, style_fname_list):
            with torch.no_grad():
                output_images = net_G.inference_tensor(content, style, **vars(inference_args))
                file_names = np.atleast_1d(file_names)
                discriminator_outputs, _ = self.net_D.module.inference(output_images, **vars(inference_args))
                discriminator_outputs = discriminator_outputs.detach().cpu().clone().numpy().ravel()
                # Use np.ravel (for a 1D view) or np.ndarray.flatten (for a 1D copy) or np.ndarray.flat (for an 1D iterator):
                # print(f'discriminator_outputs: \n{discriminator_outputs}')
                # continue
                classifier_outputs = classifier.inference(output_images)
            assert len(output_images) == 1 and len(file_names) == 1 and len(discriminator_outputs) == 1 and len(classifier_outputs) == 1, 'Check Error!! len(output_images) == 1 and len(file_names) == 1 and len(discriminator_outputs) == 1 and len(classifier_outputs) == 1'
            for output_image, file_name, disc_score, cls_score in zip(output_images, file_names, discriminator_outputs, classifier_outputs):
                fn_lst.append(file_name)
                # print(f'disc_score: {disc_score} \t cls_score: {cls_score}')
                dis_lst.append(disc_score)
                cls_lst.append(cls_score)

                # to check classifier outputs are the same using either output tensor from inference_tensor or its transformed pil image
                img_lst.append(output_image)
                output_image = tensor2pilimage(output_image.clamp_(-1, 1), minus1to1_normalized=True)
                # img_lst.append(output_image)
                img_tensor = classifier.tranform_image(output_image)
                img_tensor = img_tensor.unsqueeze(0)
                classifier_outputs_check = classifier.inference(img_tensor)
                clscheck_lst.append(classifier_outputs_check[0])

        # df = pd.DataFrame(list(zip(fn_lst, dis_lst, cls_lst, clscheck_lst)), columns=['fn', 'dis', 'cls', 'cls_check'])
        """
        df = pd.DataFrame({
            'fn': fn_lst,
            'dis': dis_lst,
            'cls': cls_lst,
            'img': img_lst,
            'cls_check': clscheck_lst
        })
        
        df['cls_0'] = df['cls'] - df['cls_check']
        print("df['cls_0'].max(), df['cls_0'].min() = {}, {}".format(df['cls_0'].max(), df['cls_0'].min()))

        Oh No! I can't use pandas. Why pandas can't be used here, and matplotlib seems also cause stdout problems. 
        It seems pandas will made the numpy error, so I have to implement my own pandas using numpy, 
        structured numpy array etc. This is stupid! Those package compatibility issues are just rubbish!
        """


        len_lst = len(fn_lst)
        assert len(dis_lst) == len_lst and len(cls_lst) == len_lst and len(img_lst) == len_lst and len(clscheck_lst) == len_lst, 'Check Error!! Mismatching list length!'

        id_lst = list(range(len_lst))
        fn_dict = dict(zip(id_lst, fn_lst))
        img_dict = dict(zip(id_lst, img_lst))

        # structure array dtypes can be dict or list of tuples
        t_nouse = np.dtype({
            'names': ('id', 'dis', 'cls', 'cls_check'),
            'formats': ('uint8', 'float32', 'float32', 'float32')
        })

        t_nouse2 = np.dtype(
            [('id', 'i8'),
             ('dis', 'f8'),
             ('cls', 'f8'),
             ('cls_check', 'f8'),
             ('cls_0', 'f8'),
             # ('img', 'f8', img_lst[0].shape)
             ]
        )

        dt = np.zeros(len_lst, dtype={'names': ('id', 'dis', 'cls', 'cls_check', 'cls_0', 'close_to_mid'),
                                      'formats': ('i4', 'f8', 'f8', 'f8', 'f8', 'f8')})
        dt['id'] = id_lst
        dt['dis'] = dis_lst
        dt['cls'] = cls_lst
        dt['cls_check'] = clscheck_lst
        dt['cls_0'] = dt['cls'] - dt['cls_check']
        dt['close_to_mid'] = np.abs(dt['cls'] - 0.5)
        print('dt[:10]: ', dt[:10])
        # dt_rec = dt.view(np.recarray)
        # dt_rec.id
        # x = np.array([(1, 2, 3), (4, 5, 6)], dtype='i8, f4, f8')
        dt_nouse = np.array(list(zip(id_lst, dis_lst, cls_lst, clscheck_lst)), dtype=t_nouse)
        print('dt_nouse[:10]: ', dt_nouse[:10])

        id = np.asarray(id_lst)
        dis = np.asarray(dis_lst)
        cls = np.asarray(cls_lst)
        cls_check = np.asarray(clscheck_lst)
        cls_0 = cls - cls_check
        dt_nouse2 = np.c_[id, dis, cls, cls_check, cls_0]
        print('dt2[:10]:', dt_nouse2[:10])


        """
        if dict_inference_args["a2b"]: # 0 to 1
            df_sort = df.sort_values(by=['cls'], inplace=False, ascending=False)
            heads_cls = df_sort.head(100)
            df_sort = df.sort_values(by=['cls'], inplace=False, ascending=True)
            tails_cls = df_sort.head(100)
            df_sort['close_to_mid'] = (df_sort['cls'] - 0.5).abs()
            mids_cls = df_sort.sort_values(by=['close_to_mid'], inplace=False, ascending=True).head(100)
        else: # 1 to 0
            df_sort = df.sort_values(by=['cls'], inplace=False, ascending=True)
            heads_cls = df_sort.head(100)
            df_sort = df.sort_values(by=['cls'], inplace=False, ascending=False)
            tails_cls = df_sort.head(100)
            df_sort['close_to_mid'] = (df_sort['cls'] - 0.5).abs()
            mids_cls = df_sort.sort_values(by=['close_to_mid'], inplace=False, ascending=True).head(100)

        df_sort = df.sort_values(by=['dis'], inplace=False, ascending=False)
        heads_dis = df_sort.head(100)
        df_sort = df.sort_values(by=['dis'], inplace=False, ascending=True)
        tails_dis = df_sort.head(100)
        """

        sorted_array = np.sort(dt, order='cls')
        heads_cls = sorted_array[::-1][:100] if dict_inference_args["a2b"] else sorted_array[:100]
        tails_cls = sorted_array[:100] if dict_inference_args["a2b"] else sorted_array[::-1][:100]
        mids_cls = np.sort(dt, order='close_to_mid')[:100]


        sorted_dis = np.sort(dt, order='dis')
        heads_dis = sorted_dis[::-1][:100]
        tails_dis = sorted_dis[:100]

        ncols = 10
        nrows = 10
        width = 15
        heigt = 15.8
        """
        
        import matplotlib.pyplot as plt
        # shit!! I can't use matplotlib here, otherwise it will cause numpy error. This is stupid! I have to implement an alternative way - torchgrid.

        target_domain = 'B' if dict_inference_args["a2b"] else 'A'

        for df, pos in zip([heads_cls, tails_cls, mids_cls], ['heads_cls', 'tails_cls', 'mids_cls']):
            fig, axis = plt.subplots(nrows, ncols, figsize=(width, heigt))
            for ax, id, probability in zip(axis.flat, df['id'].to_list(), df['cls'].to_list()):
                img = img_dict[id]
                # fn = fn_dict[id]
                ax.imshow(img)
                title = f'{probability:.2f}' if dict_inference_args["a2b"] else f'{1-probability:.2f}'
                ax.axis('off')
                ax.set_title(title)
            fig.suptitle(f'{pos} images for probability of Domain {target_domain}', fontsize=16)
            fullname = os.path.join(output_dir, '{}_{}.jpg'.format(content_fn, pos))
            print('saving {}'.format(fullname))
            fig.savefig(fullname, bbox_inches='tight')

        for df, pos in zip([heads_dis, tails_dis], ['heads_dis', 'tails_dis']):
            fig, axis = plt.subplots(nrows, ncols, figsize=(width, heigt))
            for ax, id, dis in zip(axis.flat, df['id'].to_list(), df['dis'].to_list()):
                img = img_dict[id]
                # fn = fn_dict[id]
                ax.imshow(img)
                title = f'{dis:.3f}'
                ax.axis('off')
                ax.set_title(title)
            fig.suptitle(f'{pos} images for discriminator score', fontsize=16)
            fullname = os.path.join(output_dir, '{}_{}.jpg'.format(content_fn, pos))
            print('saving {}'.format(fullname))
            fig.savefig(fullname, bbox_inches='tight')
            
        """

        target_domain = 'B' if dict_inference_args["a2b"] else 'A'

        for df, pos in zip([heads_cls, tails_cls, mids_cls, heads_dis, tails_dis], ['heads_cls', 'tails_cls', 'mids_cls', 'heads_dis', 'tails_dis']):
            fullname = os.path.join(output_dir, '{}_{}.jpg'.format(content_fn, pos))
            fullname_txt = os.path.join(output_dir, '{}_{}.txt'.format(content_fn, pos))
            vis_images = torch.cat([img_dict[id].unsqueeze(0) for id in df['id']], dim=0).float()
            vis_images = (vis_images + 1) / 2
            vis_images.clamp_(0, 1)
            os.makedirs(os.path.dirname(fullname), exist_ok=True)
            # print(f'vis_images.size(): {vis_images.size()}')
            image_grid = torchvision.utils.make_grid(vis_images, nrow=nrows, padding=2, normalize=False)
            # torchvision.utils.save_image(image_grid, path, nrow=10)
            print('saving {}'.format(fullname))
            torchvision.transforms.ToPILImage()(image_grid).save(fullname)

            print('saving {}'.format(fullname_txt))
            with open(fullname_txt, "w") as f:
                if pos in ['heads_cls', 'tails_cls', 'mids_cls']:
                    f.write(f'style_filename,probability_of_belonging_to_Domain{target_domain}\n')
                    for id, probability in zip(df['id'], df['cls']):
                        prob = f'{probability:.3f}' if dict_inference_args["a2b"] else f'{1 - probability:.3f}'
                        f.write(f'{fn_dict[id]},{prob}\n')
                elif pos in ['heads_dis', 'tails_dis']:
                    f.write(f'style_filename,discriminator_score\n')
                    for id, dis in zip(df['id'], df['dis']):
                        f.write(f'{fn_dict[id]},{dis:.3f}\n')
                else:
                    print(f"Wrong pos value! pos = {pos}. Check Error!!")

        print("Debug Done, return.")
        return


        if debugging:
            print(f'contents.size(): {contents.size()}')
            print(f'len(content_list): {len(content_list)}')
            print(f'len(content_dict): {len(content_dict)}')
            print(f'styles.size(): {styles.size()}')
            print(f'len(style_list): {len(style_list)}')
            print(f'len(style_dict): {len(style_dict)}')

            print(f'last content.size(): {content.size()}')
            print(f'last style.size(): {style.size()}')

            # import numpy as np
        content_data = {}
        style_data = {}

        # content_data['data'] = contents.detach().cpu().squeeze().numpy()
        style_data['data'] = styles.detach().cpu().squeeze().numpy()

        # content_data['filename'] = np.asarray(content_fname_list)
        style_data['filename'] = np.asarray(style_fname_list)

        # content_data['dirname'] = content_dirname
        style_data['dirname'] = style_dirname

        # content_data['a2b'] = dict_inference_args["a2b"]
        style_data['a2b'] = dict_inference_args["a2b"]

        # contents_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_contents.pkl')
        styles_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_styles.pkl')

        if debugging:
            print(f'content_data["data"].shape: {content_data["data"].shape}')
            print(f'content_data["dirname"]: {content_data["dirname"]}')
            print(f'content_data["filename"].shape: {content_data["filename"].shape}')
            print(f'style_data["data"].shape: {style_data["data"].shape}')
            print(f'style_data["dirname"]: {style_data["dirname"]}')
            print(f'style_data["filename"].shape: {style_data["filename"].shape}')

        # print('Saving content and style codes to {} and\n {}'.format(contents_pkl, styles_pkl))
        print(f'Saving style codes to {styles_pkl}')

        import pickle
        '''
        with open(contents_pkl, 'wb') as f:
            pickle.dump(content_data, f)
        '''
        with open(styles_pkl, 'wb') as f:
            pickle.dump(style_data, f)
        '''
        with open(contents_pkl, 'rb') as f:
            content_data = pickle.load(f)
        with open(styles_pkl, 'rb') as f:
            style_data = pickle.load(f)

        path = [os.path.join(content_data['dirname'], fn + '.jpg') for fn in content_data['filename']]
        '''

    def test_classifier(self, data_loader, output_dir, classifier, inference_args, top_N=10, content_front=True,
                        use_style_loader=True, batch_size_classifier=100, inference_resume=False,
                        include_random_style=False, txt_off=False, all_random_styles=False, batchsize_inference=1, sample_multiplier=10):
        if self.cfg.trainer.model_average_config.enabled:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()

        dict_inference_args = dict(inference_args)
        print(f"dict_inference_args: {dict_inference_args}")
        debugging = False  # True
        print('# of samples for getting content and style code: %d' % len(data_loader))

        content_dict = {}
        content_list = []
        content_fname_list = []
        content_image_list = []
        style_dict = {}
        style_list = []
        style_fname_list = []
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration=-1)
            with torch.no_grad():
                # style_tensors, style_filenames, style_dirname = net_G.get_style_code(data, **vars(inference_args))
                # content, content_filenames, content_dirname, style, style_filenames, style_dirname = net_G.get_content_and_style_code(data, **vars(inference_args))
                content_images, content, content_filenames, content_dirname, _, style, style_filenames, style_dirname = net_G.get_contents_and_styles(
                    data, **vars(inference_args))

                for cont, fn, img in zip(content, content_filenames, content_images):
                    if fn not in content_dict:
                        content_dict[fn] = cont
                        content_list.append(cont)
                        content_fname_list.append(fn)
                        content_image_list.append(img)

                for st, fn in zip(style, style_filenames):
                    if fn not in style_dict:
                        style_dict[fn] = st
                        style_list.append(st)
                        style_fname_list.append(fn)

        # contents = torch.cat([x.unsqueeze(0) for x in content_list], 0)
        styles = torch.cat([x.unsqueeze(0) for x in style_list], 0)

        # if not os.path.isdir(output_dir):
        #     os.makedirs(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f'created {output_dir}')
        print('# of images to translate: %d' % len(content_list))

        style_dict_loader = None
        if use_style_loader:
            from train_classifier import get_style_dict_loader
            style_dict_loader = get_style_dict_loader(style_dict, batch_size_classifier)

        for tsne_one_image_id in tqdm(range(len(content_list))):
            # print(f'{tsne_one_image_id}')
            content = content_list[tsne_one_image_id].unsqueeze(0)
            content_fn = content_fname_list[tsne_one_image_id]
            content_img = content_image_list[tsne_one_image_id]
            if all_random_styles:
                self.translate_one_image_random_styles(output_dir, net_G, classifier, content_img, content, content_fn,
                                                       dict_inference_args, inference_args, top_N=top_N,
                                                       batchsize_inference=batchsize_inference,
                                                       sample_multiplier=sample_multiplier, content_front=content_front,
                                                       inference_resume=inference_resume,
                                                       txt_off=txt_off)
            else:
                self.translate_one_image(output_dir, net_G, classifier, content_img, content, content_fn, style_dict,
                                         content_dirname, dict_inference_args, inference_args, top_N=top_N,
                                         content_front=content_front, style_dict_loader=style_dict_loader,
                                         inference_resume=inference_resume, include_random_style=include_random_style,
                                         txt_off=txt_off)

        self.save_style_codes(debugging, content_list, content_dict, styles, style_list, style_dict, content, style,
                              style_fname_list, style_dirname, dict_inference_args, output_dir)

    def save_style_codes(self, debugging, content_list, content_dict, styles, style_list, style_dict, content, style,
                         style_fname_list, style_dirname, dict_inference_args, output_dir):
        if debugging:
            # print(f'contents.size(): {contents.size()}')
            print(f'len(content_list): {len(content_list)}')
            print(f'len(content_dict): {len(content_dict)}')
            print(f'styles.size(): {styles.size()}')
            print(f'len(style_list): {len(style_list)}')
            print(f'len(style_dict): {len(style_dict)}')
            print(f'last content.size(): {content.size()}')
            print(f'last style.size(): {style.size()}')

        # import numpy as np
        content_data = {}
        style_data = {}

        # content_data['data'] = contents.detach().cpu().squeeze().numpy()
        style_data['data'] = styles.detach().cpu().squeeze().numpy()
        # content_data['filename'] = np.asarray(content_fname_list)
        style_data['filename'] = np.asarray(style_fname_list)
        # content_data['dirname'] = content_dirname
        style_data['dirname'] = style_dirname
        # content_data['a2b'] = dict_inference_args["a2b"]
        style_data['a2b'] = dict_inference_args["a2b"]
        # contents_pkl = os.path.join(output_dir, f'../styles_a2b_{dict_inference_args["a2b"]}_contents.pkl')
        styles_pkl = os.path.join(output_dir, f'styles_a2b_{dict_inference_args["a2b"]}_styles.pkl')

        if debugging:
            print(f'content_data["data"].shape: {content_data["data"].shape}')
            print(f'content_data["dirname"]: {content_data["dirname"]}')
            print(f'content_data["filename"].shape: {content_data["filename"].shape}')
            print(f'style_data["data"].shape: {style_data["data"].shape}')
            print(f'style_data["dirname"]: {style_data["dirname"]}')
            print(f'style_data["filename"].shape: {style_data["filename"].shape}')

        # print('Saving content and style codes to {} and\n {}'.format(contents_pkl, styles_pkl))
        print(f'Saving style codes to {styles_pkl}')

        import pickle
        with open(styles_pkl, 'wb') as f:
            pickle.dump(style_data, f)
        '''
        with open(contents_pkl, 'wb') as f:
            pickle.dump(content_data, f)
            
        with open(contents_pkl, 'rb') as f:
            content_data = pickle.load(f)
        with open(styles_pkl, 'rb') as f:
            style_data = pickle.load(f)
    
        path = [os.path.join(content_data['dirname'], fn + '.jpg') for fn in content_data['filename']]
        '''

    def translate_one_image(self, output_dir, net_G, classifier, content_img, content, content_fn, style_dict,
                            content_dirname, dict_inference_args, inference_args, top_N=10, content_front=True,
                            style_dict_loader=None, inference_resume=False, include_random_style=False,
                            txt_off=False):
        # print(f'translating {content_fn}.jpg')
        # content_image_src = os.path.join(content_dirname, f'{content_fn}.jpg')
        # content_image_copy = os.path.join(output_dir, f'{content_fn}_a2b_{dict_inference_args["a2b"]}.jpg')
        # print(f'Make a copy of content image from {content_image_src} to \n {content_image_copy}')
        # shutil.copyfile(content_image_src, content_image_copy)
        if inference_resume:
            if os.path.exists(os.path.join(output_dir, f'{content_fn}_heads_cls.jpg')):
                print(f'existed, skipping {content_fn}_heads_cls.jpg')
                return

        fn_lst = []
        cls_lst = []
        img_lst = []
        if style_dict_loader is not None:
            for file_names, styles in style_dict_loader:
                # print(f'content.shape: {content.shape}, styles.shape: {styles.shape}')
                # content = content.expand_as(styles)
                contents = content.expand(styles.size(dim=0), -1, -1, -1)
                # print(f'contents.shape: {contents.shape}, styles.shape: {styles.shape}')
                # style = style.unsqueeze(0)
                with torch.no_grad():
                    output_images = net_G.inference_tensor(contents, styles, **vars(inference_args))
                    # file_names = np.atleast_1d(file_names)
                    classifier_outputs = classifier.inference(output_images)
                assert len(output_images) == len(file_names) == len(
                    classifier_outputs), 'Check Error!! len(output_images) == len(file_names) == len(classifier_outputs)'
                for output_image, file_name, cls_score in zip(output_images, file_names, classifier_outputs):
                    fn_lst.append(file_name)
                    cls_lst.append(cls_score)
                    img_lst.append(output_image)
        else:
            for file_names, style in style_dict.items():  # tqdm(style_dict.items()):  # zip(styles, style_fname_list):
                style = style.unsqueeze(0)
                with torch.no_grad():
                    output_images = net_G.inference_tensor(content, style, **vars(inference_args))
                    file_names = np.atleast_1d(file_names)
                    classifier_outputs = classifier.inference(output_images)
                assert len(output_images) == 1 and len(file_names) == 1 and len(
                    classifier_outputs) == 1, 'Check Error!! len(output_images) == 1 and len(file_names) == 1 and len(classifier_outputs) == 1'
                for output_image, file_name, cls_score in zip(output_images, file_names, classifier_outputs):
                    fn_lst.append(file_name)
                    cls_lst.append(cls_score)
                    img_lst.append(output_image)

        if include_random_style:
            with torch.no_grad():
                output_images = net_G.inference_tensor_random(content, **vars(inference_args))
                file_names = 'random_style'
                file_names = np.atleast_1d(file_names)
                classifier_outputs = classifier.inference(output_images)
            assert len(output_images) == 1 and len(file_names) == 1 and len(
                classifier_outputs) == 1, 'Check Error!! len(output_images) == 1 and len(file_names) == 1 and len(classifier_outputs) == 1'
            for output_image, file_name, cls_score in zip(output_images, file_names, classifier_outputs):
                fn_lst.append(file_name)
                cls_lst.append(cls_score)
                img_lst.append(output_image)

        len_lst = len(fn_lst)
        assert len(cls_lst) == len_lst and len(img_lst) == len_lst, 'Check Error!! Mismatching list length!'

        id_lst = list(range(len_lst))
        fn_dict = dict(zip(id_lst, fn_lst))
        img_dict = dict(zip(id_lst, img_lst))

        dt = np.zeros(len_lst, dtype={'names': ('id', 'cls', 'close_to_mid'),
                                      'formats': ('i4', 'f8', 'f8')})
        dt['id'] = id_lst
        dt['cls'] = cls_lst
        dt['close_to_mid'] = np.abs(dt['cls'] - 0.5)
        # print('dt[:10]: ', dt[:10])

        sorted_array = np.sort(dt, order='cls')
        heads_cls = sorted_array[::-1][:top_N] if dict_inference_args["a2b"] else sorted_array[:top_N]
        # tails_cls = sorted_array[:top_N] if dict_inference_args["a2b"] else sorted_array[::-1][:top_N]
        # mids_cls = np.sort(dt, order='close_to_mid')[:top_N]

        # nrow (int, optional) – Number of images displayed in each row of the grid.
        if top_N > 10:
            nrows = math.ceil(math.sqrt(top_N))
        else:
            nrows = top_N + 1

        target_domain = 'B' if dict_inference_args["a2b"] else 'A'
        for df, pos in zip([heads_cls], ['heads_cls']):
            fullname = os.path.join(output_dir, '{}_{}.jpg'.format(content_fn, pos))
            fullname_txt = os.path.join(output_dir, '{}_{}.txt'.format(content_fn, pos))

            vis_images = torch.cat([img_dict[id].unsqueeze(0) for id in df['id']], dim=0).float()
            content_img = content_img.unsqueeze(0)
            # print(f'before torch.cat: content_img.size() = {content_img.size()}; vis_images.size() = {vis_images.size()}')
            if content_front:
                vis_images = torch.cat([content_img, vis_images], dim=0)
            else:
                vis_images = torch.cat([vis_images, content_img], dim=0)
            vis_images = (vis_images + 1) / 2
            vis_images.clamp_(0, 1)
            # os.makedirs(os.path.dirname(fullname), exist_ok=True)
            # print(f'vis_images.size(): {vis_images.size()}')
            image_grid = torchvision.utils.make_grid(vis_images, nrow=nrows, padding=2, normalize=False)
            # torchvision.utils.save_image(image_grid, path, nrow=10)
            print('saving {}'.format(fullname))
            torchvision.transforms.ToPILImage()(image_grid).save(fullname)

            if not txt_off:
                # print('saving {}'.format(fullname_txt))
                with open(fullname_txt, "w") as f:
                    if pos in ['heads_cls', 'tails_cls', 'mids_cls']:
                        f.write(f'style_filename,probability_of_belonging_to_Domain{target_domain}\n')
                        for id, probability in zip(df['id'], df['cls']):
                            prob = f'{probability:.3f}' if dict_inference_args["a2b"] else f'{1 - probability:.3f}'
                            f.write(f'{fn_dict[id]},{prob}\n')
                    else:
                        print(f"Wrong pos value! pos = {pos}. Check Error!!")

    def translate_one_image_random_styles(self, output_dir, net_G, classifier, content_img, content, content_fn,
                                          dict_inference_args, inference_args, top_N=10, batchsize_inference=1,
                                          sample_multiplier=10, content_front=True, inference_resume=False,
                                          txt_off=False):
        out_fn = f'{content_fn}_heads_cls.jpg'
        if inference_resume:
            if os.path.exists(os.path.join(output_dir, f'{out_fn}')):
                print(f'existed, skipping {out_fn}')
                return

        fn_lst = []
        cls_lst = []
        img_lst = []

        for i in range(top_N * sample_multiplier):
            with torch.no_grad():
                contents = content.expand(batchsize_inference, -1, -1, -1)
                output_images = net_G.inference_tensor_random(contents, **vars(inference_args))
                file_names = [f'random_style_{i}_{j}' for j in range(batchsize_inference)]
                file_names = np.atleast_1d(file_names)
                classifier_outputs = classifier.inference(output_images)
            assert len(output_images) == len(file_names) == len(
                classifier_outputs), 'Check Error!! len(output_images) == len(file_names) == len(classifier_outputs)'
            for output_image, file_name, cls_score in zip(output_images, file_names, classifier_outputs):
                fn_lst.append(file_name)
                cls_lst.append(cls_score)
                img_lst.append(output_image)

        len_lst = len(fn_lst)
        assert len(cls_lst) == len_lst and len(img_lst) == len_lst, 'Check Error!! Mismatching list length!'

        id_lst = list(range(len_lst))
        fn_dict = dict(zip(id_lst, fn_lst))
        img_dict = dict(zip(id_lst, img_lst))

        dt = np.zeros(len_lst, dtype={'names': ('id', 'cls', 'close_to_mid'),
                                      'formats': ('i4', 'f8', 'f8')})
        dt['id'] = id_lst
        dt['cls'] = cls_lst
        dt['close_to_mid'] = np.abs(dt['cls'] - 0.5)
        # print('dt[:10]: ', dt[:10])

        sorted_array = np.sort(dt, order='cls')
        heads_cls = sorted_array[::-1][:top_N] if dict_inference_args["a2b"] else sorted_array[:top_N]
        # tails_cls = sorted_array[:top_N] if dict_inference_args["a2b"] else sorted_array[::-1][:top_N]
        # mids_cls = np.sort(dt, order='close_to_mid')[:top_N]

        # nrow (int, optional) – Number of images displayed in each row of the grid.
        if top_N > 10:
            nrows = math.ceil(math.sqrt(top_N))
        else:
            nrows = top_N + 1

        txt_out_dir = os.path.join(output_dir, 'txt')
        if not os.path.exists(txt_out_dir):
            os.makedirs(txt_out_dir, exist_ok=True)
            print(f'created {txt_out_dir}')

        target_domain = 'B' if dict_inference_args["a2b"] else 'A'
        df = heads_cls
        pos = 'heads_cls'
        fullname = os.path.join(output_dir, out_fn)
        fullname_txt = os.path.join(txt_out_dir, f'{content_fn}_heads_cls.txt')

        vis_images = torch.cat([img_dict[id].unsqueeze(0) for id in df['id']], dim=0).float()
        content_img = content_img.unsqueeze(0)
        # print(f'before torch.cat: content_img.size() = {content_img.size()}; vis_images.size() = {vis_images.size()}')
        if content_front:
            vis_images = torch.cat([content_img, vis_images], dim=0)
        else:
            vis_images = torch.cat([vis_images, content_img], dim=0)
        vis_images = (vis_images + 1) / 2
        vis_images.clamp_(0, 1)
        # os.makedirs(os.path.dirname(fullname), exist_ok=True)
        # print(f'vis_images.size(): {vis_images.size()}')
        image_grid = torchvision.utils.make_grid(vis_images, nrow=nrows, padding=2, normalize=False)
        # torchvision.utils.save_image(image_grid, path, nrow=10)
        print('saving {}'.format(fullname))
        torchvision.transforms.ToPILImage()(image_grid).save(fullname)

        if not txt_off:
            # print('saving {}'.format(fullname_txt))
            with open(fullname_txt, "w") as f:
                if pos in ['heads_cls', 'tails_cls', 'mids_cls']:
                    f.write(f'style_filename,probability_of_belonging_to_Domain{target_domain}\n')
                    for id, probability in zip(df['id'], df['cls']):
                        prob = f'{probability:.3f}' if dict_inference_args["a2b"] else f'{1 - probability:.3f}'
                        f.write(f'{fn_dict[id]},{prob}\n')
                else:
                    print(f"Wrong pos value! pos = {pos}. Check Error!!")


    def test_classifier_score_map(self, data_loader, output_dir, classifier, inference_args):
        if self.cfg.trainer.model_average_config.enabled:
            net_G = self.net_G.module.averaged_model
        else:
            net_G = self.net_G.module
        net_G.eval()

        dict_inference_args = dict(inference_args)
        print(f"dict_inference_args: {dict_inference_args}")
        print('# of samples for getting content and style code: %d' % len(data_loader))

        content_dict = {}
        content_list = []
        content_fname_list = []
        content_image_list = []
        style_dict = {}
        style_list = []
        style_fname_list = []
        style_image_list = []
        for it, data in enumerate(tqdm(data_loader)):
            data = self.start_of_iteration(data, current_iteration=-1)
            with torch.no_grad():
                content_images, content, content_filenames, content_dirname, style_images, style, style_filenames, style_dirname = net_G.get_contents_and_styles(
                    data, **vars(inference_args))

                for cont, fn, img in zip(content, content_filenames, content_images):
                    if fn not in content_dict:
                        content_dict[fn] = cont
                        content_list.append(cont)
                        content_fname_list.append(fn)
                        content_image_list.append(img)

                for st, fn, img in zip(style, style_filenames, style_images):
                    if fn not in style_dict:
                        style_dict[fn] = st
                        style_list.append(st)
                        style_fname_list.append(fn)
                        style_image_list.append(img)

        styles = torch.cat([x.unsqueeze(0) for x in style_list], 0)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f'created {output_dir}')

        content_score_list = []
        for tsne_one_image_id in tqdm(range(len(content_list))):
            classifier_output = classifier.inference_one_image(content_image_list[tsne_one_image_id])
            content_score_list.append(classifier_output)

        style_score_list = []
        for tsne_one_image_id in tqdm(range(len(style_list))):
            classifier_output = classifier.inference_one_image(style_image_list[tsne_one_image_id])
            style_score_list.append(classifier_output)

        self.save_classifier_scores(output_dir, content_dirname, content_fname_list, content_score_list, style_dirname, style_fname_list, style_score_list, styles, dict_inference_args)

    def save_classifier_scores(self, output_dir, content_dirname, content_fname_list, content_score_list, style_dirname, style_fname_list, style_score_list, styles, dict_inference_args):
        content_data = {}
        style_data = {}
        # content_data['data'] = contents.detach().cpu().squeeze().numpy()
        style_data['data'] = styles.detach().cpu().squeeze().numpy()
        content_data['filename'] = np.asarray(content_fname_list)
        style_data['filename'] = np.asarray(style_fname_list)
        content_data['dirname'] = content_dirname
        style_data['dirname'] = style_dirname
        content_data['a2b'] = dict_inference_args["a2b"]
        style_data['a2b'] = dict_inference_args["a2b"]
        content_data['score'] = np.asarray(content_score_list)
        style_data['score'] = np.asarray(style_score_list)

        scores_pkl = os.path.join(output_dir, f'scores_a2b_{dict_inference_args["a2b"]}.pkl')
        print('Saving content and style scores to {}'.format(scores_pkl))
        data = {'contents': content_data, 'styles': style_data}
        import pickle
        with open(scores_pkl, 'wb') as f:
            pickle.dump(data, f)

    def _get_total_loss(self, gen_forward):
        r"""Return the total loss to be backpropagated.
        Args:
            gen_forward (bool): If ``True``, backpropagates the generator loss,
                otherwise the discriminator loss.
        """
        losses = self.gen_losses if gen_forward else self.dis_losses
        total_loss = torch.tensor(0., device=torch.device('cuda'))
        # Iterates over all possible losses.
        for loss_name in self.weights:
            # If it is for the current model (gen/dis).
            if loss_name in losses:
                # Multiply it with the corresponding weight
                # and add it to the total loss.
                total_loss += losses[loss_name] * self.weights[loss_name]
        losses['total'] = total_loss  # logging purpose
        return total_loss

    def _detach_losses(self):
        r"""Detach all logging variables to prevent potential memory leak."""
        for loss_name in self.gen_losses:
            self.gen_losses[loss_name] = self.gen_losses[loss_name].detach()
        for loss_name in self.dis_losses:
            self.dis_losses[loss_name] = self.dis_losses[loss_name].detach()

    def _time_before_forward(self):
        r"""
        Record time before applying forward.
        """
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            self.forw_time = time.time()

    def _time_before_loss(self):
        r"""
        Record time before computing loss.
        """
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            self.loss_time = time.time()

    def _time_before_backward(self):
        r"""
        Record time before applying backward.
        """
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            self.back_time = time.time()

    def _time_before_step(self):
        r"""
        Record time before updating the weights
        """
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            self.step_time = time.time()

    def _time_before_model_avg(self):
        r"""
        Record time before applying model average.
        """
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            self.avg_time = time.time()

    def _time_before_leave_gen(self):
        r"""
        Record forward, backward, loss, and model average time for the
        generator update.
        """
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            end_time = time.time()
            self.accu_gen_forw_iter_time += self.loss_time - self.forw_time
            self.accu_gen_loss_iter_time += self.back_time - self.loss_time
            self.accu_gen_back_iter_time += self.step_time - self.back_time
            self.accu_gen_step_iter_time += self.avg_time - self.step_time
            self.accu_gen_avg_iter_time += end_time - self.avg_time

    def _time_before_leave_dis(self):
        r"""
        Record forward, backward, loss time for the discriminator update.
        """
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            end_time = time.time()
            self.accu_dis_forw_iter_time += self.loss_time - self.forw_time
            self.accu_dis_loss_iter_time += self.back_time - self.loss_time
            self.accu_dis_back_iter_time += self.step_time - self.back_time
            self.accu_dis_step_iter_time += end_time - self.step_time


@master_only
def _save_checkpoint(cfg,
                     net_G, net_D,
                     opt_G, opt_D,
                     sch_G, sch_D,
                     current_epoch, current_iteration):
    r"""Save network weights, optimizer parameters, scheduler parameters
    in the checkpoint.

    Args:
        cfg (obj): Global configuration.
        net_D (obj): Discriminator network.
        opt_G (obj): Optimizer for the generator network.
        opt_D (obj): Optimizer for the discriminator network.
        sch_G (obj): Scheduler for the generator optimizer.
        sch_D (obj): Scheduler for the discriminator optimizer.
        current_epoch (int): Current epoch.
        current_iteration (int): Current iteration.
    """
    latest_checkpoint_path = 'epoch_{:05}_iteration_{:09}_checkpoint.pt'.format(
        current_epoch, current_iteration)
    save_path = os.path.join(cfg.logdir, latest_checkpoint_path)
    torch.save(
        {
            'net_G': net_G.state_dict(),
            'net_D': net_D.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict(),
            'sch_G': sch_G.state_dict(),
            'sch_D': sch_D.state_dict(),
            'current_epoch': current_epoch,
            'current_iteration': current_iteration,
        },
        save_path,
    )
    fn = os.path.join(cfg.logdir, 'latest_checkpoint.txt')
    with open(fn, 'wt') as f:
        f.write('latest_checkpoint: %s' % latest_checkpoint_path)
    print('Save checkpoint to {}'.format(save_path))
    return save_path
