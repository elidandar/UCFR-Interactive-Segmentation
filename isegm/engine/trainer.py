import os
import random
import logging
import pprint
from copy import deepcopy
from collections import defaultdict

import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points
from isegm.utils.misc import save_checkpoint
from isegm.utils.serialization import get_config_repr
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from .optimizer import get_optimizer, get_optimizer_with_layerwise_decay


class ISTrainer(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg,
                 trainset, valset,
                 optimizer='adam',
                 optimizer_params=None,
                 layerwise_decay=False,
                 image_dump_interval=1,
                 checkpoint_interval=10,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 net_inputs=('images', 'points','image_gradients'),
                 max_num_next_clicks=0,
                 click_models=None,
                 prev_mask_drop_prob=0.0,
                 use_iterloss=False,
                 iterloss_weights=None,
                 use_random_clicks=True,
                 ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks

        # iterloss
        self.use_iterloss = use_iterloss
        self.iterloss_weights = iterloss_weights
        self.use_random_clicks = use_random_clicks

        self.click_models = click_models
        self.prev_mask_drop_prob = prev_mask_drop_prob

        if cfg.distributed:
            cfg.batch_size //= cfg.ngpus
            cfg.val_batch_size //= cfg.ngpus

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None

        self.trainset = trainset
        self.valset = valset

        logger.info(f'Dataset of {trainset.get_samples_number()} samples was loaded for training.')
        logger.info(f'Dataset of {valset.get_samples_number()} samples was loaded for validation.')

        self.train_data = DataLoader(
            trainset, cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=True, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        self.val_data = DataLoader(
            valset, cfg.val_batch_size,
            sampler=get_sampler(valset, shuffle=False, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        if layerwise_decay:
            self.optim = get_optimizer_with_layerwise_decay(model, optimizer, optimizer_params)
        else:
            self.optim = get_optimizer(model, optimizer, optimizer_params)
        model = self._load_weights(model)

        if cfg.multi_gpu:
            model = get_dp_wrapper(cfg.distributed)(model, device_ids=cfg.gpu_ids,
                                                    output_device=cfg.gpu_ids[0])

        if self.is_master:
            logger.info(model)
            logger.info(get_config_repr(model._config))
            logger.info('Running experiment with config:')
            logger.info(pprint.pformat(cfg, indent=4))

        self.device = cfg.device
        self.net = model.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        if self.click_models is not None:
            for click_model in self.click_models:
                for param in click_model.parameters():
                    param.requires_grad = False
                click_model.to(self.device)
                click_model.eval()

        self.scaler: torch.cuda.amp.GradScaler

    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        logger.info(f'Starting Epoch: {start_epoch}')
        logger.info(f'Total Epochs: {num_epochs}')

        if self.cfg.amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(start_epoch, num_epochs):
            self.training(epoch)
            if validation:
                self.validation(epoch)

    def training(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        if self.cfg.distributed:
            self.train_data.sampler.set_epoch(epoch)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100)\
            if self.is_master else self.train_data

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        train_loss = 0.0
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i
            self.global_step = global_step

            loss, losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data)

            accumulate_grad = ((i + 1) % self.cfg.accumulate_grad == 0) or \
                (i + 1 == len(self.train_data))

            if self.cfg.amp:
                loss /= self.cfg.accumulate_grad
                self.scaler.scale(loss).backward()
                if accumulate_grad:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.optim.zero_grad()
            else:
                loss.backward()
                if accumulate_grad:
                    self.optim.step()
                    self.optim.zero_grad()

            losses_logging['overall'] = loss
            reduce_loss_dict(losses_logging)

            train_loss += losses_logging['overall'].item()

            if self.is_master:
                for loss_name, loss_value in losses_logging.items():
                    self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                       value=loss_value.item(),
                                       global_step=global_step)

                for k, v in self.loss_cfg.items():
                    if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                        v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

                if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                    self.save_visualization(splitted_batch_data, outputs, global_step, prefix='train')

                self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate',
                                   value=self.lr if not hasattr(self, 'lr_scheduler') else self.lr_scheduler.get_lr()[-1],
                                   global_step=global_step)

                tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.4f}')
                for metric in self.train_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for metric in self.train_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}',
                                   value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

            # save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
            #                 epoch=None, multi_gpu=self.cfg.multi_gpu)

            if isinstance(self.checkpoint_interval, (list, tuple)):
                checkpoint_interval = [x for x in self.checkpoint_interval if x[0] <= epoch][-1][1]
            else:
                checkpoint_interval = self.checkpoint_interval

            if epoch % checkpoint_interval == 0:
                save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                                epoch=epoch, multi_gpu=self.cfg.multi_gpu)

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()


                #--------------------------------#
                #  ------- Validation ---------  #
                #--------------------------------#
    def validation(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100) if self.is_master else self.val_data

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_data) + i
            loss, batch_losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, validation=True)

            batch_losses_logging['overall'] = loss
            reduce_loss_dict(batch_losses_logging)
            for loss_name, loss_value in batch_losses_logging.items():
                losses_logging[loss_name].append(loss_value.item())

            val_loss += batch_losses_logging['overall'].item()

            if self.is_master:
                tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss/(i + 1):.4f}')
                for metric in self.val_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                                   global_step=epoch, disable_avg=True)

            for metric in self.val_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)


    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            image, image_grad, gt_mask, points = batch_data['images'], batch_data['images_grad'], batch_data['instances'], batch_data['points']

            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
            prev_output_edges = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]

            loss = 0.0

            if not self.use_random_clicks:
                points[:] = -1
                points = get_next_points(prev_output,
                                         gt_mask,
                                         points)

            # Iteractive Click Loss (ICL)
            num_iters = random.randint(1, self.max_num_next_clicks)
            if self.use_iterloss:
                # iterloss
                for click_indx in range(num_iters):

                    # v1
                    # net_input = torch.cat((image, prev_output), dim=1) \
                    #     if self.net.with_prev_mask else image
                    # v2
                    # TODO: change image concat input
                    # TODO: change needed if ICL is used
                    
                    # net_input = torch.cat((image, prev_output.detach()), dim=1) \
                    #     if self.net.with_prev_mask else image
                    
                    net_input = torch.cat((image, prev_output, prev_output_edges), dim=1) \
                    if self.net.with_prev_mask and self.net.with_prev_edges \
                        else torch.cat((image, prev_output), dim=1) \
                            if self.net.with_prev_mask \
                                else image
                    
                    
                    output = self._forward(self.net, net_input, points)
                    
                    
                    loss = self.add_loss(
                        'instance_loss', 
                        loss, 
                        losses_logging, 
                        validation,
                        lambda: (output['instances'], batch_data['instances']),
                        iterloss_step=click_indx,
                        iterloss_weight=self.iterloss_weights[click_indx])
                    
                    loss = self.add_loss(
                        'edges', 
                        loss, 
                        losses_logging, 
                        validation,
                        lambda: (output['edges'], batch_data['edges']),
                        iterloss_step=click_indx,
                        iterloss_weight=self.iterloss_weights[click_indx])
                    
                    loss = self.add_loss(
                        'instance_aux_loss', 
                        loss, 
                        losses_logging, 
                        validation,
                        lambda: (output['instances'], batch_data['instances']),
                        iterloss_step=click_indx,
                        iterloss_weight=self.iterloss_weights[click_indx])

                    prev_output = torch.sigmoid(output['instances'])
                    if click_indx < num_iters - 1:
                        points = get_next_points(prev_output,
                                                gt_mask,
                                                points)

                    if self.net.with_prev_mask and self.prev_mask_drop_prob > 0:
                        zero_mask = np.random.random(size=prev_output.size(0)) < self.prev_mask_drop_prob
                        prev_output[zero_mask] = torch.zeros_like(prev_output[zero_mask])

            else:
                # iter mask (RITM)
                points, prev_output, prev_output_edges = self.find_next_n_points(
                    image,
                    gt_mask,
                    points,
                    prev_output,
                    prev_output_edges,
                    num_iters,
                    not validation
                )

                # net_input = torch.cat((image, prev_output,prev_output_edges), dim=1) \
                #     if self.net.with_prev_mask else image
                
                net_input = torch.cat((image, prev_output, prev_output_edges), dim=1) \
                    if self.net.with_prev_mask and self.net.with_prev_edges \
                        else torch.cat((image, prev_output), dim=1) \
                            if self.net.with_prev_mask \
                                else image
                    
                    
                output = self._forward(self.net, net_input, points)

                loss = self.add_loss(
                    'instance_loss',
                    loss,
                    losses_logging,
                    validation,
                    lambda: (output['instances'], batch_data['instances']))
               
                # loss = self.add_loss(
                #     'hausdorff_loss',
                #     loss,
                #     losses_logging,
                #     validation,
                #     lambda: (output['edges'], batch_data['edges']))
                
                # loss = self.add_loss(
                #     'active_boundary_loss',
                #     loss,
                #     losses_logging,
                #     validation,
                #     lambda: (output['edges'], batch_data['edges'].squeeze(1)))

                loss = self.add_loss(
                    'edge_loss',
                    loss,
                    losses_logging,
                    validation,
                    lambda: (output['edges'], batch_data['edges'])) 
                
                loss = self.add_loss(
                    'instance_aux_loss',
                    loss,
                    losses_logging,
                    validation,
                    lambda: (output['instances_aux'], batch_data['instances']))

            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        m.update(*(output.get(x) for x in m.pred_outputs),
                                 *(batch_data[x] for x in m.gt_outputs))

        batch_data['points'] = points
        return loss, losses_logging, batch_data, output

    def find_next_n_points(self, image, gt_mask, points, prev_output,prev_output_edges,
                           num_points, eval_mode=False, grad=False):
        with torch.set_grad_enabled(grad):
            for _ in range(num_points):

                if eval_mode:
                    self.net.eval()

                # net_input = torch.cat((image, prev_output,prev_output_edges), dim=1) \
                #     if self.net.with_prev_mask else image
                
                net_input = torch.cat((image, prev_output, prev_output_edges), dim=1) \
                    if self.net.with_prev_mask and self.net.with_prev_edges \
                        else torch.cat((image, prev_output), dim=1) \
                            if self.net.with_prev_mask \
                                else image

                
                pred_output = self._forward(
                    self.net,
                    net_input,
                    points
                    )
                
                prev_output = torch.sigmoid(
                    pred_output['instances']
                    )
                prev_output_edges = torch.sigmoid(
                    pred_output['edges']
                    )
                
                 ##Simiulate next point/clicks
                points = get_next_points(prev_output, gt_mask, points)
                
                # p_output = self._forward(self.net,net_input,points)
                # prev_output,prev_edge = torch.sigmoid(p_output['instances']),torch.sigmoid(p_output['edges'])
                
                # points = get_next_edge_points(image,prev_output,prev_edge,gt_mask, points,
                #                                     save_interval=self.image_dump_interval > 0 and self.global_step % self.image_dump_interval == 0,
                #                                     save_path=self.cfg.VIS_PATH / 'masks_viz' / str(self.global_step))
                
                if eval_mode:
                    self.net.train()

            if self.net.with_prev_mask and self.prev_mask_drop_prob > 0 and num_points > 0:
                zero_mask = np.random.random(
                    size=prev_output.size(0)) < self.prev_mask_drop_prob
                prev_output[zero_mask] = \
                    torch.zeros_like(prev_output[zero_mask])
        return points, prev_output, prev_output_edges

    def _forward(self, model, net_input, points, *args, **kwargs):
        # handle autocast for automatic mixed precision
        if self.cfg.amp:
            with torch.cuda.amp.autocast():
                output = model(net_input, points, *args, **kwargs)
        else: # Input image and prev_mask and points/clicks to get disk maps
            output = model(net_input, points, *args, **kwargs)
            
            # if model.with_crf:
            #     visualize_predictions(net_input[:,:3,:,:], outputs['instances'], image_data['instances'])


        return output

    def add_loss(self, loss_name, total_loss, losses_logging, validation,
                 lambda_loss_inputs, iterloss_step=None, iterloss_weight=1):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            if loss_criterion is None:
                raise ValueError(f"Loss criterion for {loss_name} not found in the configuration.")
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)

            if iterloss_step is not None:
                losses_logging[
                    loss_name + f'_{iterloss_step}_{iterloss_weight}'
                ] = loss 
                loss = loss_weight * loss * iterloss_weight
            else:
                # iter mask (RITM)
                losses_logging[loss_name] = loss
                loss = loss_weight * loss

            total_loss = total_loss + loss

        return total_loss
    
    
    def add_dynamically_weighted_loss(self, loss_name, total_loss, losses_logging, validation,
                 lambda_loss_inputs, iterloss_step=None, iterloss_weight=1):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            if loss_criterion is None:
                raise ValueError(f"Loss criterion for {loss_name} not found in the configuration.")
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)
            
            if iterloss_step is not None:
                losses_logging[
                    loss_name + f'_{iterloss_step}_{iterloss_weight}'
                ] = loss 
                loss = loss_weight * loss * iterloss_weight
            else:
                # iter mask (RITM)
                # Dynamic weight adjustment: Scale based on the ratio of current loss to initial loss
                initial_loss_value = self.loss_cfg.get(loss_name + '_initial_value', 1.0)
                dyn_weight = min(initial_loss_value / (loss + 1e-12),5)
                dynamic_weighted_loss =  dyn_weight * loss  #loss_weight * iterloss_weight
                
                losses_logging[loss_name] = loss
                #weighted_loss = loss_weight * loss

            total_loss = total_loss + dynamic_weighted_loss

        return total_loss
    

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = splitted_batch_data['images']
        points = splitted_batch_data['points']
        instance_masks = splitted_batch_data['instances']

        gt_instance_masks = instance_masks.cpu().numpy()
        predicted_instance_masks = torch.sigmoid(outputs['instances']).detach().cpu().numpy()
        predicted_edge_masks = torch.sigmoid(outputs['edges']).detach().cpu().numpy() 
        predicted_aux_instance_masks = (
            torch.sigmoid(outputs['instances_aux']).detach().cpu().numpy() 
            if outputs['instances_aux'] is not None 
            else None
            )
        points = points.detach().cpu().numpy()

        image_blob, points = images[0], points[0]
        gt_mask = np.squeeze(gt_instance_masks[0], axis=0)
        predicted_mask = np.squeeze(predicted_instance_masks[0], axis=0)
        predicted_edge = np.squeeze(predicted_edge_masks[0], axis=0)
        predicted_aux_mask = (
            np.squeeze(predicted_aux_instance_masks[0], axis=0) 
            if predicted_aux_instance_masks is not None 
            else None)

        image = image_blob.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))

        image_with_points = draw_points(image, points[:self.max_interactive_points], (0, 255, 0))
        image_with_points = draw_points(image_with_points, points[self.max_interactive_points:], (255, 0, 0))

        gt_mask[gt_mask < 0] = 0.25
        gt_mask = draw_probmap(gt_mask)
        predicted_mask = draw_probmap(predicted_mask)
        predicted_edge = draw_probmap(predicted_edge)
        if predicted_aux_mask is not None:
            predicted_aux_mask = draw_probmap(predicted_aux_mask)
            viz_image = np.hstack((image_with_points, gt_mask, predicted_mask, predicted_edge, predicted_aux_mask)).astype(np.uint8)
        else:
            viz_image = np.hstack((image_with_points, gt_mask, predicted_mask,predicted_edge)).astype(np.uint8)

        _save_image('instance_segmentation', viz_image[:, :, ::-1])

    def _load_weights(self, net):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                load_weights(net, self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            logger.info(f'Load checkpoint from path: {checkpoint_path}')
            load_weights(net, str(checkpoint_path))
        return net

    @property
    def is_master(self):
        return self.cfg.local_rank == 0


def get_next_points(pred, gt, points, pred_thresh=0.49):
    # Convert tensors to numpy arrays and compute error masks
    pred = pred.detach().cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    # False Negative (FN): GT is True(1) but prediction is False(0)
    fn_mask = np.logical_and(gt, pred < pred_thresh)
    # False Positive (FP): GT is False(0) but prediction is high(1)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    # Pad masks to handle edge cases in distance transform
    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    
    num_points = points.size(1) // 2  # Number of points per group (positive/negative)
    points = points.clone()  # Avoid modifying the original tensor

    for bindx in range(fn_mask.shape[0]):
        # Compute distance transforms (identify largest error regions)
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        # Determine dominant error type (FN or FP)
        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)
        is_positive = fn_max_dist > fp_max_dist
        
        # Identify inner region (central half of the largest error)
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            # Select a random point from candidates
            coords = indices[np.random.randint(0, len(indices))]
            order = max(points[bindx, :, 2].max(), 0) + 1
            if is_positive:
                loc = torch.argwhere(points[bindx, :num_points, 2] < 0)
                loc = loc[0, 0] if len(loc) > 0 else num_points - 1
                points[bindx, loc, 0] = float(coords[0])
                points[bindx, loc, 1] = float(coords[1])
                points[bindx, loc, 2] = float(order)
            else:
                loc = torch.argwhere(points[bindx, num_points:, 2] < 0)
                loc = loc[0, 0] + num_points if len(loc) > 0 else 2 * num_points - 1
                points[bindx, loc, 0] = float(coords[0])
                points[bindx, loc, 1] = float(coords[1])
                points[bindx, loc, 2] = float(order)

    return points

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import os


def visualize_masks(image,gt, pred, fn_mask, fp_mask, fn_dt, fp_dt, region_mask,
                    gt_edges, edge_pred, edge_fn_mask, edge_fp_mask, edge_fn_dt, 
                    edge_fp_dt, edge_mask, save_path, batch_idx):
    """Function to visualize and save segmentation and edge masks."""
    fig, axs = plt.subplots(2, 8, figsize=(18, 6))

    # Region Processing Visualization
    titles = ["Image","GT Mask", "Pred Mask", "FN Mask", "FP Mask", "FN Distance", "FP Distance", "Region Error Mask"]
    images = [image,gt, pred, fn_mask, fp_mask, fn_dt, fp_dt, region_mask]
    
    for i, (title, img) in enumerate(zip(titles, images)):
        axs[0, i].imshow(img, cmap="gray" if i < 5 else "jet")
        axs[0, i].set_title(title)

    # Edge Processing Visualization
    edge_titles = ["Image","GT Edge Mask", "Pred Edge Mask", "Edge FN Mask", "Edge FP Mask", "Edge FN Distance", "Edge FP Distance", "Edge Error Mask"]
    edge_images = [image,gt_edges, edge_pred, edge_fn_mask, edge_fp_mask, edge_fn_dt, edge_fp_dt, edge_mask]

    for i, (title, img) in enumerate(zip(edge_titles, edge_images)):
        axs[1, i].imshow(img, cmap="gray" if i < 5 else "jet")
        axs[1, i].set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"batch_{batch_idx}.png"))
    plt.close()
    
def get_next_points_with_edges(image,pred, edge_pred, gt, points, pred_thresh=0.49,
                               edge_thresh=0.49,save_interval=None, save_path="masks_vis"):

    # Convert tensors to numpy arrays
    pred = pred.detach().cpu().numpy()[:, 0, :, :]
    edge_pred = edge_pred.detach().cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5  # Convert GT mask to boolean

    # Compute False Negative (FN) and False Positive (FP) masks for segmentation
    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    # Compute Ground Truth (GT) edges
    dt_gt = np.array([ndimage.distance_transform_edt(gt_i) for gt_i in gt])
    gt_edges = np.logical_and(dt_gt <= 1, gt)

    # Compute FN and FP masks for edges (errors in predicted edges)
    edge_fn_mask = np.logical_and(gt_edges, edge_pred < edge_thresh)
    edge_fp_mask = np.logical_and(np.logical_not(gt_edges), edge_pred > edge_thresh)

    # Pad masks for distance transform computation
    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    edge_fn_mask = np.pad(edge_fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    edge_fp_mask = np.pad(edge_fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)

    num_points = points.size(1) // 2  # Number of points per group (positive/negative)
    points = points.clone()  # Avoid modifying the original tensor

    for bindx in range(fn_mask.shape[0]):
        # Compute distance transforms
        fn_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        edge_fn_dt = cv2.distanceTransform(edge_fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        edge_fp_dt = cv2.distanceTransform(edge_fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        # Determine dominant error type (FN or FP) for both region and edges
        fn_max_dist, fp_max_dist = np.max(fn_dt), np.max(fp_dt)
        edge_fn_max_dist, edge_fp_max_dist = np.max(edge_fn_dt), np.max(edge_fp_dt)

        is_region_positive = fn_max_dist > fp_max_dist
        is_edge_positive = edge_fn_max_dist > edge_fp_max_dist

        # Identify inner region of largest segmentation error
        region_dt = fn_dt if is_region_positive else fp_dt
        region_mask = region_dt > max(fn_max_dist, fp_max_dist) / 2.0

        # Identify inner region of largest edge error
        edge_dt = edge_fn_dt if is_edge_positive else edge_fp_dt
        edge_mask = edge_dt > max(edge_fn_max_dist, edge_fp_max_dist) / 2.0

        # Select points from region errors
        region_indices = np.argwhere(region_mask)
        edge_indices = np.argwhere(edge_mask)

        #  function to assign points
        def assign_point(coords, is_positive):
            order = max(points[bindx, :, 2].max(), 0) + 1
            if is_positive:
                loc = torch.argwhere(points[bindx, :num_points, 2] < 0)
                loc = loc[0, 0] if len(loc) > 0 else num_points - 1
            else:
                loc = torch.argwhere(points[bindx, num_points:, 2] < 0)
                loc = loc[0, 0] + num_points if len(loc) > 0 else 2 * num_points - 1
            points[bindx, loc, 0] = float(coords[0])
            points[bindx, loc, 1] = float(coords[1])
            points[bindx, loc, 2] = float(order)

        # select one region base and one edge base click
        if len(region_indices) > 0:
            assign_point(region_indices[np.random.randint(0, len(region_indices))], is_region_positive)

        if len(edge_indices) > 0:
            assign_point(edge_indices[np.random.randint(0, len(edge_indices))], is_edge_positive)

    image = (image[bindx].cpu().numpy()).transpose((1, 2, 0))
    if save_interval == True:
        # Visualization
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        visualize_masks(image,gt[bindx], pred[bindx], fn_mask[bindx][1:-1, 1:-1], fp_mask[bindx][1:-1, 1:-1], 
                        fn_dt, fp_dt, region_mask, gt_edges[bindx], edge_pred[bindx], 
                        edge_fn_mask[bindx][1:-1, 1:-1], edge_fp_mask[bindx][1:-1, 1:-1], 
                        edge_fn_dt, edge_fp_dt, edge_mask, save_path, bindx)

    return points


def get_next_edge_points(image,pred, edge_pred, gt, points, pred_thresh=0.49,
                               edge_thresh=0.49,save_interval=None, save_path="masks_vis"):

    # Convert tensors to numpy arrays
    pred = pred.detach().cpu().numpy()[:, 0, :, :]
    edge_pred = edge_pred.detach().cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5  # Convert GT mask to boolean

    # Compute False Negative (FN) and False Positive (FP) masks for segmentation
    fn_mask = np.logical_and(gt, pred < pred_thresh)
    #fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    # Compute Ground Truth (GT) edges
    dt_gt = np.array([ndimage.distance_transform_edt(gt_i) for gt_i in gt])
    gt_edges = np.logical_and(dt_gt <= 1, gt)

    # Compute FN and FP masks for edges (errors in predicted edges)
    edge_fn_mask = np.logical_and(gt_edges, edge_pred < edge_thresh)
    edge_fp_mask = np.logical_and(np.logical_not(gt_edges), edge_pred > edge_thresh)

    # Pad masks for distance transform computation
    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    #fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    edge_fn_mask = np.pad(edge_fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    edge_fp_mask = np.pad(edge_fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)

    num_points = points.size(1) // 2  # Number of points per group (positive/negative)
    points = points.clone()  # Avoid modifying the original tensor

    for bindx in range(fn_mask.shape[0]):
        # Compute distance transforms
        #fn_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        #fp_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        edge_fn_dt = cv2.distanceTransform(edge_fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        edge_fp_dt = cv2.distanceTransform(edge_fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        # Determine dominant error type (FN or FP) for both region and edges
        #fn_max_dist, fp_max_dist = np.max(fn_dt), np.max(fp_dt)
        edge_fn_max_dist, edge_fp_max_dist = np.max(edge_fn_dt), np.max(edge_fp_dt)

        #is_region_positive = fn_max_dist > fp_max_dist
        is_edge_positive = edge_fn_max_dist > edge_fp_max_dist

        # Identify inner region of largest segmentation error
        #region_dt = fn_dt if is_region_positive else fp_dt
        #region_mask = region_dt > max(fn_max_dist, fp_max_dist) / 2.0

        # Identify inner region of largest edge error
        edge_dt = edge_fn_dt if is_edge_positive else edge_fp_dt
        edge_mask = edge_dt > max(edge_fn_max_dist, edge_fp_max_dist) / 2.0

        # Select points from region errors
        #region_indices = np.argwhere(region_mask)
        edge_indices = np.argwhere(edge_mask)

        #  function to assign points
        def assign_point(coords, is_positive):
            order = max(points[bindx, :, 2].max(), 0) + 1
            if is_positive:
                loc = torch.argwhere(points[bindx, :num_points, 2] < 0)
                loc = loc[0, 0] if len(loc) > 0 else num_points - 1
            else:
                loc = torch.argwhere(points[bindx, num_points:, 2] < 0)
                loc = loc[0, 0] + num_points if len(loc) > 0 else 2 * num_points - 1
            points[bindx, loc, 0] = float(coords[0])
            points[bindx, loc, 1] = float(coords[1])
            points[bindx, loc, 2] = float(order)

        # select one region base and one edge base click
        #if len(region_indices) > 0:
        #    assign_point(region_indices[np.random.randint(0, len(region_indices))], is_region_positive)

        if len(edge_indices) > 0:
            assign_point(edge_indices[np.random.randint(0, len(edge_indices))], is_edge_positive)

    #image = (image[bindx].cpu().numpy()).transpose((1, 2, 0))
    # if save_interval == True:
    #     # Visualization
    #     os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    #     visualize_masks(image,gt[bindx], pred[bindx], fn_mask[bindx][1:-1, 1:-1], #fp_mask[bindx][1:-1, 1:-1], 
    #                     #fn_dt, fp_dt, region_mask, 
    #                     gt_edges[bindx], edge_pred[bindx], 
    #                     edge_fn_mask[bindx][1:-1, 1:-1], edge_fp_mask[bindx][1:-1, 1:-1], 
    #                     edge_fn_dt, edge_fp_dt, edge_mask, save_path, bindx)

    return points



def get_iou(pred, gt, pred_thresh=0.49):
    pred_mask = pred > pred_thresh
    gt_mask = gt > 0.5

    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return intersection / union


def load_weights(model, path_to_weights):
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict)
