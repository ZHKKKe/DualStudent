import os
import time
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from third_party.mean_teacher import data
from third_party.mean_teacher import mt_func
from third_party.mean_teacher.utils import *
from third_party.mean_teacher.data import NO_LABEL

from src import architectures, ramps, losses, cli, run_context, datasets
from dual_student import create_model, create_data_loaders, adjust_learning_rate,  validate
import dual_student


LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0


def train_epoch(train_loader, model_list, optimizer_list, epoch, log):
    global global_step

    meters = AverageMeterSet()

    # define criterions
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    residual_logit_criterion = losses.symmetric_mse_loss
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
        stabilization_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
        stabilization_criterion = losses.softmax_kl_loss

    for model in model_list:
        model.train()

    end = time.time()
    for i, (input_list, target) in enumerate(train_loader):
        meters.update('data_time', time.time() - end)

        for odx, optimizer in enumerate(optimizer_list):
            adjust_learning_rate(optimizer, epoch, i, len(train_loader))
            meters.update('lr_{0}'.format(odx), optimizer.param_groups[0]['lr'])

        input_var_list, nograd_input_var_list = [], []
        for idx, inp in enumerate(input_list):
            input_var_list.append(Variable(inp))
            nograd_input_var_list.append(Variable(inp, requires_grad=False, volatile=True))
        
        target_var = Variable(target.cuda(async=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        unlabeled_minibatch_size = minibatch_size - labeled_minibatch_size
        assert labeled_minibatch_size >= 0 and unlabeled_minibatch_size >= 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)
        meters.update('unlabeled_minibatch_size', unlabeled_minibatch_size)

        loss_list = []
        cls_v_list, nograd_cls_v_list = [], []
        cls_i_list, nograd_cls_i_list = [], []
        mask_list, nograd_mask_list = [], []
        class_logit_list, nograd_class_logit_list = [], []
        cons_logit_list = []
        in_cons_logit_list, tar_class_logit_list = [], []

        # for each student model
        for mdx, model in enumerate(model_list):
            # forward
            class_logit, cons_logit = model(input_var_list[mdx])
            nograd_class_logit, nograd_cons_logit = model(nograd_input_var_list[mdx])

            # calculate - res_loss, class_loss, consistency_loss - inside each student model
            res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
            meters.update('{0}_res_loss'.format(mdx), res_loss.data[0])

            class_loss = class_criterion(class_logit, target_var) / minibatch_size
            meters.update('{0}_class_loss'.format(mdx), res_loss.data[0])

            consistency_weight = args.consistency_scale * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
            nograd_class_logit = Variable(nograd_class_logit.detach().data, requires_grad=False)
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, nograd_class_logit) / minibatch_size
            meters.update('{0}_cons_loss'.format(mdx), consistency_loss.data[0])

            loss = class_loss + res_loss + consistency_loss
            loss_list.append(loss)

            # store variables for calculating the stabilization loss
            cls_v, cls_i = torch.max(F.softmax(class_logit, dim=1), dim=1)
            nograd_cls_v, nograd_cls_i = torch.max(F.softmax(nograd_class_logit, dim=1), dim=1)
            cls_v_list.append(cls_v)
            cls_i_list.append(cls_i.data.cpu().numpy())
            nograd_cls_v_list.append(nograd_cls_v)
            nograd_cls_i_list.append(nograd_cls_i.data.cpu().numpy())

            mask_raw = torch.max(F.softmax(class_logit, dim=1), 1)[0]
            mask = (mask_raw > args.stable_threshold)
            nograd_mask_raw = torch.max(F.softmax(nograd_class_logit, dim=1), 1)[0]
            nograd_mask = (nograd_mask_raw > args.stable_threshold)
            mask_list.append(mask.data.cpu().numpy())
            nograd_mask_list.append(nograd_mask.data.cpu().numpy())

            class_logit_list.append(class_logit)
            cons_logit_list.append(cons_logit)
            nograd_class_logit_list.append(nograd_class_logit)

            in_cons_logit = Variable(cons_logit.detach().data, requires_grad=False)
            in_cons_logit_list.append(in_cons_logit)

            tar_class_logit = Variable(class_logit.clone().detach().data, requires_grad=False)
            tar_class_logit_list.append(tar_class_logit)

        # calculate stablization weight
        stabilization_weight = args.stabilization_scale * ramps.sigmoid_rampup(epoch, args.stabilization_rampup)
        if not args.exclude_unlabeled:
            stabilization_weight = (unlabeled_minibatch_size / minibatch_size) * stabilization_weight

        model_idx = np.arange(0, len(model_list))
        np.random.shuffle(model_idx)

        for idx in range(0, len(model_idx)):
            if idx % 2 != 0:
                continue
            
            # l and r construct Dual Student
            l_mdx, r_mdx = model_idx[idx], model_idx[idx + 1]

            for sdx in range(0, minibatch_size):
                l_stable = False
                # unstable: do not satisfy the 2nd condition
                if mask_list[l_mdx][sdx] == 0 and nograd_mask_list[l_mdx][sdx] == 0:
                    tar_class_logit_list[l_mdx][sdx, ...] = in_cons_logit_list[r_mdx][sdx, ...]
                # unstable: do not satisfy the 1st condition
                elif cls_i_list[l_mdx][sdx] != nograd_cls_i_list[l_mdx][sdx]:
                    tar_class_logit_list[l_mdx][sdx, ...] = in_cons_logit_list[r_mdx][sdx, ...]
                else:
                    l_stable = True
                
                r_stable = False
                # unstable: do not satisfy the 2nd condition
                if mask_list[r_mdx][sdx] == 0 and nograd_mask_list[r_mdx][sdx] == 0:
                    tar_class_logit_list[r_mdx][sdx, ...] = in_cons_logit_list[l_mdx][sdx, ...]
                # unstable: do not satisfy the 1st condition
                elif cls_i_list[r_mdx][sdx] != nograd_cls_i_list[r_mdx][sdx]:
                    tar_class_logit_list[r_mdx][sdx, ...] = in_cons_logit_list[l_mdx][sdx, ...]
                else:
                    r_stable = True
            
            # calculate stability if both l and r models are stable for a sample
            if l_stable and r_stable:
                l_sample_cons = consistency_criterion(cons_logit_list[l_mdx][sdx:sdx+1, ...], nograd_class_logit_list[r_mdx][sdx:sdx+1, ...])
                r_sample_cons = consistency_criterion(cons_logit_list[r_mdx][sdx:sdx+1, ...], nograd_class_logit_list[l_mdx][sdx:sdx+1, ...])
                # loss: l -> r
                if l_sample_cons.data.cpu().numpy()[0] < r_sample_cons.data.cpu().numpy()[0]:
                    tar_class_logit_list[r_mdx][sdx, ...] = in_cons_logit_list[l_mdx][sdx, ...]
                # loss: r -> l
                elif l_sample_cons.data.cpu().numpy()[0] > r_sample_cons.data.cpu().numpy()[0]:
                    tar_class_logit_list[l_mdx][sdx, ...] = in_cons_logit_list[r_mdx][sdx, ...]

            if args.exclude_unlabeled:
                l_stabilization_loss = stabilization_weight * stabilization_criterion(cons_logit_list[l_mdx], tar_class_logit_list[r_mdx]) / minibatch_size
                r_stabilization_loss = stabilization_weight * stabilization_criterion(cons_logit_list[r_mdx], tar_class_logit_list[l_mdx]) / minibatch_size
            else:
                for sdx in range(unlabeled_minibatch_size, minibatch_size):
                    tar_class_logit_list[l_mdx][sdx, ...] = in_cons_logit_list[r_mdx][sdx, ...]
                    tar_class_logit_list[r_mdx][sdx, ...] = in_cons_logit_list[l_mdx][sdx, ...]
                
                l_stabilization_loss = stabilization_weight * stabilization_criterion(cons_logit_list[l_mdx], tar_class_logit_list[r_mdx]) / unlabeled_minibatch_size
                r_stabilization_loss = stabilization_weight * stabilization_criterion(cons_logit_list[r_mdx], tar_class_logit_list[l_mdx]) / unlabeled_minibatch_size

            meters.update('{0}_stable_loss'.format(l_mdx), l_stabilization_loss.data[0])
            meters.update('{0}_stable_loss'.format(r_mdx), r_stabilization_loss.data[0])
            
            loss_list[l_mdx] = loss_list[l_mdx] + l_stabilization_loss
            loss_list[r_mdx] = loss_list[r_mdx] + r_stabilization_loss

            meters.update('{0}_loss'.format(l_mdx), loss_list[l_mdx].data[0])
            meters.update('{0}_loss'.format(r_mdx), loss_list[r_mdx].data[0])

        for mdx, model in enumerate(model_list):
            # calculate prec
            prec = mt_func.accuracy(class_logit_list[mdx].data, target_var.data, topk=(1, ))[0]
            meters.update('{0}_top1'.format(mdx), prec[0], labeled_minibatch_size)

            # backward and update
            optimizer_list[mdx].zero_grad()
            loss_list[mdx].backward()
            optimizer_list[mdx].step()

        # record
        global_step += 1
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info('Epoch: [{0}][{1}/{2}]\t'
                     'Batch-T {meters[batch_time]:.3f}\t'
                     .format(epoch, i, len(train_loader), meters=meters))

            for mdx, model in enumerate(model_list):
                cur_class_loss = meters['{0}_class_loss'.format(mdx)].val
                avg_class_loss = meters['{0}_class_loss'.format(mdx)].avg
                cur_res_loss = meters['{0}_res_loss'.format(mdx)].val
                avg_res_loss = meters['{0}_res_loss'.format(mdx)].avg
                cur_cons_loss = meters['{0}_cons_loss'.format(mdx)].val
                avg_cons_loss = meters['{0}_cons_loss'.format(mdx)].avg
                cur_stable_loss = meters['{0}_stable_loss'.format(mdx)].val
                avg_stable_loss = meters['{0}_stable_loss'.format(mdx)].avg
                cur_top1_acc = meters['{0}_top1'.format(mdx)].val
                avg_top1_acc = meters['{0}_top1'.format(mdx)].avg
                LOG.info('model-{0}: Class {1:.4f}({2:.4f})\tRes {3:.4f}({4:.4f})\tCons {5:.4f}({6:.4f})\t'
                         'Stable {7:.4f}({8:.4f})\tPrec@1 {9:.3f}({10:.3f})\t'.format(
                             mdx, cur_class_loss, avg_class_loss, cur_res_loss, avg_res_loss, cur_cons_loss, 
                             avg_cons_loss, cur_stable_loss, avg_stable_loss, cur_top1_acc, avg_top1_acc))

            LOG.info('\n')
            log.record(epoch + i / len(train_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()})


def main(context):
    global best_prec1
    global global_step

    # set variable 'args' in the file 'dual_student.py'
    dual_student.args = args

    # create loggers
    checkpoint_path = context.transient_dir
    training_log = context.create_train_log('training')
    validate_logs = []
    for mdx in range(0, args.model_num):
        validate_logs.append(context.create_train_log('{0}_validation'.format(mdx)))

    # create dataloaders
    dataset_config = datasets.__dict__[args.dataset](tnum=args.model_num)
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)

    # create models and optimizers
    model_list, optimizer_list = [], []
    for mdx in range(0, args.model_num):
        model = create_model(name=str(mdx), num_classes=num_classes)
        LOG.info(parameters_string(model))
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)

        model_list.append(model)
        optimizer_list.append(optimizer)

    # restore saved checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), '=> no checkpoint found at: {}'.format(args.resume)
        LOG.info('=> loading checkpoint: {}'.format(args.resume))

        checkpoint = torch.load(args.resume)

        # globel parameters
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        
        # models and optimizers
        for mdx, model in enumerate(model_list):
            model.load_state_dict(checkpoint['{0}_model'.format(mdx)])
        for mdx, optimizer in enumerate(optimizer_list):
            optimizer.load_state_dict(checkpoint['{0}_optimizer'.format(mdx)])

        LOG.info('=> loaded checkpoint {} (epoch {})'.format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    # validation
    if args.validation:
        prec1_list =[]
        for mdx, model in enumerate(model_list):
            LOG.info('Validating the model-{0}: '.format(mdx))
            prec1 = validate(eval_loader, model, validate_logs[mdx], global_step, args.start_epoch)
            prec1_list.append(prec1)
        best_prec1 = np.max(np.asarray(prec1_list))
        LOG.info('Best top1 prediction: {0}'.format(best_prec1))
        return

    # training
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        train_epoch(train_loader, model_list, optimizer_list, epoch, training_log)
        LOG.info('--- training epoch in {} seconds ---'.format(time.time() - start_time))

        is_best = False
        if args.validation_epochs and (epoch + 1) % args.validation_epochs == 0:
            start_time = time.time()
            prec1_list =[]
            for mdx, model in enumerate(model_list):
                LOG.info('Validating the model-{0}: '.format(mdx))
                prec1 = validate(eval_loader, model, validate_logs[mdx], global_step, epoch + 1)
                prec1_list.append(prec1)

            LOG.info('--- validation in {} seconds ---'.format(time.time() - start_time))
            current_best_prec1 = np.max(np.asarray(prec1_list))
            is_best = current_best_prec1 > best_prec1
            best_prec1 = max(current_best_prec1, best_prec1)

        # save checkpoint
        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            checkpoint_dict = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'best_prec1': best_prec1,
                'arch': args.arch 
            }
            for mdx, model in enumerate(model_list):
                checkpoint_dict['{0}_model'.format(mdx)] = model.state_dict()
            for mdx, optimizer in enumerate(optimizer_list):
                checkpoint_dict['{0}_optimizer'.format(mdx)] = optimizer.state_dict()
            
            mt_func.save_checkpoint(checkpoint_dict, is_best, checkpoint_path, epoch + 1)

    LOG.info('Best top1 prediction: {0}'.format(best_prec1))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parser_commandline_args()
    main(run_context.RunContext(__file__, 0))
