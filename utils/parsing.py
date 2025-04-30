import os
import yaml
import random
import argparse

import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--cls_data', type=str, default='/home/saemeechoi/LETSUR_project/snu/ConvRNN/data/datasets/snubh_cls_data.csv')
    parser.add_argument('--pred_data', type=str, default='/home/saemeechoi/LETSUR_project/snu/ConvRNN/data/datasets/censored_snubh_pred_data.csv')
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpus', type=str, default='1')
    parser.add_argument('--split', type=str, default='0')
    parser.add_argument('--slices_select', type=str, default='center')
    parser.add_argument('--view', type=str, default='axial', help='axial | sagittal')
    parser.add_argument('--task', type=str, default='PRED', help='CLS | PRED | MTL | fine-tuning')
    parser.add_argument('--max_followup', type=int, default=10)
    parser.add_argument('--channels_num_2d', type=int, default=3)
    parser.add_argument('--sort_slice', action='store_true')
    parser.add_argument('--classifier', type=str, default='fc', help='fc | gru')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--nashmtl', action='store_true')
    parser.add_argument('--auto-weight', action='store_true')
    parser.add_argument('--adjust_random', type=str, default='overlap')
    parser.add_argument('--ensemble', type=int, default=3)
    parser.add_argument('--heads_balance', type=int, default=0, help='0 | 1 | 2')
    parser.add_argument('--bn_order', type=str, default='bn_relu', help='bn_relu | relu_bn')
    parser.add_argument('--saving_model_metric', type=str, default='cindex', help='cindex | cls_auc')
    parser.add_argument('--save_roc_curve', action='store_true')

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for k, v in config.items():
            args.__setattr__(k, v)

    print("\n==================================== Arguments ====================================\n")
    print('   Config file : %s' % (args.config))
    print()
    print('   Task : %s' % (args.task))
    print('   CLS Data Root : %s' % (args.cls_data))
    print('   PRED Data Root : %s' % (args.pred_data))
    print()
    print('   Slice Selection : %s' % args.slices_select)
    print('   Dimension : %s' % (args.dimension))
    print('   Classifier : %s' % (args.classifier))
    print('   Heads Banalcing : %s' % (args.heads_balance))
    print('   BN Ordering : %s' % (args.bn_order))
    print()
    print('   Input Image : %s' % args.bone_only)
    print('   Sorted Slice: %r' % args.sort_slice)
    print('   View : %s' % args.view)
    print('   Learning rate : %s' % args.lr)
    print('   Split : %s' % args.split)
    print()
    print('   GPU ID : %s' % args.gpus)
    print("\n=================================================================================\n")
        
    if args.channels_num_2d == 3:
        if args.sort_slice:        
            if args.classifier == 'fc':
                if args.hidden_dim != 256:
                    args.output = 'experiments/{}/{}_{}/{}/{}_hiddenDim{}_lr{}_{}loss_sortSlice_{}'.format(args.task, args.slices_select, args.dimension, args.bone_only, args.view, args.hidden_dim, args.lr, args.loss, args.split)
                else:        
                    args.output = 'experiments/{}/{}_{}/{}/{}_lr{}_{}loss_sortSlice_{}'.format(args.task, args.slices_select, args.dimension, args.bone_only, args.view, args.lr, args.loss, args.split)
                    if args.nashmtl:
                        if args.auto_weight:
                            args.output = 'experiments/{}/{}_{}/{}/{}_lr{}_{}loss_sortSlice_weightloss_nashmtl_{}'.format(args.task, args.slices_select, args.dimension, args.bone_only, args.view, args.lr, args.loss, args.split)
                        else:
                            if args.max_followup == 10:
                                if args.adjust_random == "overlap":
                                    if args.heads_balance == 0: # not balancing heads between PRED and CLS
                                        args.output = f'experiments/{args.task}/{args.slices_select}_{args.dimension}/{args.bone_only}/{args.view}_lr{args.lr}_{args.loss}loss_sortSlice_nashmtl_{args.split}_{args.saving_model_metric}'
                                    elif args.heads_balance == 1:
                                        args.output = f'experiments/{args.task}/{args.slices_select}_{args.dimension}/{args.bone_only}/{args.view}_lr{args.lr}_{args.loss}loss_sortSlice_nashmtl_headsBalance_{args.heads_balance}_{args.bn_order}_{args.split}'
                                    elif args.heads_balance == 2:
                                        args.output = f'experiments/{args.task}/{args.slices_select}_{args.dimension}/{args.bone_only}/{args.view}_lr{args.lr}_{args.loss}loss_sortSlice_nashmtl_headsBalance_{args.heads_balance}_{args.bn_order}_{args.split}'
                                else:
                                    args.output = 'experiments/{}/{}_{}/{}/{}_lr{}_{}loss_sortSlice_nashmtl_{}_{}'.format(args.task, args.slices_select, args.dimension, args.bone_only, args.view, args.lr, args.loss, args.adjust_random, args.split)
                            else:
                                if 'reclass' in args.pred_data:
                                    args.output = 'experiments/{}/{}_{}/{}/{}_lr{}_{}loss_sortSlice_nashmtl_reclass_{}'.format(args.task, args.slices_select, args.dimension, args.bone_only, args.view, args.lr, args.loss, args.split)
                                else:
                                    args.output = 'experiments/{}/{}_{}/{}/{}_lr{}_{}loss_sortSlice_nashmtl_maxup{}_{}'.format(args.task, args.slices_select, args.dimension, args.bone_only, args.view, args.lr, args.loss, args.max_followup, args.split)
                    else:
                        if args.auto_weight:
                            args.output = 'experiments/{}/{}_{}/{}/{}_lr{}_{}loss_sortSlice_weightloss_{}'.format(args.task, args.slices_select, args.dimension, args.bone_only, args.view, args.lr, args.loss, args.split)
                        else:
                            args.output = 'experiments/{}/{}_{}/{}/{}_lr{}_{}loss_sortSlice_{}'.format(args.task, args.slices_select, args.dimension, args.bone_only, args.view, args.lr, args.loss, args.split)
            elif args.classifier == 'gru':
                args.output = 'experiments/{}/{}_{}/{}/{}_gru_lr{}_{}loss_sortSlice_{}'.format(args.task, args.slices_select, args.dimension, args.bone_only, args.view, args.lr, args.loss, args.split)
        else:
            args.output = 'experiments/{}/{}_{}/{}/{}_lr{}_{}loss_{}'.format(args.task, args.slices_select, args.dimension, args.bone_only, args.view, args.lr, args.loss, args.split)
        # args.output = 'experiments/{}/{}_{}/{}/{}_batch{}_lr{}_{}loss_{}'.format(args.task, args.slices_select, args.dimension, args.bone_only, args.view, args.batch_size, args.lr, args.loss, args.split)
    else:
        args.output = 'experiments/{}/{}_{}/{}/{}_lr{}_{}loss_sliceNum{}_{}'.format(args.task, args.slices_select, args.dimension, args.bone_only, args.view, args.lr, args.loss, args.channels_num_2d, args.split)
    os.makedirs(args.output, exist_ok=True, mode=0o777)

    if args.task == 'CLS':
        args.enabled_tasks = (1, 0)
    elif args.task == 'PRED':
        args.enabled_tasks = (0, 1)
    elif args.task == 'MTL':
        args.enabled_tasks = (1, 1)
    elif args.task == 'fine-tuning':
        args.enabled_tasks = (0, 1)
    else:
        ImportError('Task must be one of [CLS, PRED, MTL, fine-tuning]')
        
    args.cls_enabled, args.pred_enabled = args.enabled_tasks
    
    if (args.task == 'fine-tuning') & (args.mode =='train'):
        args.weight = args.output.replace('fine-tuning', 'CLS')
    
    if args.slices_select == 'random' and args.dimension == '2D':
        args.dim_in = args.hidden_dim * args.channels_num_2d * 3
    else:
        args.dim_in = args.hidden_dim * 3
        
    args.dim_out = 2
    args.start_epoch = 0
    args.device = torch.device(f'cuda')
    
    return args

def parse_argument_data_preprocess():
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--data_root', default='', type=str, help='input nifti directory path')
    parser.add_argument('--file_type', default='.nii.gz', type=str, help='niftif file type (.nii or .nii.gz)')
    parser.add_argument('--gpu_id', default="4", type=str)
    parser.add_argument('--config', default=None, type=str, help='config file path')

    args = parser.parse_args()
    
    # Set config
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for k, v in config.items():
            args.__setattr__(k, v)    

    return args